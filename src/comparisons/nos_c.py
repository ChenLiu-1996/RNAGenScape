import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any


class NOS_C(nn.Module):
    """RNA continuous (Gaussian-embedding) NOS baseline (NOS-C).

    Official NOS ([ngruver/NOS](https://github.com/ngruver/NOS)) trains sequence
    diffusion with Gaussian corruption in embedding space (``model=gaussian``)
    and guides sampling via gradients w.r.t. hidden states
    (``step_size``, ``stability_coef``). This RNA baseline mirrors that recipe
    on nucleotide tokens (PAD/A/G/C/T/U/N), with a compact Transformer denoiser
    and an attached property head used for guidance.

    Paper: Protein Design with Guided Discrete Diffusion (NeurIPS 2023).
    """

    def __init__(
        self,
        seq_len: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_timesteps: int = 1000,
        noise_scale: float = 5.0,
        num_properties: int = 1,
        device: Optional[Union[str, torch.device]] = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        elif isinstance(device, str):
            device = torch.device(device)
        self.device = device

        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.max_timesteps = max_timesteps
        self.noise_scale = noise_scale
        self.num_properties = num_properties

        # RNA vocabulary mapping
        self.rna_vocab = {'PAD': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4, 'U': 5, 'N': 6}
        self.idx_to_base = {0: 'PAD', 1: 'A', 2: 'G', 3: 'C', 4: 'T', 5: 'U', 6: 'N'}
        self.vocab_size = len(self.rna_vocab)

        # Token embedding
        self.token_embedding = nn.Embedding(self.vocab_size, hidden_dim)

        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, seq_len + 100)

        # Time embedding
        self.time_embedding = TimeEmbedding(hidden_dim)

        # Transformer backbone
        self.transformer_layers = nn.ModuleList([
            NOSTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        # Output layers
        self.output_norm = nn.LayerNorm(hidden_dim)
        self.noise_pred_head = nn.Linear(hidden_dim, hidden_dim)  # Predict noise in embedding space

        # Property prediction head
        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, num_properties)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def sequence_to_ids(self, sequences):
        """Convert RNA sequences to token IDs"""
        if isinstance(sequences, str):
            sequences = [sequences]

        batch_ids = []
        for seq in sequences:
            ids = [self.rna_vocab.get(base.upper(), 0) for base in seq]  # Default to PAD
            batch_ids.append(ids)

        return torch.tensor(batch_ids, dtype=torch.long, device=self.device)

    def ids_to_sequence(self, token_ids):
        """Convert token IDs back to RNA sequences"""
        if token_ids.dim() == 1:
            token_ids = token_ids.unsqueeze(0)

        sequences = []
        for ids in token_ids:
            seq = ''.join([self.idx_to_base.get(idx.item(), 'N') for idx in ids if idx.item() != 0])
            sequences.append(seq)

        return sequences if len(sequences) > 1 else sequences[0]

    def forward(self, x, timesteps, mask=None, return_properties=True):
        """
        Forward pass for denoising

        Args:
            x: [B, L, H] embeddings or [B, L] token IDs
            timesteps: [B] timesteps
            mask: [B, L] attention mask (1 for valid tokens, 0 for padding)

        Returns:
            Dictionary with predictions
        """
        if x.dim() == 2:  # Convert token IDs to embeddings
            x = self.token_embedding(x)

        if mask is not None:
            mask = mask.to(x.device)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Time embedding
        time_emb = self.time_embedding(timesteps)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, time_emb, mask)

        # Output
        x = self.output_norm(x)
        noise_pred = self.noise_pred_head(x)

        # Property prediction
        if return_properties:
            if mask is not None:
                # Pool over valid tokens only
                pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
            property_pred = self.property_head(pooled)
            return noise_pred, property_pred

        return noise_pred

    def add_noise(self, x_0, t):
        """
        Add Gaussian noise to embeddings for forward diffusion process

        Args:
            x_0: Clean sequences [B, L] token IDs or [B, L, H] embeddings
            t: Timesteps [B]

        Returns:
            x_t: Noisy embeddings [B, L, H]
            noise: Added noise [B, L, H]
        """
        if x_0.dim() == 2:  # Convert tokens to embeddings
            x_0 = self.token_embedding(x_0)

        # Linear noise schedule
        alpha_t = 1.0 - (t.float() / self.max_timesteps)
        alpha_t = alpha_t.view(-1, 1, 1)  # [B, 1, 1]

        # Sample noise
        noise = torch.randn_like(x_0) * self.noise_scale

        # Add noise: x_t = sqrt(alpha_t) * x_0 + sqrt(1 - alpha_t) * noise
        x_t = torch.sqrt(alpha_t) * x_0 + torch.sqrt(1 - alpha_t) * noise

        return x_t, noise

    def compute_loss(self, x_0, targets=None, mask=None, recon_weight=1.0, property_weight=0.1):
        """
        Compute training loss

        Args:
            x_0: Clean sequences [B, L] token IDs
            targets: Property targets [B, num_properties] (optional)
            mask: Attention mask [B, L]

        Returns:
            Dictionary of losses, property predictions
        """
        batch_size = x_0.shape[0]

        # Sample random timesteps
        t = torch.randint(0, self.max_timesteps, (batch_size,), device=x_0.device)

        # Add noise
        x_t, noise = self.add_noise(x_0, t)

        # Forward pass
        noise_pred, property_pred = self.forward(x_t, t, mask)

        # Reconstruction loss: predict the noise
        recon_loss = F.mse_loss(noise_pred, noise)

        # Property prediction loss
        if targets is not None:
            prop_loss = F.mse_loss(property_pred, targets)

        total_loss = recon_weight * recon_loss + property_weight * prop_loss

        return total_loss, recon_loss, prop_loss, property_pred

    def sample(self, batch_size, num_steps=10, hijack_step=None, guidance_kwargs=None, mask=None, return_traj=False):
        f"""
        Sample RNA sequences using guided diffusion

        Args:
            batch_size: Number of sequences to generate
            num_steps: Number of denoising steps
            hijack_step: If not None, early stop to return intermediate result.
            guidance_kwargs: Dict with guidance parameters
            mask: Attention mask for generated sequences
            return_traj: Whether to return trajectories

        Returns:
            Generated token sequences [B, L]
            traj: [N_steps, B, L]
        """
        if guidance_kwargs is None:
            guidance_kwargs = {}

        # Start from random noise in embedding space
        x = torch.randn(batch_size, self.seq_len, self.hidden_dim, device=self.device)

        trajs = []
        # Denoising loop
        for i in reversed(range(num_steps)):
            t = torch.full((batch_size,), i * (self.max_timesteps // num_steps),
                          device=self.device, dtype=torch.long)

            if "target_values" in guidance_kwargs and self.num_properties > 0:
                x = self._guided_step(x, t, guidance_kwargs, mask)
            else:
                x = self._denoising_step(x, t, mask)
            
            if return_traj:
                # Convert embeddings to tokens
                with torch.no_grad():
                    # Project embeddings to vocabulary space
                    token_logits = self.token_embedding.weight @ x.transpose(-2, -1)  # [V, H] @ [B, H, L] -> [B, V, L]
                    token_logits = token_logits.transpose(-2, -1)  # [B, L, V]
                    tokens = torch.argmax(token_logits, dim=-1)
                trajs.append(tokens[None, :])

            if hijack_step is not None and i == (num_steps - hijack_step):
                break

        # Convert embeddings to tokens
        with torch.no_grad():
            # Project embeddings to vocabulary space
            token_logits = self.token_embedding.weight @ x.transpose(-2, -1)  # [V, H] @ [B, H, L] -> [B, V, L]
            token_logits = token_logits.transpose(-2, -1)  # [B, L, V]
            tokens = torch.argmax(token_logits, dim=-1)

        if return_traj:
            trajs = torch.cat(trajs, dim=0) # [n_steps, B, L]

        #import pdb; pdb.set_trace()

        return tokens, trajs
        

    def _denoising_step(self, x_t, t, mask):
        """Single denoising step without guidance"""
        noise_pred, property_pred = self.forward(x_t, t, mask)

        # Reverse diffusion: x_0 = (x_t - sqrt(1-alpha_t) * noise) / sqrt(alpha_t)
        alpha_t = 1.0 - (t.float() / self.max_timesteps)
        alpha_t = alpha_t.view(-1, 1, 1)

        x_pred = (x_t - torch.sqrt(1 - alpha_t) * noise_pred) / torch.sqrt(alpha_t)
        return x_pred

    def _guided_step(self, x_t, t, guidance_kwargs, mask):
        """Guided denoising step using property gradients"""
        step_size = guidance_kwargs.get("step_size", 1.0)
        stability_coef = guidance_kwargs.get("stability_coef", 0.01)
        target_values = guidance_kwargs["target_values"]

        # Enable gradients for guidance
        x_t.requires_grad_(True)

        # Forward pass
        noise_pred, property_pred = self.forward(x_t, t, mask)

        if self.num_properties > 0:
            # Convert target values to tensor
            if isinstance(target_values, (int, float)):
                target_values = torch.full_like(property_pred, target_values)
            elif isinstance(target_values, (list, tuple)):
                target_values = torch.tensor(target_values, device=self.device).expand_as(property_pred)

            # Compute guidance loss
            guidance_loss = F.mse_loss(property_pred, target_values)

            # Compute gradients w.r.t. hidden states
            grad = torch.autograd.grad(guidance_loss, x_t, retain_graph=True)[0]

            # Apply guided update
            with torch.no_grad():
                x_t = x_t - step_size * grad
                # Add stability regularization
                x_t = x_t + stability_coef * torch.randn_like(x_t)

        # Detach and perform regular denoising step
        x_t = x_t.detach()
        return self._denoising_step(x_t, t, mask)

    def generate_sequences(self, batch_size, num_steps=50, guidance_kwargs=None, mask=None):
        """
        Generate RNA sequences as strings

        Returns:
            List of RNA sequence strings
        """
        tokens = self.sample(batch_size, num_steps, guidance_kwargs, mask)
        return self.ids_to_sequence(tokens)


class NOSTransformerBlock(nn.Module):
    """Transformer block with time conditioning for NOS"""

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()

        self.self_attention = nn.MultiheadAttention(
            hidden_dim, num_heads, dropout=dropout, batch_first=True
        )
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 4 * hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * hidden_dim, hidden_dim)
        )

        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        # Time conditioning via Adaptive Layer Normalization
        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb, mask=None):
        # Time conditioning via AdaLN
        time_out = self.time_proj(time_emb)
        scale, shift = time_out.chunk(2, dim=-1)

        # Self-attention with time conditioning
        x_norm = self.norm1(x)
        x_norm = x_norm * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

        # Convert mask to key_padding_mask format
        if mask is not None:
            key_padding_mask = ~mask.bool()  # True for positions to ignore
        else:
            key_padding_mask = None

        attn_out, _ = self.self_attention(x_norm, x_norm, x_norm, key_padding_mask=key_padding_mask)
        x = x + self.dropout(attn_out)

        # Feed-forward network
        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return x


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps"""

    def __init__(self, hidden_dim):
        super().__init__()
        self.hidden_dim = hidden_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, timesteps):
        # Create sinusoidal embeddings
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        return self.time_mlp(emb)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""

    def __init__(self, hidden_dim, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, hidden_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() *
                            (-math.log(10000.0) / hidden_dim))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(1)].unsqueeze(0)


def training_step(model, batch, optimizer):
    """
    Single training step for NOS-C

    Args:
        model: NOSC model
        batch: Dict with 'sequences', 'targets', 'mask'
        optimizer: PyTorch optimizer

    Returns:
        Dictionary of loss values
    """
    sequences = batch['sequences']  # [B, L] RNA token IDs
    targets = batch.get('targets', None)  # [B, num_properties] property values
    mask = batch.get('mask', None)  # [B, L] attention mask

    # Compute losses
    losses, property_pred = model.compute_loss(sequences, targets, mask)

    # Backward pass
    optimizer.zero_grad()
    losses['total_loss'].backward()
    optimizer.step()

    return {k: v.item() if torch.is_tensor(v) else v for k, v in losses.items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len = 2, 16
    model = NOS_C(seq_len=seq_len, num_properties=1, device="cpu")
    sequences = torch.randint(1, model.vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    total, recon, prop, pred = model.compute_loss(sequences, targets, mask)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)

    tokens, _traj = model.sample(batch_size=batch_size, num_steps=5, return_traj=False)
    assert tokens.shape == (batch_size, seq_len)

    guided, _ = model.sample(
        batch_size=batch_size,
        num_steps=5,
        guidance_kwargs={"step_size": 1.0, "stability_coef": 0.01, "target_values": [0.5]},
    )
    assert guided.shape == (batch_size, seq_len)
    print("NOS_C unit tests passed.")
