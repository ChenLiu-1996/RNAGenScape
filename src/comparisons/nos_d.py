import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Union, Dict, Any


class NOS_D(nn.Module):
    """RNA discrete (mask-token) NOS baseline (NOS-D).

    Official NOS ([ngruver/NOS](https://github.com/ngruver/NOS)) also supports
    discrete MLM-style corruption (``model=mlm``) with the same hidden-state
    guidance interface. This RNA baseline masks tokens at rate ``t/T``, predicts
    originals with CE on corrupted sites, and applies property-gradient guidance
    in embedding space before rematerializing tokens. Vocabulary:
    PAD/A/G/C/T/U/N/[MASK].

    Paper: Protein Design with Guided Discrete Diffusion (NeurIPS 2023).
    """

    def __init__(
        self,
        seq_len: int = 150,
        hidden_dim: int = 128,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.1,
        max_timesteps: int = 1000,
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
        self.num_properties = num_properties

        # RNA vocabulary mapping
        self.rna_vocab = {'PAD': 0, 'A': 1, 'G': 2, 'C': 3, 'T': 4, 'U': 5, 'N': 6, '[MASK]': 7}
        self.idx_to_base = {0: 'PAD', 1: 'A', 2: 'G', 3: 'C', 4: 'T', 5: 'U', 6: 'N', 7: '[MASK]'}
        self.mask_id = len(self.rna_vocab) - 1
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
        self.token_pred_head = nn.Linear(hidden_dim, self.vocab_size)  # Predict tokens directly

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
        # NOTE: it won't work if the sequence contains [MASK] token.
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
            x: [B, L] token IDs
            timesteps: [B] timesteps
            mask: [B, L] attention mask (1 for valid tokens, 0 for padding)

        Returns:
            Dictionary with predictions
        """
        # Token embeddings
        x = self.token_embedding(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Time embedding
        time_emb = self.time_embedding(timesteps)

        # Transformer layers
        for layer in self.transformer_layers:
            x = layer(x, time_emb, mask)

        # Output
        x = self.output_norm(x)
        logits = self.token_pred_head(x)

        # Property prediction
        if return_properties:
            if mask is not None:
                # Pool over valid tokens only
                pooled = (x * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
            property_pred = self.property_head(pooled)
            return logits, property_pred

        return logits

    def add_noise(self, x_0, t):
        """
        Add discrete noise (token corruption) for forward diffusion process

        Args:
            x_0: Clean sequences [B, L] token IDs
            t: Timesteps [B]

        Returns:
            x_t: Corrupted sequences [B, L] token IDs
            corruption_mask: Which positions were corrupted [B, L]
        """
        batch_size, seq_len = x_0.shape

        # Linear corruption rate
        corruption_rate = t.float() / self.max_timesteps

        # Create corruption mask
        corruption_mask = torch.rand(batch_size, seq_len, device=x_0.device) < corruption_rate.unsqueeze(1)

        # Random tokens for corruption (avoid PAD token for corruption)
        #noise_tokens = torch.randint(1, self.vocab_size, (batch_size, seq_len), device=x_0.device)
        # Mask token for corruption
        noise_tokens = torch.full((batch_size, seq_len), self.mask_id, device=x_0.device)

        # Apply corruption
        x_t = torch.where(corruption_mask, noise_tokens, x_0)

        return x_t, corruption_mask

    def compute_loss(self, x_0, targets, mask=None, recon_weight=1.0):
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

        # Add discrete noise
        x_t, corruption_mask = self.add_noise(x_0, t)

        # Forward pass
        logits, property_pred = self.forward(x_t, t, mask)

        # Reconstruction loss: only on corrupted tokens
        recon_loss = F.cross_entropy(
            logits.view(-1, self.vocab_size),
            x_0.view(-1),
            reduction='none'
        ).view(batch_size, -1)

        # Mask to only corrupted positions
        recon_loss = (recon_loss * corruption_mask.float()).sum() / (corruption_mask.sum() + 1e-8)

        # Property prediction loss
        if targets is not None:
            prop_loss = F.mse_loss(property_pred, targets)

        total_loss = recon_weight * recon_loss + prop_loss

        return total_loss, recon_loss, prop_loss, property_pred

    @torch.no_grad()
    def sample(self, batch_size, num_steps=50, guidance_kwargs=None, mask=None, use_reveal_schedule=False):
        """
        Sample RNA sequences using guided discrete diffusion

        Args:
            batch_size: Number of sequences to generate
            num_steps: Number of denoising steps
            guidance_kwargs: Dict with guidance parameters
            mask: Attention mask for generated sequences

        Returns:
            Generated token sequences [B, L]
        """
        if guidance_kwargs is None:
            guidance_kwargs = {}

        # Start from random tokens (avoid PAD token)
        #x = torch.randint(1, self.vocab_size, (batch_size, self.seq_len), device=self.device)
        # Start from mask tokens
        x = torch.full((batch_size, self.seq_len), self.mask_id, device=self.device)

        # Denoising loop
        for i in reversed(range(num_steps)):
            # print(f"Denoising step {i} of {num_steps}")
            # print(f"x: {self.ids_to_sequence(x)}")

            t = torch.full((batch_size,), i * (self.max_timesteps // num_steps),
                          device=self.device, dtype=torch.long)

            if "target_values" in guidance_kwargs and self.num_properties > 0:
                x = self._guided_step(x, t, guidance_kwargs, mask, use_reveal_schedule)
            else:
                x = self._denoising_step(x, t, mask, use_reveal_schedule=use_reveal_schedule)

        return x

    def _denoising_step(self, x_t, t, mask=None, *, topk=5, temperature=1.0,
                    use_reveal_schedule=False):
        """
        Single denoising step without guidance (MASK-based NOS-D).
        - Only updates masked positions.
        - Samples from top-k filtered logits (excludes MASK).
        - Optionally reveals only a fraction of masked sites per step.

        Args:
            x_t: [B, L] current tokens (contains MASKs)
            t:   [B] timesteps
            mask: [B, L] attention mask
            topk: int, top-k per position before sampling (0 => no top-k)
            temperature: float temperature for sampling
            use_reveal_schedule: bool, if True, use reveal schedule
                r = self.reveal_schedule(t) -> fraction in [0,1] of masked positions to reveal this step.
        """
        B, L = x_t.shape
        masked = (x_t == self.mask_id)                      # [B, L] bool

        # Base logits
        logits, property_pred = self.forward(x_t, t, mask)

        # Add noise during intermediate steps for better sampling
        if t[0] > 0:
            logits = logits + 0.1 * torch.randn_like(logits)

        # Exclude MASK token from candidates at masked positions
        logits_masked = logits.clone()
        logits_masked[..., self.mask_id] = float('-inf')

        # Top-k filter (per position)
        if topk and topk > 0 and topk < self.vocab_size:
            v, ix = torch.topk(logits_masked, k=topk, dim=-1)         # [B,L,topk]
            filt = torch.full_like(logits_masked, float('-inf'))
            logits_masked = filt.scatter(-1, ix, v)

        # Softmax with temperature
        probs = torch.softmax(logits_masked / temperature, dim=-1)    # [B,L,V]

        # Sample proposal tokens
        sampled = torch.multinomial(probs.view(-1, self.vocab_size), 1).view(B, L)

        # Progressive reveal: optionally only unmask a fraction this step
        if use_reveal_schedule:
            # r_t in [0,1]; allow tensor or scalar
            r = self.reveal_schedule(t)                      # shape [B] or scalar
            if isinstance(r, (int, float)):
                r = torch.full((B,), float(r), device=x_t.device)
            r = r.view(B, 1).clamp(0.0, 1.0)           # [B,1]
            reveal_draw = torch.rand(B, L, device=x_t.device) < r     # [B,L] bool
            reveal_mask = masked & reveal_draw
        else:
            reveal_mask = masked

        # Apply updates only where we're revealing
        x_next = x_t.clone()
        x_next[reveal_mask] = sampled[reveal_mask]

        return x_next

    # Linearly increase reveal as time decreases
    def reveal_schedule(self, t):
        # t is [B], with larger = earlier/noisier
        # Map to fraction in (0,1]; e.g., reveal more as t->0
        T = float(self.max_timesteps - 1)
        frac = 1.0 - (t.float() / max(T, 1.0))
        return 0.2 + 0.8 * frac  # start at 0.2, end at 1.0

    def _guided_step(self, x_t, t, guidance_kwargs, mask, use_reveal_schedule):
        """Guided denoising step using property gradients"""
        step_size = guidance_kwargs.get("step_size", 1.0)
        stability_coef = guidance_kwargs.get("stability_coef", 0.01)
        target_values = guidance_kwargs["target_values"]

        # For discrete guidance, we need to work in embedding space then convert back
        with torch.enable_grad():
            # Convert to embeddings for gradient computation
            x_emb = self.token_embedding(x_t)
            x_emb = x_emb.clone().detach().requires_grad_(True)

            # Forward pass through rest of network
            x_emb_pos = self.pos_encoding(x_emb)
            time_emb = self.time_embedding(t)

            # Apply transformer layers
            for layer in self.transformer_layers:
                x_emb_pos = layer(x_emb_pos, time_emb, mask)

            x_emb_pos = self.output_norm(x_emb_pos)

            if self.num_properties > 0:
                # Property prediction for guidance
                if mask is not None:
                    pooled = (x_emb_pos * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
                else:
                    pooled = x_emb_pos.mean(dim=1)

                property_pred = self.property_head(pooled)

                # Convert target values to tensor
                if isinstance(target_values, (int, float)):
                    target_values = torch.full_like(property_pred, target_values)
                elif isinstance(target_values, (list, tuple)):
                    target_values = torch.tensor(target_values, device=self.device).expand_as(property_pred)

                # Compute guidance loss
                guidance_loss = F.mse_loss(property_pred, target_values)

                # Compute gradients w.r.t. embeddings
                grad = torch.autograd.grad(guidance_loss, x_emb, retain_graph=False)[0]

                # Apply guided update in embedding space
                x_emb_guided = x_emb - step_size * grad
                x_emb_guided = x_emb_guided + stability_coef * torch.randn_like(x_emb_guided)

                # Convert back to tokens using similarity to embedding matrix
                with torch.no_grad():
                    # Compute similarity to all token embeddings
                    token_similarities = F.cosine_similarity(
                        x_emb_guided.unsqueeze(-2),  # [B, L, 1, H]
                        self.token_embedding.weight.unsqueeze(0).unsqueeze(0),  # [1, 1, V, H]
                        dim=-1
                    )  # [B, L, V]

                    # Sample from similarities (with temperature)
                    temperature = 0.1
                    token_probs = F.softmax(token_similarities / temperature, dim=-1)
                    x_t = torch.multinomial(token_probs.view(-1, self.vocab_size), 1).view(x_t.shape)

        # Regular denoising step
        return self._denoising_step(x_t, t, mask, use_reveal_schedule=use_reveal_schedule)

    def generate_sequences(self, batch_size, num_steps=50, guidance_kwargs=None, mask=None, use_reveal_schedule=False):
        """
        Generate RNA sequences as strings

        Returns:
            List of RNA sequence strings
        """
        tokens = self.sample(batch_size, num_steps, guidance_kwargs, mask, use_reveal_schedule)
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
    Single training step for NOS-D

    Args:
        model: NOSD model
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
    model = NOS_D(seq_len=seq_len, num_properties=1, device="cpu")
    sequences = torch.randint(1, model.vocab_size - 1, (batch_size, seq_len))
    targets = torch.randn(batch_size, 1)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    total, recon, prop, pred = model.compute_loss(sequences, targets, mask)
    assert total.ndim == 0 and pred.shape == (batch_size, 1)

    tokens = model.sample(batch_size=batch_size, num_steps=5, use_reveal_schedule=True)
    assert tokens.shape == (batch_size, seq_len)

    guided = model.sample(
        batch_size=batch_size,
        num_steps=5,
        guidance_kwargs={"step_size": 1.0, "stability_coef": 0.01, "target_values": [0.5]},
        use_reveal_schedule=True,
    )
    assert guided.shape == (batch_size, seq_len)
    print("NOS_D unit tests passed.")
