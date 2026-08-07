import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiffAb(nn.Module):
    """DiffAb designs sequences by iterative denoising under property guidance.

    Originally developed for antigen-conditioned antibody CDR (and structure) design. This RNA
    adaptation applies guided discrete diffusion and property-aware candidate selection to
    nucleotide sequences.

    Paper: Antigen-Specific Antibody Design and Optimization with Diffusion-Based Generative Models for
    Protein Structures (NeurIPS 2022)
    Github: https://github.com/luost26/diffab
    """
    def __init__(self,
                 vocab_size=4,
                 hidden_dim=128,
                 num_layers=2,
                 num_heads=4,
                 seq_len=124,
                 num_timesteps=100,
                 dropout=0.1,
                 num_properties: int = 1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.num_timesteps = num_timesteps
        self.seq_len = seq_len
        self.num_properties = num_properties
        self.device = device

        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_encoding = PositionalEncoding(hidden_dim, seq_len)
        self.time_embedding = TimeEmbedding(hidden_dim)

        self.transformer_layers = nn.ModuleList([
            DiffAbTransformerBlock(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])

        self.output_norm = nn.LayerNorm(hidden_dim)
        self.output_projection = nn.Linear(hidden_dim, vocab_size)

        self.property_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_properties)
        ).to(self.device)

        self.register_buffer('betas', self._make_beta_schedule())
        self.register_buffer('alphas', 1 - self.betas)
        self.register_buffer('alpha_bars', torch.cumprod(self.alphas, dim=0))

        self._initialize_weights()

    def _make_beta_schedule(self):
        beta_start = 0.0001
        beta_end = 0.02
        return torch.linspace(beta_start, beta_end, self.num_timesteps)

    def _initialize_weights(self):
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

    def forward(self, x, timesteps, mask=None, return_properties=False):
        x = self.token_embedding(x)
        x = self.pos_encoding(x)
        time_emb = self.time_embedding(timesteps)

        for layer in self.transformer_layers:
            x = layer(x, time_emb, mask)

        x = self.output_norm(x)
        logits = self.output_projection(x)

        if return_properties:
            if mask is not None:
                masked_x = x * mask.unsqueeze(-1).float()
                pooled = masked_x.sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp(min=1)
            else:
                pooled = x.mean(dim=1)
            properties = self.property_head(pooled)
            return logits, properties

        return logits

    def predict_property(self, sequences):
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)

        sequences = sequences.to(self.device)
        _, properties = self.forward(sequences, torch.zeros(sequences.shape[0]).to(sequences.device), return_properties=True)
        return properties.squeeze(-1)

    def add_noise(self, x_0, t):
        if len(x_0.shape) == 1:
            x_0 = x_0.unsqueeze(0)
        batch_size, seq_len = x_0.shape

        t = t.to(device=self.alpha_bars.device, dtype=torch.long)
        alpha_bar_t = self.alpha_bars[t].unsqueeze(1)
        noise_prob = 1 - alpha_bar_t

        noise_mask = torch.rand(batch_size, seq_len, device=x_0.device) < noise_prob.to(x_0.device)
        noise_tokens = torch.randint(0, self.vocab_size, (batch_size, seq_len), device=x_0.device)
        x_t = torch.where(noise_mask, noise_tokens, x_0)

        return x_t, noise_mask

    def categorical_posterior(self, x_t, x_0_probs, t):
        batch_size, seq_len = x_t.shape

        if t.dim() == 0:
            t = t.unsqueeze(0).expand(batch_size)
        t = t.to(device=self.alpha_bars.device, dtype=torch.long)

        alpha_bar_t = self.alpha_bars[t].unsqueeze(1).unsqueeze(2)
        alpha_bar_t_prev = torch.where(
            t > 0,
            self.alpha_bars[(t - 1).clamp(min=0)],
            torch.ones_like(self.alpha_bars[0]),
        ).unsqueeze(1).unsqueeze(2)
        alpha_bar_t = alpha_bar_t.to(x_t.device)
        alpha_bar_t_prev = alpha_bar_t_prev.to(x_t.device)

        x_t_onehot = F.one_hot(x_t, self.vocab_size).float()

        q_t = alpha_bar_t * x_t_onehot + (1 - alpha_bar_t) / self.vocab_size
        q_t_prev = alpha_bar_t_prev * x_t_onehot + (1 - alpha_bar_t_prev) / self.vocab_size

        q_t = q_t.clamp(min=1e-8, max=1.0)
        q_t_prev = q_t_prev.clamp(min=1e-8, max=1.0)
        x_0_probs = x_0_probs.clamp(min=1e-8, max=1.0)

        posterior = (q_t_prev / q_t) * x_0_probs
        posterior = posterior.clamp(min=1e-8)
        posterior = posterior / posterior.sum(dim=-1, keepdim=True)

        return posterior

    def denoise_step(self, x_t, t_tensor, mask=None, temperature=1.0):
        logits = self.forward(x_t, t_tensor, mask)

        if temperature > 0:
            posterior = self.categorical_posterior(x_t, F.softmax(logits, dim=-1), t_tensor)
            x_prev = torch.multinomial(posterior.view(-1, self.vocab_size), 1).view(x_t.shape)
        else:
            x_prev = torch.argmax(logits, dim=-1)

        return x_prev

    def compute_loss(self, x_0, targets=None, mask=None):
        batch_size = x_0.shape[0]
        t = torch.randint(0, self.num_timesteps, (batch_size,), device=x_0.device)

        x_t, _ = self.add_noise(x_0, t)
        logits, pred_properties = self.forward(x_t, t, mask, return_properties=True)

        c_denoised = F.softmax(logits, dim=-1)

        x_0_onehot = F.one_hot(x_0, self.vocab_size).float()
        post_true = self.categorical_posterior(x_t, x_0_onehot, t)
        post_pred = self.categorical_posterior(x_t, c_denoised, t)

        log_post_pred = torch.log(post_pred + 1e-8)
        kldiv = F.kl_div(log_post_pred, post_true, reduction='none', log_target=False).sum(dim=-1)

        if mask is not None:
            loss_seq = (kldiv * mask).sum() / (mask.sum().float() + 1e-8)
        else:
            loss_seq = kldiv.mean()

        losses = {"recon_loss": loss_seq}

        if targets is not None:
            prop_loss = F.mse_loss(pred_properties.view(-1), targets.view(-1))
            losses["property_loss"] = prop_loss

        return losses, pred_properties

    @torch.no_grad()
    def sample(self, shape, device, num_steps=100, mask=None, condition=None):
        '''Unconditional generation.'''
        batch_size, seq_len = shape

        if condition is None:
            x = torch.randint(0, self.vocab_size, shape, device=device)
        else:
            ts = torch.full((batch_size,), num_steps, device=condition.device)
            x, _ = self.add_noise(condition, ts)

        for i in reversed(range(num_steps)):
            t_tensor = torch.full((batch_size,), i * (self.num_timesteps // num_steps), dtype=torch.long, device=device)

            x = self.denoise_step(x, t_tensor, mask, temperature=i / num_steps)
        return x

    @torch.no_grad()
    def optimize(self, sequences, device, target_direction="increase",
                 num_candidates=1, forward_steps=100, mask=None, verbose=False):
        '''Optimization starting from a sequence.'''
        if len(sequences.shape) == 1:
            sequences = sequences.unsqueeze(0)

        batch_size, seq_len = sequences.shape
        sequences = sequences.to(device)
        # Clamp into the trained schedule (valid indices are 0 .. num_timesteps-1).
        t_start_idx = min(int(forward_steps), self.num_timesteps - 1)
        t_start = torch.full((batch_size,), t_start_idx, device=device)
        x_noisy, _ = self.add_noise(sequences, t_start)

        traj = [x_noisy]
        step_scale = max(self.num_timesteps // max(forward_steps, 1), 1)
        for i in reversed(range(forward_steps)):
            t_idx = min(i * step_scale, self.num_timesteps - 1)
            t_tensor = torch.full((batch_size,), t_idx, dtype=torch.long, device=device)

            candidates_list = []
            for _ in range(num_candidates):
                candidate = self.denoise_step(traj[-1], t_tensor, mask, temperature=0.1)
                candidates_list.append(candidate)

            candidates = torch.stack(candidates_list, dim=1)

            scores_list = []
            for j in range(num_candidates):
                score = self.predict_property(candidates[:, j])
                scores_list.append(score)
            scores = torch.stack(scores_list, dim=1)

            if target_direction == "increase":
                best_idx = torch.argmax(scores, dim=1)
            else:
                best_idx = torch.argmin(scores, dim=1)

            batch_idx = torch.arange(batch_size, device=device)
            x_next = candidates[batch_idx, best_idx]

            traj.append(x_next.clone())

        traj = torch.stack(traj)
        return traj


class DiffAbTransformerBlock(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.self_attention = EquivariantMultiHeadAttention(hidden_dim, num_heads, dropout)
        self.ffn = FeedForwardNetwork(hidden_dim, dropout)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.time_conditioning = AdaptiveLayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_emb, mask=None):
        x_norm = self.time_conditioning(self.norm1(x), time_emb)
        attn_out = self.self_attention(x_norm, x_norm, x_norm, mask)
        x = x + self.dropout(attn_out)

        x_norm = self.norm2(x)
        ffn_out = self.ffn(x_norm)
        x = x + self.dropout(ffn_out)

        return x


class EquivariantMultiHeadAttention(nn.Module):

    def __init__(self, hidden_dim, num_heads, dropout=0.1):
        super().__init__()

        assert hidden_dim % num_heads == 0

        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, mask=None):
        batch_size, seq_len, _ = query.shape

        Q = self.q_proj(query)
        K = self.k_proj(key)
        V = self.v_proj(value)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)
            scores = scores.masked_fill(~mask, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_dim)
        out = self.out_proj(out)

        return out


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        self.linear1 = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.linear2 = nn.Linear(4 * hidden_dim, hidden_dim)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.linear2(x)
        return x


class AdaptiveLayerNorm(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.time_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * hidden_dim)
        )

    def forward(self, x, time_emb):
        time_out = self.time_proj(time_emb)
        scale, shift = time_out.chunk(2, dim=-1)

        scale = scale.unsqueeze(1)
        shift = shift.unsqueeze(1)

        return x * (1 + scale) + shift


class TimeEmbedding(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, timesteps):
        half_dim = self.hidden_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)

        emb = self.time_mlp(emb)

        return emb


class PositionalEncoding(nn.Module):
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


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 32, 7
    model = DiffAb(vocab_size=vocab_size, seq_len=seq_len, num_timesteps=50, device="cpu")
    sequences = torch.randint(0, vocab_size, (batch_size, seq_len))
    targets = torch.randn(batch_size)
    mask = torch.ones(batch_size, seq_len, dtype=torch.bool)

    losses, pred = model.compute_loss(sequences, targets=targets, mask=mask)
    assert "recon_loss" in losses and "property_loss" in losses
    assert pred.shape[0] == batch_size

    generated = model.sample((batch_size, seq_len), sequences.device, num_steps=5)
    assert generated.shape == (batch_size, seq_len)

    traj = model.optimize(sequences, sequences.device, forward_steps=5, num_candidates=2)
    assert traj.shape[0] == 6 and traj.shape[-2:] == (batch_size, seq_len)
    print("DiffAb unit tests passed.")
