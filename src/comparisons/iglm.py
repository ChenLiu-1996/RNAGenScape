import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers
import numpy as np
from einops import rearrange

class IgLM(nn.Module):
    """IgLM optimizes sequences by infilling masked spans and selecting property-favored candidates.

    Originally an antibody span-infilling language model. This RNA adaptation uses infilling-based
    candidate selection on nucleotide sequences.

    Paper: IgLM: Infilling language modeling for antibody sequence design (Cell Systems 2023)
    Github: https://github.com/Graylab/IgLM
    """
    def __init__(self,
                 vocab_size=10,
                 seq_len=512,
                 n_embd=128,
                 n_layer=2,
                 n_head=4,
                 dropout=0.1,
                 num_properties: int = 1,
                 device='cuda' if torch.cuda.is_available() else 'cpu'):
        super().__init__()

        self.vocab_size = vocab_size
        self.seq_len = seq_len
        self.n_embd = n_embd
        self.num_properties = num_properties
        self.device = device

        # Special tokens
        # Tokens: ['<pad>', 'A', 'G', 'C', 'T', 'U', 'N', 'CLS', 'SEP', 'MASK']
        self.pad_token_id = 0
        self.mask_token_id = vocab_size - 1
        self.sep_token_id = vocab_size - 2
        self.cls_token_id = vocab_size - 3

        # GPT-2 configuration
        max_position_embeddings = seq_len * 2  # Generous buffer for infilling format
        config = transformers.GPT2Config(
            vocab_size=vocab_size,
            n_positions=max_position_embeddings,
            n_embd=n_embd,
            n_layer=n_layer,
            n_head=n_head,
            n_inner=4 * n_embd,
            activation_function="gelu_new",
            resid_pdrop=dropout,
            embd_pdrop=dropout,
            attn_pdrop=dropout,
            layer_norm_epsilon=1e-5,
            initializer_range=0.02,
            summary_type="cls_index",
            summary_use_proj=True,
            summary_activation=None,
            summary_proj_to_labels=True,
            summary_first_dropout=0.1,
            use_cache=True,
            bos_token_id=None,
            eos_token_id=self.cls_token_id,
            sep_token_id=self.sep_token_id,
            pad_token_id=self.pad_token_id,
        )

        # Initialize the GPT-2 model
        self.model = transformers.GPT2LMHeadModel(config).to(self.device)
        self.max_position_embeddings = max_position_embeddings

        # Property prediction head
        self.property_head = nn.Sequential(
            nn.Linear(n_embd, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(32, num_properties)
        ).to(self.device)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize property head weights"""
        for module in self.property_head.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x, attention_mask=None, return_properties=True):
        """
        Forward pass

        Args:
            x: Token sequences [batch_size, seq_len]
            attention_mask: Attention mask [batch_size, seq_len]
            return_properties: Whether to return property predictions

        Returns:
            logits: [batch_size, seq_len, vocab_size]
            properties: [batch_size, 1] if return_properties=True
        """
        outputs = self.model(input_ids=x, attention_mask=attention_mask, output_hidden_states=True)
        lm_logits = outputs.logits

        if return_properties:
            hidden_states = outputs.hidden_states[-1]
            # Always use nucleotide token averaging for consistency
            valid_mask = (x >= 1) & (x <= 6)  # Only nucleotide tokens
            if attention_mask is not None:
                valid_mask = valid_mask & attention_mask.bool()

            masked_hidden = hidden_states * valid_mask.unsqueeze(-1)
            pooled = masked_hidden.sum(dim=1) / valid_mask.sum(dim=1, keepdim=True).clamp(min=1)
            properties = self.property_head(pooled)
            return lm_logits, properties

        return lm_logits

    def mask_span(self, seqs, start, end, append_span=False):
        """Create masked input following IgLM format"""
        if len(seqs.shape) == 1:
            seqs = seqs.unsqueeze(0)

        batch_size = seqs.shape[0]
        device = seqs.device

        mask_tokens = torch.full((batch_size, 1), self.mask_token_id, device=device, dtype=seqs.dtype)
        sep_tokens = torch.full((batch_size, 1), self.sep_token_id, device=device, dtype=seqs.dtype)

        if append_span:
            sequences_iglm = torch.cat([seqs[:, :start], mask_tokens, seqs[:, end:], sep_tokens, seqs[:, start:end]], dim=1)
        else:
            sequences_iglm = torch.cat([seqs[:, :start], mask_tokens, seqs[:, end:], sep_tokens], dim=1)
        return sequences_iglm

    def validate_generated_seq(self, input_ids):
        """
        Validate that generated input ids follow IgLM format
        ([MASK] before [SEP] and [CLS] and that there's one of each)
        """
        if isinstance(input_ids, torch.Tensor):
            input_ids = input_ids.cpu().numpy()

        mask_idx = np.where(input_ids == self.mask_token_id)[0]
        sep_idx = np.where(input_ids == self.sep_token_id)[0]
        cls_idx = np.where(input_ids == self.cls_token_id)[0]

        if len(mask_idx) != 1 or len(sep_idx) != 1 or len(cls_idx) != 1:
            return False

        mask_idx = mask_idx.squeeze()
        sep_idx = sep_idx.squeeze()
        cls_idx = cls_idx.squeeze()

        return (mask_idx < sep_idx) and (sep_idx < cls_idx)

    def iglm_to_infilled(self, token_seq, target_length=None):
        """
        Convert IgLM inputs to the infilled tokenized sequence without any special tokens.
        """
        token_seq = np.array(token_seq)
        sep_token_idx = np.nonzero(token_seq == self.sep_token_id)[0].item()
        cls_token_idx = np.nonzero(token_seq == self.cls_token_id)[0].min()
        mask_token_idx = np.nonzero(token_seq == self.mask_token_id)[0].item()

        infilled_seq = np.concatenate([
            token_seq[:mask_token_idx],
            token_seq[sep_token_idx + 1:cls_token_idx],
            token_seq[mask_token_idx + 1:sep_token_idx]
        ], axis=0)

        # Remove any special tokens (keep only nucleotide tokens 1-6: A,G,C,T,U,N)
        infilled_seq = infilled_seq[(infilled_seq >= 1) & (infilled_seq <= 6)]

        # Pad or truncate to target length if specified
        if target_length is not None:
            if len(infilled_seq) < target_length:
                # Pad with token 1 (A) if too short
                infilled_seq = np.concatenate([infilled_seq, np.ones(target_length - len(infilled_seq), dtype=infilled_seq.dtype)])
            elif len(infilled_seq) > target_length:
                # Truncate if too long
                infilled_seq = infilled_seq[:target_length]

        return infilled_seq

    def infilling_loss(self, pred_logits, targets, attention_mask):
        """
        Compute infilling loss

        Args:
            pred_logits: [B, L, V] predicted logits
            targets: [B, L] target tokens
            attention_mask: [B, L] attention mask
        """
        batch_size, seq_len, vocab_size = pred_logits.shape

        # Shift for next token prediction
        shift_logits = pred_logits[..., :-1, :].contiguous()
        shift_labels = targets[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        # Flatten for cross entropy
        loss = F.cross_entropy(
            shift_logits.view(-1, vocab_size),
            shift_labels.view(-1),
            reduction='none'
        )

        # Apply attention mask
        loss = loss.view(batch_size, -1) * shift_mask

        # Average over valid positions
        return loss.sum() / shift_mask.sum().clamp(min=1)

    def regression_loss(self, pred_properties, target_properties):
        """Compute property prediction loss"""
        return nn.functional.mse_loss(pred_properties.squeeze(-1), target_properties)

    def compute_loss(self, sequences, target_properties, mask_start, mask_end,
                     recon_weight=1.0):
        """
        Joint training loss for language modeling + property prediction

        Args:
            sequences: [B, L] original clean sequences
            target_properties: [B] target property values
            mask_start: Start position of mask
            mask_end: End position of mask
            recon_weight: Weight for language modeling loss

        Returns:
            total_loss: Total loss
            lm_loss: Language modeling loss
            prop_loss: Property prediction loss
            pred_properties: Predicted property values
        """
        # For property prediction: use masked sequence WITHOUT ground truth infill
        sequences_masked = self.mask_span(sequences, mask_start, mask_end, append_span=False)
        attention_mask_prop = (sequences_masked != self.pad_token_id).float()
        _, pred_properties = self.forward(sequences_masked, attention_mask_prop, return_properties=True)

        # For language modeling: use full IgLM format WITH ground truth infill
        sequences_iglm = self.mask_span(sequences, mask_start, mask_end, append_span=True)
        attention_mask_lm = (sequences_iglm != self.pad_token_id).float()
        lm_logits = self.forward(sequences_iglm, attention_mask_lm, return_properties=False)

        lm_loss = self.infilling_loss(lm_logits, sequences_iglm, attention_mask_lm)
        prop_loss = self.regression_loss(pred_properties, target_properties)
        total_loss = recon_weight * lm_loss + prop_loss
        return total_loss, lm_loss, prop_loss, pred_properties

    @torch.no_grad()
    def _generate(self, starting_tokens, num_to_generate, top_p, temperature, target_length=None):
        """
        Generate sequences following the IgLM methodology
        This performs deduplication, so the input is expected to have shape [1, L].
        """
        assert starting_tokens.shape[0] == 1
        device = next(self.parameters()).device
        starting_tokens = starting_tokens.to(device)
        decoded_seqs = []

        bad_words_ids = [[self.mask_token_id], [self.sep_token_id], [self.cls_token_id], [self.pad_token_id]]

        # Calculate safe max_length based on starting tokens and position embeddings
        starting_length = starting_tokens.shape[1]
        max_length = min(starting_length + 50, self.max_position_embeddings - 1)

        # Generate in batches to reduce sequential calls
        max_attempts = num_to_generate * 3  # Generate more to account for rejection
        attempts = 0

        while len(decoded_seqs) < num_to_generate and attempts < max_attempts:
            # Calculate how many more we need and batch size for this round
            remaining_needed = num_to_generate - len(decoded_seqs)
            batch_size = min(remaining_needed * 2, max_attempts - attempts, 16)  # Cap batch size

            # Expand starting tokens for batch generation
            batch_starting_tokens = starting_tokens.expand(batch_size, -1)

            seqs = self.model.generate(
                batch_starting_tokens,
                max_length=max_length,
                pad_token_id=self.pad_token_id,
                eos_token_id=self.cls_token_id,
                forced_eos_token_id=self.cls_token_id,
                bad_words_ids=bad_words_ids,
                do_sample=True,
                top_p=top_p,
                temperature=temperature
            ).detach().cpu().numpy()

            # Process each generated sequence
            for seq in seqs:
                if len(decoded_seqs) >= num_to_generate:
                    break

                if self.validate_generated_seq(seq):
                    decoded_tokens = self.iglm_to_infilled(seq, target_length=target_length)
                    decoded_seq = ''.join([str(t) for t in decoded_tokens])
                    decoded_seqs.append(decoded_seq)

            attempts += batch_size

        # If we still don't have enough, pad with the last valid sequence or generate deterministically
        while len(decoded_seqs) < num_to_generate:
            if decoded_seqs:
                decoded_seqs.append(decoded_seqs[-1])  # Duplicate last valid sequence
            else:
                # Fallback: generate one more with higher temperature
                seq = self.model.generate(
                    starting_tokens,
                    max_length=max_length,
                    pad_token_id=self.pad_token_id,
                    eos_token_id=self.cls_token_id,
                    forced_eos_token_id=self.cls_token_id,
                    bad_words_ids=bad_words_ids,
                    do_sample=True,
                    top_p=0.9,
                    temperature=1.5
                ).detach().cpu().numpy().squeeze(0)

                decoded_tokens = self.iglm_to_infilled(seq, target_length=target_length)
                decoded_seq = ''.join([str(t) for t in decoded_tokens])
                decoded_seqs.append(decoded_seq)

        return torch.stack([torch.tensor([int(c) for c in seq], device=device) for seq in decoded_seqs])

    @torch.no_grad()
    def infill(self, sequences, infill_range, num_to_generate=10, top_p=1, temperature=1):
        """
        Infill sequences following the IgLM methodology

        Args:
            sequence: [L] or [B, L] input sequence(s)
            infill_range: (start, end) tuple or list of tuples for each sequence in batch
            num_to_generate: Number of sequences to generate
            top_p: Top-p sampling parameter
            temperature: Sampling temperature

        Returns:
            List of infilled sequences
        """
        if sequences.dim() == 1:
            sequences = sequences.unsqueeze(0)
            single_seq = True
        else:
            single_seq = False

        sequences = sequences.to(next(self.parameters()).device)
        start, end = infill_range
        original_length = sequences.shape[1]

        masked_sequences = self.mask_span(sequences, start, end, append_span=False)

        all_seqs = []
        for masked_seq in masked_sequences:
            generated_seqs = self._generate(
                masked_seq.unsqueeze(0),
                num_to_generate=num_to_generate,
                top_p=top_p,
                temperature=temperature,
                target_length=original_length
            )
            all_seqs.append(generated_seqs)
        all_seqs = torch.stack(all_seqs)

        if single_seq:
            return all_seqs[0]
        return all_seqs

    def predict_property(self, sequence):
        """Predict property value for a sequence using regression head"""
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)

        sequence = sequence.to(next(self.parameters()).device)
        attention_mask = (sequence != self.pad_token_id).float()
        _, properties = self.forward(sequence, attention_mask, return_properties=True)
        return properties.squeeze(-1)

    @torch.no_grad()
    def optimize(self, sequences, target_direction="increase", span_start=None, span_end=None,
                 num_candidates=10, span_length=4):
        """
        Optimize sequences for desired properties:
        1. Generate many candidates via infilling
        2. Score with internal property_head
        3. Select best

        Args:
            sequences: [B, L] input sequences to optimize
            target_direction: "increase" or "decrease"
            span_start: Start position of span to optimize (if None, choose automatically)
            span_end: End position of span to optimize (if None, choose automatically)
            num_candidates: Number of candidates to generate
            span_length: Length of span if positions not specified
        """
        if len(sequences.shape) == 1:
            sequences = sequences.unsqueeze(0)

        sequences = sequences.to(next(self.parameters()).device)
        batch_size, seq_len = sequences.shape

        if span_start is None or span_end is None:
            span_start = seq_len // 4
            span_end = span_start + span_length
            span_end = min(span_end, seq_len)

        candidate_tensor = self.infill(sequences, (span_start, span_end), num_to_generate=num_candidates)
        scores = self.predict_property(rearrange(candidate_tensor, 'b n l -> (b n) l'))
        scores = rearrange(scores, '(b n) -> b n', b=batch_size)

        if target_direction == "increase":
            best_candidate_indices = torch.argmax(scores, dim=1)
        else:
            best_candidate_indices = torch.argmin(scores, dim=1)
        batch_idx = torch.arange(best_candidate_indices.shape[0], device=candidate_tensor.device)
        best_candidates = candidate_tensor[batch_idx, best_candidate_indices, :]

        return best_candidates

    @torch.no_grad()
    def evaluate(self, sequence):
        """
        Evaluate sequence using both language model perplexity and property prediction

        Args:
            sequence: [L] input sequence

        Returns:
            dict with perplexity and property prediction
        """
        if sequence.dim() == 1:
            sequence = sequence.unsqueeze(0)

        sequence = sequence.to(next(self.parameters()).device)
        attention_mask = (sequence != self.pad_token_id).float()

        outputs = self.model(input_ids=sequence, attention_mask=attention_mask, labels=sequence)
        perplexity = torch.exp(outputs.loss).item()

        property_pred = self.predict_property(sequence.squeeze())
        property_value = property_pred.item() if isinstance(property_pred, torch.Tensor) else property_pred

        return {
            'perplexity': perplexity,
            'property': property_value
        }


def training_step(model, batch, optimizer):
    """
    Single training step for NOS-C

    Args:
        model: IgLM model
        batch: Dict with 'sequences', 'targets', 'mask'
        optimizer: PyTorch optimizer

    Returns:
        Dictionary of loss values
    """
    sequences = batch['sequences']  # [B, L] RNA token IDs
    targets = batch.get('targets', None)  # [B, target_channels] property values

    seq_len = sequences.shape[1]
    mask_start = np.random.randint(0, seq_len)
    mask_len = np.random.randint(1, seq_len//2)
    mask_end = min(mask_start + mask_len, seq_len - 1)

    total_loss, lm_loss, prop_loss, pred_properties = model.compute_loss(
        sequences, targets, mask_start, mask_end, recon_weight=1.0)

    batch_loss_dict = {
        'loss': total_loss,
        'model_loss': lm_loss,
        'reg_loss': prop_loss,
    }

    # Backward pass
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()

    return {k: v.item() if torch.is_tensor(v) else v for k, v in batch_loss_dict.items()}


if __name__ == "__main__":
    torch.manual_seed(0)
    batch_size, seq_len, vocab_size = 2, 32, 10
    model = IgLM(vocab_size=vocab_size, seq_len=seq_len, device="cpu")
    sequences = torch.randint(1, 5, (batch_size, seq_len))
    properties = torch.randn(batch_size)

    total, lm_loss, prop_loss, pred = model.compute_loss(sequences, properties, 5, 10)
    assert total.ndim == 0 and pred.shape[0] == batch_size
    assert lm_loss.ndim == 0 and prop_loss.ndim == 0

    optimized = model.optimize(
        sequences,
        target_direction="increase",
        span_start=5,
        span_end=10,
        num_candidates=2,
    )
    assert optimized.shape == sequences.shape
    print("IgLM unit tests passed.")
