"""UTR-LM oracle regressors for RNAGenScape.

Trainable (via ``train_oracle.py`` only):

* ``UTRLM`` - official pretrained UTR-LM backbone + per-dataset linear probe.
  Weights come from the a96123155/UTR-LM ``.pkl`` (SISS), remapped into a
  MultiMolecule ``UtrLmModel`` shell with the official 10-token alphabet.
  Never uses TE/MRL hubs. Other trainable archs (e.g. Conv1d) live elsewhere.

Frozen zero-shot validators (never enter ``train_oracle.py``):

* ``UTRLM_TE`` / ``UTRLM_MRL`` - MultiMolecule TE/MRL finetunes, frozen only.
  Scoring: CLS -> sequence head (``mfe_head`` in the released weights).

Note: official ESM naming/vocab (size 10) differs from MultiMolecule HF hubs
(vocab 28). Trainable loading remaps layer weights 1:1 into MultiMolecule with
``vocab_size=10`` and the official alphabet; TE/MRL keep the HF tokenizer.
"""

from __future__ import annotations

import os
from collections import OrderedDict
from typing import Dict, List, Sequence, Union

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from multimolecule import RnaTokenizer, UtrLmConfig, UtrLmForPreTraining, UtrLmModel
from safetensors.torch import load_file


# Official pretrained pickle (repo-relative). Must NOT be TE/MRL finetunes.
UTRLM_PRETRAINED_RELPATH = "checkpoints/utrlm/utrlm_pretrained_siss_ep93.pkl"
# Frozen zero-shot only — never used by train_oracle.py.
UTRLM_TE_PRETRAINED = "multimolecule/utrlm-te_el"
UTRLM_MRL_PRETRAINED = "multimolecule/utrlm-mrl"
UTRLM_ORACLE_NAMES = frozenset({"UTRLM", "UTRLM_TE", "UTRLM_MRL"})
FROZEN_HF_ORACLES = frozenset({"UTRLM_TE", "UTRLM_MRL"})

# Official UTR-LM alphabet (a96123155/UTR-LM).
OFFICIAL_TOKENS = {
    "<pad>": 0,
    "<eos>": 1,
    "<unk>": 2,
    "A": 3,
    "G": 4,
    "C": 5,
    "T": 6,
    "<cls>": 7,
    "<mask>": 8,
    "<sep>": 9,
}
OFFICIAL_VOCAB_SIZE = len(OFFICIAL_TOKENS)

_BACKBONE_CACHE: dict[str, UtrLmModel] = {}
_PRETRAIN_CACHE: dict[str, UtrLmForPreTraining] = {}
_TOKENIZER_CACHE: dict[str, RnaTokenizer] = {}
_OFFICIAL_TOKENIZER = None


def repo_root_from_models() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))


def default_pretrained_pkl() -> str:
    return os.path.join(repo_root_from_models(), UTRLM_PRETRAINED_RELPATH)


def is_utrlm_oracle(model_or_name) -> bool:
    if isinstance(model_or_name, str):
        return model_or_name in UTRLM_ORACLE_NAMES
    return isinstance(model_or_name, (UTRLMRegressor, UTRLMFrozenHFOracle))


def is_frozen_hf_oracle(model_or_name) -> bool:
    if isinstance(model_or_name, str):
        return model_or_name in FROZEN_HF_ORACLES
    return isinstance(model_or_name, UTRLMFrozenHFOracle)


class OfficialUtrLmTokenizer:
    """Minimal tokenizer matching the official UTR-LM 10-token alphabet."""

    pad_token_id = OFFICIAL_TOKENS["<pad>"]
    cls_token_id = OFFICIAL_TOKENS["<cls>"]
    eos_token_id = OFFICIAL_TOKENS["<eos>"]
    unk_token_id = OFFICIAL_TOKENS["<unk>"]
    mask_token_id = OFFICIAL_TOKENS["<mask>"]

    def __call__(
        self,
        texts: Union[str, Sequence[str]],
        return_tensors: str = "pt",
        padding: bool = True,
    ) -> Dict[str, torch.Tensor]:
        if isinstance(texts, str):
            texts = [texts]
        rows: List[List[int]] = []
        for text in texts:
            ids = [self.cls_token_id]
            for ch in str(text).upper().replace("U", "T"):
                ids.append(OFFICIAL_TOKENS.get(ch, self.unk_token_id))
            ids.append(self.eos_token_id)
            rows.append(ids)
        max_len = max(len(r) for r in rows) if rows else 0
        input_ids = []
        attention_mask = []
        for row in rows:
            pad = max_len - len(row)
            input_ids.append(row + [self.pad_token_id] * pad)
            attention_mask.append([1] * len(row) + [0] * pad)
        if return_tensors != "pt":
            raise ValueError("OfficialUtrLmTokenizer only supports return_tensors='pt'")
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        }


def get_official_tokenizer() -> OfficialUtrLmTokenizer:
    global _OFFICIAL_TOKENIZER
    if _OFFICIAL_TOKENIZER is None:
        _OFFICIAL_TOKENIZER = OfficialUtrLmTokenizer()
    return _OFFICIAL_TOKENIZER


def get_hf_tokenizer(model_name: str) -> RnaTokenizer:
    if model_name not in _TOKENIZER_CACHE:
        _TOKENIZER_CACHE[model_name] = RnaTokenizer.from_pretrained(model_name)
    return _TOKENIZER_CACHE[model_name]


def _remap_hf_state_dict(state_dict: dict) -> dict:
    """Map HF ``model.*`` keys onto multimolecule ``utrlm.*`` module names."""
    remapped = {}
    for key, value in state_dict.items():
        if key.startswith("model."):
            remapped["utrlm." + key[len("model."):]] = value
        else:
            remapped[key] = value
    return remapped


def _strip_module_prefix(state_dict: dict) -> OrderedDict:
    out = OrderedDict()
    for key, value in state_dict.items():
        out[key[len("module."):] if key.startswith("module.") else key] = value
    return out


def remap_official_esm_to_utrlm(esm_state: dict) -> OrderedDict:
    """Map official ESM2-SISS ``.pkl`` keys onto MultiMolecule ``UtrLmModel`` keys.

    Layer / embedding tensors map 1:1 when ``vocab_size=10``. Rotary ``inv_freq``,
    LM / MFE / structure / contact heads are not transferred (probe / frozen TE-MRL
    paths do not need them on the backbone).
    """
    esm = _strip_module_prefix(esm_state)
    mapped: OrderedDict[str, torch.Tensor] = OrderedDict()
    mapped["embeddings.word_embeddings.weight"] = esm["embed_tokens.weight"]
    mapped["encoder.emb_layer_norm_after.weight"] = esm["emb_layer_norm_after.weight"]
    mapped["encoder.emb_layer_norm_after.bias"] = esm["emb_layer_norm_after.bias"]

    n_layers = max(
        int(k.split(".")[1])
        for k in esm
        if k.startswith("layers.") and k.split(".")[1].isdigit()
    ) + 1
    for i in range(n_layers):
        src = f"layers.{i}."
        dst = f"encoder.layer.{i}."
        mapped[dst + "attention.self.query.weight"] = esm[src + "self_attn.q_proj.weight"]
        mapped[dst + "attention.self.query.bias"] = esm[src + "self_attn.q_proj.bias"]
        mapped[dst + "attention.self.key.weight"] = esm[src + "self_attn.k_proj.weight"]
        mapped[dst + "attention.self.key.bias"] = esm[src + "self_attn.k_proj.bias"]
        mapped[dst + "attention.self.value.weight"] = esm[src + "self_attn.v_proj.weight"]
        mapped[dst + "attention.self.value.bias"] = esm[src + "self_attn.v_proj.bias"]
        mapped[dst + "attention.output.dense.weight"] = esm[src + "self_attn.out_proj.weight"]
        mapped[dst + "attention.output.dense.bias"] = esm[src + "self_attn.out_proj.bias"]
        mapped[dst + "attention.layer_norm.weight"] = esm[src + "self_attn_layer_norm.weight"]
        mapped[dst + "attention.layer_norm.bias"] = esm[src + "self_attn_layer_norm.bias"]
        mapped[dst + "intermediate.dense.weight"] = esm[src + "fc1.weight"]
        mapped[dst + "intermediate.dense.bias"] = esm[src + "fc1.bias"]
        mapped[dst + "output.dense.weight"] = esm[src + "fc2.weight"]
        mapped[dst + "output.dense.bias"] = esm[src + "fc2.bias"]
        mapped[dst + "layer_norm.weight"] = esm[src + "final_layer_norm.weight"]
        mapped[dst + "layer_norm.bias"] = esm[src + "final_layer_norm.bias"]
    return mapped


def official_utrlm_config() -> UtrLmConfig:
    """MultiMolecule config matching official pretrained architecture + alphabet."""
    return UtrLmConfig(
        vocab_size=OFFICIAL_VOCAB_SIZE,
        hidden_size=128,
        num_hidden_layers=6,
        num_attention_heads=16,
        intermediate_size=512,
        position_embedding_type="rotary",
        emb_layer_norm_before=False,
        token_dropout=False,
        pad_token_id=OFFICIAL_TOKENS["<pad>"],
        bos_token_id=OFFICIAL_TOKENS["<cls>"],
        eos_token_id=OFFICIAL_TOKENS["<eos>"],
        mask_token_id=OFFICIAL_TOKENS["<mask>"],
        unk_token_id=OFFICIAL_TOKENS["<unk>"],
    )


def load_pretrained_utrlm(pkl_path: str | None = None) -> UtrLmModel:
    """Load official pretrained UTR-LM backbone into a MultiMolecule shell."""
    path = os.path.abspath(pkl_path or default_pretrained_pkl())
    if path in _BACKBONE_CACHE:
        return _BACKBONE_CACHE[path]
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Official UTR-LM pretrained weights not found: {path}. "
            "See README (Pretrained UTR-LM weights)."
        )

    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict):
        raise TypeError(f"Expected state_dict dict in {path}, got {type(raw)}")

    mapped = remap_official_esm_to_utrlm(raw)
    model = UtrLmModel(official_utrlm_config())
    missing, unexpected = model.load_state_dict(mapped, strict=False)
    ignored = ("pooler.",)
    bad_missing = [k for k in missing if not k.startswith(ignored)]
    if bad_missing or unexpected:
        raise RuntimeError(
            f"Failed to load official UTR-LM weights from {path}. "
            f"missing={bad_missing}, unexpected={unexpected}"
        )
    _BACKBONE_CACHE[path] = model
    return model


def load_utrlm_for_pretraining(model_name: str) -> UtrLmForPreTraining:
    """Load frozen MultiMolecule ``UtrLmForPreTraining`` with HF weights."""
    if model_name in _PRETRAIN_CACHE:
        return _PRETRAIN_CACHE[model_name]

    config = UtrLmConfig.from_pretrained(model_name)
    model = UtrLmForPreTraining(config)
    weights_path = hf_hub_download(model_name, "model.safetensors")
    state_dict = _remap_hf_state_dict(load_file(weights_path))
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    ignored_prefixes = ("pooler.", "lm_head.decoder.")
    unexpected_missing = [
        key for key in missing
        if not key.startswith(ignored_prefixes)
    ]
    if unexpected_missing or unexpected:
        raise RuntimeError(
            f"Failed to load MultiMolecule UTR-LM weights from {model_name}. "
            f"missing={unexpected_missing}, unexpected={unexpected}"
        )
    if model.mfe_head is None:
        raise RuntimeError(f"{model_name} has no sequence head (mfe_head); cannot score.")
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    _PRETRAIN_CACHE[model_name] = model
    return model


class UTRLMRegressor(nn.Module):
    """Trainable UTR-LM sequence regressor (linear probe on official pretrained)."""

    pretrained_path: str | None = None
    default_pool: str = "mean"

    def __init__(
        self,
        device,
        seq_len: int,
        vocab_size: int,
        latent_dim: int = 128,
        pool: str | None = None,
        pretrained_path: str | None = None,
        freeze_backbone: bool = False,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.pool = pool or self.default_pool
        self.device = device
        self.loss_fn = nn.MSELoss()

        ckpt_path = pretrained_path or self.pretrained_path or default_pretrained_pkl()
        self.pretrained_path = ckpt_path
        base_model = load_pretrained_utrlm(ckpt_path)
        self.tokenizer = get_official_tokenizer()

        backbone = type(base_model)(base_model.config)
        backbone.load_state_dict(base_model.state_dict())
        self.base_model = backbone.to(device)
        if freeze_backbone:
            for param in self.base_model.parameters():
                param.requires_grad = False

        if latent_dim != int(base_model.config.hidden_size):
            raise ValueError(
                f"latent_dim={latent_dim} must match UTR-LM hidden_size="
                f"{base_model.config.hidden_size}"
            )
        self.MLP = nn.Linear(latent_dim, 1)

    def forward(self, x) -> torch.Tensor:
        return self.regress(self.encode(x))

    def regress(self, z) -> torch.Tensor:
        return torch.squeeze(self.MLP(z))

    def regression_loss(self, pred, target):
        if len(pred.shape) == 2:
            pred = pred.squeeze(-1)
        return self.loss_fn(pred, target)

    def encode(self, x) -> torch.Tensor:
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        output = self.base_model(**inputs)
        hidden_states = output["last_hidden_state"]

        if self.pool == "max":
            embeddings, _ = torch.max(hidden_states, dim=1)
        elif self.pool == "mean":
            embeddings = torch.mean(hidden_states, dim=1)
        elif self.pool == "cls":
            embeddings = hidden_states[:, 0, :]
        else:
            raise ValueError(f"Unsupported pool mode: {self.pool}")
        return embeddings

    def decode(self, z) -> torch.Tensor:
        batch_size = z.shape[0]
        return torch.randn((batch_size, self.seq_len, self.vocab_size), device=z.device)


class UTRLM(UTRLMRegressor):
    """Trainable oracle: official pretrained UTR-LM + mean-pool linear probe.

    Used only via ``train_oracle.py`` (per-dataset fine-tune -> ``model.pt``).
    Never initialized from UTRLM_TE / UTRLM_MRL hubs.
    """

    pretrained_path = None  # resolved to default_pretrained_pkl()
    default_pool = "mean"


class UTRLMFrozenHFOracle(nn.Module):
    """Frozen MultiMolecule TE/MRL oracle (zero-shot only; no train_oracle).

    Matches MultiMolecule's ``pipeline("mean-ribosome-load", ...)`` scoring path:
    CLS token -> sequence head (``mfe_head`` in the released safetensors).
    """

    pretrained_name: str = UTRLM_TE_PRETRAINED

    def __init__(
        self,
        device,
        seq_len: int,
        vocab_size: int,
        latent_dim: int = 128,
        pretrained_name: str | None = None,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.vocab_size = vocab_size
        self.device = device
        self.loss_fn = nn.MSELoss()

        ckpt_name = pretrained_name or self.pretrained_name
        self.pretrained_name = ckpt_name
        self.tokenizer = get_hf_tokenizer(ckpt_name)

        base = load_utrlm_for_pretraining(ckpt_name)
        # Fresh copy so device placement does not mutate the module-level cache.
        config = base.config
        model = UtrLmForPreTraining(config)
        model.load_state_dict(base.state_dict())
        model.eval()
        for param in model.parameters():
            param.requires_grad = False
        self.model = model.to(device)

    def _tokenize(self, x):
        inputs = self.tokenizer(x, return_tensors="pt", padding=True)
        for key, value in inputs.items():
            inputs[key] = value.to(self.device)
        return inputs

    def encode(self, x) -> torch.Tensor:
        """Return CLS embeddings [B, D] (for API / optional latent analysis)."""
        inputs = self._tokenize(x)
        with torch.no_grad():
            hidden = self.model.utrlm(**inputs, return_dict=True).last_hidden_state
        return hidden[:, 0, :]

    def score_sequences(self, x) -> torch.Tensor:
        """MultiMolecule sequence-level score [B]."""
        inputs = self._tokenize(x)
        with torch.no_grad():
            hidden = self.model.utrlm(**inputs, return_dict=True).last_hidden_state
            # ForPreTraining disables the pooler; CLS is the intended sequence vector.
            cls = hidden[:, 0, :]
            logits = self.model.mfe_head({"pooler_output": cls}).logits
        return torch.squeeze(logits, dim=-1)

    def regress(self, z) -> torch.Tensor:
        """If ``z`` is already scores [B] / [B,1], squeeze; else Linear is unused."""
        if z.dim() == 1 or (z.dim() == 2 and z.shape[-1] == 1):
            return torch.squeeze(z)
        # Fallback: score from CLS embedding via the frozen sequence head.
        with torch.no_grad():
            logits = self.model.mfe_head({"pooler_output": z}).logits
        return torch.squeeze(logits, dim=-1)

    def forward(self, x) -> torch.Tensor:
        return self.score_sequences(x)

    def regression_loss(self, pred, target):
        if len(pred.shape) == 2:
            pred = pred.squeeze(-1)
        return self.loss_fn(pred, target)

    def decode(self, z) -> torch.Tensor:
        batch_size = z.shape[0]
        return torch.randn((batch_size, self.seq_len, self.vocab_size), device=z.device)


class UTRLM_TE(UTRLMFrozenHFOracle):
    """Frozen MultiMolecule ``utrlm-te_el`` oracle (zebrafish / TE)."""

    pretrained_name = UTRLM_TE_PRETRAINED


class UTRLM_MRL(UTRLMFrozenHFOracle):
    """Frozen MultiMolecule ``utrlm-mrl`` oracle (ribosome / MRL)."""

    pretrained_name = UTRLM_MRL_PRETRAINED


if __name__ == "__main__":
    text = ["UAGCUUAUCAGACUGAUGUUG", "UAGCUUAUCAGACUGAUGUUU"]
    model = UTRLM(device="cpu", seq_len=50, vocab_size=7, latent_dim=128)
    z = model.encode(text)
    y = model.regress(z)
    print("UTRLM", z.shape, y.shape, y.tolist(), model.pretrained_path)
    for cls in (UTRLM_TE, UTRLM_MRL):
        frozen = cls(device="cpu", seq_len=50, vocab_size=7)
        out = frozen(text)
        print(cls.__name__, out.shape, out.tolist(), frozen.pretrained_name)
