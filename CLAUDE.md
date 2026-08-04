# RNAGenScape — agent / coding rules

## No backward-compat shims (HARD RULE)

This is a **fresh** codebase. Experiments are tentative until the stack is solid.

- NEVER add backward-compatible aliases (e.g. `_float_tag = float_tag`).
- NEVER keep legacy path formats, legacy checkpoint key loaders, or dual APIs "so old runs still work."
- Prefer deleting / renaming and retraining over compatibility layers.
- If something breaks old dirs or checkpoints, that is fine — retrain.

## argparse: one line per argument (HARD RULE)

**Every** `p.add_argument(...)` / `parser.add_argument(...)` call MUST be a **single physical line**.

- NEVER split flags, `type=`, `default=`, `help=`, `choices=`, etc. across multiple lines.
- Keep `help=` short enough to fit; do not wrap the call.

```python
# GOOD
p.add_argument("--latent_dim", type=int, default=128, help="OAE latent dim; path tag d{latent}_recon{w}.")

# BAD — never do this
p.add_argument(
    "--latent_dim",
    type=int,
    default=128,
    help="...",
)
```

This applies to all Python in this repo (`src/`, tests, scripts).

## Comments: ASCII only (HARD RULE)

Do **not** use non-ASCII symbols in comments (or docstrings used as comments).

- NEVER use special Unicode glyphs that are awkward to type (e.g. arrows like `\u2192`, fancy dashes, bullets, Greek letters as symbols, checkmarks, emoji).
- Prefer plain ASCII: `->`, `<-`, `--`, `*`, `x`, `z`, `L'`, etc.
- Math / LaTeX-style formulas in comments are fine when written in ASCII, e.g. `z in R^D`, `` `L' = downsample(L)` ``, `MSE + recon_w * CE`.
- Code identifiers and string literals are unrelated; this rule is about human-written comment text.

```python
# GOOD
# Encode: [B, L, V] -> [B, C, L'] -> GAP -> [B, D]

# BAD — never do this
# Encode: [B, L, V] → [B, C, L′] → GAP → [B, D]
```
