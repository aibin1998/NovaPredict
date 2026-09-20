# Full-data result

The selected direct distribution-to-spectrum decoder was trained on the TDRepo multi-charge HDF5 split.

| Metric | Value |
|---|---:|
| Best epoch | 28 |
| Validation SA loss | 0.27640 |
| Charge-resolved cosine | 0.880199 |
| Charge-summed cosine | 0.910254 |
| Stage-2 distribution weight | 0.01 |
| Batch size | 2048 |
| Random seed | 1337 |

The decoder receives only `pi`, `mu`, `s`, and rank-4 `U`; it does not receive sequence features or Stage-1 hidden states. Stage 1 and the decoder are jointly fine-tuned in Stage 2.
