# Pretrained weights

Both files contain the original model tensors without modification. The release
copies omit optimizer state and machine-specific training paths; they keep the
model state dictionary, best epoch, validation loss, and parameter count. They
are ordinary PyTorch checkpoints and load on either CUDA or ROCm.

| Release file | Role | Best epoch | Original checkpoint SHA256 | Release SHA256 |
| --- | --- | ---: | --- | --- |
| `cnn_bilstm_best.pt` | Deterministic CNN–BiLSTM initialization for LogMVN | 24 | `a25a9f510232333c1fcb0356b608ab15a240471b064ca89aef1d2b42ed86c260` | `60f605c527560ed0dca07858451a9337a8b12df6502eb26702043e801dbca1ab` |
| `logmvn_best.pt` | LogMVN initialization used by π-HurdleSpec Stage 1 | 7 | `67a6042690d49cd539874047f76dd46050299ecc212da38c2a0d3c1cd1397638` | `affffa8a0c57ad81080f86a7f082f00d79e9ccf5e519a2d2a943f6cf8c78da92` |

Original checkpoint locations in the research workspace:

- `NovaPredict_CNN_BiLSTM/log/best.pt`
- `NovaPredict_CNN_BiLSTM_LogMVN/log/best.pt`

The LogMVN checkpoint was trained on the older, charge-collapsed B/Y target
`[N, 199, 2]`: each ion type and cleavage retains the strongest observed
charge-state peak. π-HurdleSpec Stage 1 transfers compatible encoder weights to a
charge-resolved model; its new output heads are initialized during Stage 1.

Verify the release copies from the release root:

```bash
sha256sum -c weights/SHA256SUMS
```
