# π-HurdleSpec

[中文说明](README.zh-CN.md)

This directory contains the π-HurdleSpec training and inference code, the deterministic CNN–BiLSTM and LogMVN pretraining pipeline, and the two pretrained weights used to initialize that pipeline. The NVIDIA/CUDA and AMD/ROCm entry points are kept in separate directories.

All commands below run from this directory. Annotated spectra are **not** included.

### 1. Repository layout

```text
piHurdleSpec/
├── README.md                         # English instructions
├── README.zh-CN.md                   # Chinese instructions
├── SOURCE_MANIFEST.md                # Source provenance and release changes
├── requirements-common.txt          # Shared Python dependencies
├── weights/
│   ├── cnn_bilstm_best.pt            # Deterministic initialization for LogMVN
│   ├── logmvn_best.pt                # LogMVN initialization for Stage 1
│   ├── README.md                     # Checkpoint provenance
│   └── SHA256SUMS
├── data/
│   ├── build_logmvn_hdf5.py          # Annotated msalign → legacy [N, 199, 2] B/Y HDF5
│   ├── logmvn_support/               # Shared, hardware-independent pretraining code
│   ├── build_multi_charge_hdf5.py    # Annotated msalign → [N, 199, 2, 30] HDF5
│   └── validate_multi_charge_hdf5.py
├── nvidia/
│   ├── pretrain_cnn_bilstm.sh, run_logmvn.sh
│   ├── pretrain_cnn_bilstm.py, pretrain_logmvn.py
│   ├── deterministic_pretrain/
│   ├── run.sh                        # Stage 1 followed by Stage 2
│   ├── train.py, train_decoder.py     # Stage 1 and direct-decoder Stage 2
│   ├── predict.py, evaluate.py
│   ├── config.py, dataset.py, model.py, losses.py, metrics.py
│   ├── decision_model.py, smoke_test.py
│   └── tests/
└── amd/
    ├── pretrain_cnn_bilstm.sh, run_logmvn.sh
    ├── pretrain_cnn_bilstm.py, pretrain_logmvn.py
    ├── deterministic_pretrain/
    ├── run.sh, train.py, train_decoder.py, predict.py, evaluate.py
    ├── config.py, dataset.py, model.py, losses.py, metrics.py
    ├── decision_model.py, smoke_test.py
    └── tests/
```

The two hardware directories each contain their own model training and prediction entry points. They share only the hardware-independent legacy B/Y pretraining helpers in `data/logmvn_support/`. The `outputs/` directory in the examples is created by the scripts and excluded from Git. HDF5 data, annotated msalign files, and newly generated checkpoints are also excluded; the two distributed checkpoints in `weights/` are explicitly retained.

### 2. Workflow

1. Split annotated spectra by **proteoform**, then supply the already separated training and validation msalign files. The HDF5 builders do not create a split.
2. Build two target formats from those files. Deterministic CNN–BiLSTM and LogMVN pretraining use charge-collapsed B/Y targets of shape `[N, 199, 2]`. π-HurdleSpec Stage 1 and Stage 2 use charge-resolved targets of shape `[N, 199, 2, 30]`.
3. Train the deterministic CNN–BiLSTM and initialize LogMVN from its best checkpoint. Train LogMVN and use its best checkpoint to initialize the compatible Stage 1 encoder. Both historical best checkpoints are provided, so these two pretraining steps can be skipped.
4. Train Stage 1 to estimate ion occurrence probabilities and the conditional intensity distribution. Select `stage1_best.pt` using validation distribution loss.
5. Initialize the direct-decision Stage 2 decoder from Stage 1 and jointly fine-tune it with the distribution model. Select `best.pt` using validation spectral-angle loss.
6. Run Stage 2 inference on an HDF5 file with the charge-resolved input schema; optionally calculate metrics against its experimental labels.

### 3. Quick start

Use Python 3.10 and a PyTorch build compatible with your GPU and driver. The dependency versions used to check this release were PyTorch 2.10.0, NumPy 1.26.4, h5py 3.16.0, and pandas 2.3.3. The examples below use CUDA 12.8 or ROCm 7.1 wheels; see the [PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/) for other supported combinations. Install **one** of the following environments.

**NVIDIA / CUDA 12.8**

```bash
conda create -n pihurdlespec-cu python=3.10 -y
conda activate pihurdlespec-cu
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-common.txt
python -c 'import torch; assert torch.version.cuda is not None'
```

**AMD / ROCm 7.1 on Linux**

```bash
conda create -n pihurdlespec-rocm python=3.10 -y
conda activate pihurdlespec-rocm
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/rocm7.1
python -m pip install -r requirements-common.txt
python -c 'import torch; assert torch.version.hip is not None'
```

Build and validate the charge-resolved HDF5 files from **pre-split** annotated msalign inputs. The second file is a validation set, not an independent test set.

```bash
python data/build_multi_charge_hdf5.py --input /path/to/annotated_train.msalign --output data/train_multi_charge.hdf5
python data/build_multi_charge_hdf5.py --input /path/to/annotated_val.msalign --output data/val_multi_charge.hdf5
python data/validate_multi_charge_hdf5.py data/train_multi_charge.hdf5 data/val_multi_charge.hdf5
```

The launchers load `weights/logmvn_best.pt` by default to initialize Stage 1. To use another LogMVN checkpoint, set `INIT_CHECKPOINT=/path/to/checkpoint.pt`. To start Stage 1 without pretrained weights, set `USE_LOGMVN_PRETRAIN=0`.

Run the launcher for your hardware:

```bash
GPU=0 ./nvidia/run.sh
# On a ROCm installation:
GPU=0 ./amd/run.sh
```

Run Stage 2 inference. Add `--metrics-json` to evaluate against the experimental intensities in the input HDF5; omit it to save predictions only.

```bash
CUDA_VISIBLE_DEVICES=0 python nvidia/predict.py \
  --checkpoint outputs/nvidia/stage2/best.pt \
  --hdf5 data/val_multi_charge.hdf5 \
  --output outputs/nvidia/val_predictions.hdf5 \
  --metrics-json outputs/nvidia/val_metrics.json

# On a ROCm installation:
ROCR_VISIBLE_DEVICES=0 python amd/predict.py \
  --checkpoint outputs/amd/stage2/best.pt \
  --hdf5 data/val_multi_charge.hdf5 \
  --output outputs/amd/val_predictions.hdf5 \
  --metrics-json outputs/amd/val_metrics.json
```

The prediction HDF5 contains `prsm_id` and `predicted_intensity` with shape `[N, 199, 2, 30]`, in input order. The prediction command requires the charge-resolved input schema produced by this release's builder. It will not overwrite an existing output unless `--overwrite` is supplied.

### 4. Data and model details

- **Charge-resolved targets:** Each proteoform has at most 200 amino acids. The target dimensions are 199 cleavage positions, two ion-type slots, and 30 fragment-charge slots. N-terminal b/c ions and C-terminal y/z ions share the cleavage-position indexing. The builder divides each spectrum's intensities by the maximum charge-resolved ion intensity in that spectrum. Its default acceptance thresholds are E-value ≤ `1e-4`, annotated-ion coverage ≥ `0.2`, and precursor charge ≤ 30.
- **Legacy pretraining targets:** The deterministic CNN–BiLSTM and LogMVN use `[N, 199, 2]` B/Y targets. For each ion type and cleavage position, the converter keeps the highest-intensity charge-state peak and its charge label, then divides all intensities by the per-spectrum maximum. These labels cannot be substituted for the charge-resolved Stage 1/2 labels.
- **Stage 1:** The hurdle-MVN model predicts ion occurrence probability `π`, conditional mean `μ`, and a low-rank covariance representation. Its output heads are initialized for the charge-resolved task; only compatible encoder parameters are transferred from LogMVN.
- **Stage 2:** The direct-decision decoder consumes the Stage 1 distribution parameters and predicts a nonnegative spectrum under the theoretical ion mask, normalized to a per-spectrum maximum of one. Training uses spectral-angle loss plus `0.01 ×` the Stage 1 distribution loss. The launchers use this direct decoder, rather than the older expected-intensity Stage 2 in `train.py`.
- **Data isolation:** Make the training/validation split at the proteoform level before conversion. The original TopRepo file named `test_multi_charge.hdf5` corresponds to the validation split in the research workspace; this release calls the analogous file `val_multi_charge.hdf5`.
- **GPU backends:** NVIDIA code expects a CUDA PyTorch build and uses `CUDA_VISIBLE_DEVICES`. AMD code expects a ROCm PyTorch build and uses `ROCR_VISIBLE_DEVICES`. PyTorch represents a ROCm GPU as `cuda:0`. AMD training uses FP32 and the native PyTorch LSTM backend; its first training batches check the encoder gradient because the tested ROCm/MIOpen stack produced zero gradients with the accelerated two-layer LSTM.
- **Checkpoint provenance:** The distributed checkpoints preserve the source model tensors while omitting optimizer state and machine-specific paths. Their original locations, best epochs, and SHA256 hashes are documented in [weights/README.md](weights/README.md). Verify the copies with `sha256sum -c weights/SHA256SUMS`.

### 5. Training options

Both `nvidia/run.sh` and `amd/run.sh` accept these environment variables: `GPU`, `PYTHON`, `TRAIN_HDF5`, `VAL_HDF5`, `OUTPUT_DIR`, `BATCH_SIZE`, `NUM_WORKERS`, `STAGE1_EPOCHS`, `STAGE2_EPOCHS`, `INIT_CHECKPOINT`, and `USE_LOGMVN_PRETRAIN`.

```bash
BATCH_SIZE=1024 NUM_WORKERS=2 STAGE1_EPOCHS=20 STAGE2_EPOCHS=30 GPU=0 ./nvidia/run.sh
# On a ROCm installation:
BATCH_SIZE=1024 NUM_WORKERS=2 STAGE1_EPOCHS=20 STAGE2_EPOCHS=30 GPU=0 ./amd/run.sh
```

The two phases can also be run separately. This NVIDIA example shows how the Stage 1 checkpoint is passed to Stage 2:

```bash
CUDA_VISIBLE_DEVICES=0 python nvidia/train.py \
  --train-hdf5 data/train_multi_charge.hdf5 --val-hdf5 data/val_multi_charge.hdf5 \
  --out-dir outputs/nvidia/stage1 --device cuda:0 --stage1-epochs 20 --stage2-epochs 0 \
  --init-checkpoint weights/logmvn_best.pt

CUDA_VISIBLE_DEVICES=0 python nvidia/train_decoder.py \
  --train-hdf5 data/train_multi_charge.hdf5 --val-hdf5 data/val_multi_charge.hdf5 \
  --stage1-checkpoint outputs/nvidia/stage1/stage1_best.pt \
  --out-dir outputs/nvidia/stage2 --device cuda:0 --epochs 30 \
  --stage2-distribution-weight 0.01
```

For AMD, use the matching `amd/` entry points, `ROCR_VISIBLE_DEVICES`, and `--no-amp` with both training commands. The launchers set these options automatically.

Run the small loss and decoder unit tests with:

```bash
python -m unittest discover -s nvidia/tests -v
python -m unittest discover -s amd/tests -v
```

The launchers write `outputs/<hardware>/stage1/stage1_best.pt` and `outputs/<hardware>/stage2/best.pt`. Predictions are written to the path given by `--output`. These newly generated files are excluded from Git.

### 6. Reproducing both pretraining stages

Create separate legacy B/Y HDF5 files from the same **pre-split** annotated training and validation msalign files:

```bash
python data/build_logmvn_hdf5.py --input /path/to/annotated_train.msalign --output data/logmvn/train.hdf5
python data/build_logmvn_hdf5.py --input /path/to/annotated_val.msalign --output data/logmvn/val.hdf5
```

Train the deterministic CNN–BiLSTM, initialize LogMVN from its new best checkpoint, and then initialize Stage 1 from the newly trained LogMVN checkpoint:

```bash
GPU=0 bash nvidia/pretrain_cnn_bilstm.sh
GPU=0 DETERMINISTIC_CHECKPOINT=outputs/nvidia/pretrain_cnn_bilstm/best.pt bash nvidia/run_logmvn.sh
GPU=0 INIT_CHECKPOINT=outputs/nvidia/logmvn/best.pt bash nvidia/run.sh
```

On AMD, replace `nvidia/` with `amd/` throughout these three commands after activating the ROCm environment. If you only need to retrain LogMVN, `run_logmvn.sh` defaults to the distributed `weights/cnn_bilstm_best.pt`.

The historical LogMVN run used 50 epochs, batch size 2048, covariance rank 8, a two-epoch backbone freeze, and backbone/head learning rates of `1e-4` and `5e-4`. The best historical LogMVN checkpoint was selected at epoch 7 using validation loss. The distributed legacy data builder writes spectra in input order; the original research builder used unordered parallel workers. Target values were checked against real annotated spectra, but a new training run is not expected to reproduce the historical weights bit for bit. Use `weights/logmvn_best.pt` to reuse the historical Stage 1 initialization.
