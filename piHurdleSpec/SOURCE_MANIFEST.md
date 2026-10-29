# Source provenance

This folder contains release code and two pretrained model checkpoints assembled
from the existing workspace. The original source directories were not modified.

| Release files | Original workspace location | Release changes |
| --- | --- | --- |
| `nvidia/{config,dataset,model,losses,metrics,train,evaluate,smoke_test}.py` | `Top-down/NovaPredict/NovaPredict_CNN_BiLSTM_MultiCharge_Hurdle_MVN_TwoStage/` | Portable data defaults, Stage-1 evaluation phase and CUDA-build check |
| `nvidia/{decision_model,train_decoder}.py` | `Top-down/NovaPredict/NovaPredict_CNN_BiLSTM_MultiCharge_Hurdle_MVN_DirectDecisionDecoder_PiMeanCov/` | Portable data and Stage-1 checkpoint defaults; CUDA-build check |
| `amd/` model and trainer files | Same two source directories as NVIDIA | ROCm-build check, FP32 default, native LSTM backend, first-batch encoder gradient checks |
| `nvidia/tests/`, `amd/tests/` | Tests in the two source directories | Copied without changes |
| `data/{build_multi_charge_hdf5,validate_multi_charge_hdf5}.py` | `data/TD-repo/HDF/mutil_charge/` | Copied without changes |
| `*/predict.py`, `*/run.sh` | New release entry points | Stream Stage-2 predictions and run both phases |
| `*/deterministic_pretrain/{train,model,dataset}.py`, `*/pretrain_cnn_bilstm.py`, `*/pretrain_cnn_bilstm.sh` | `Top-down/NovaPredict/NovaPredict_CNN_BiLSTM/` | Portable old-data training entry points; AMD uses native LSTM and checks encoder gradients |
| `data/logmvn_support/{compat,metrics,dataset,model,losses,train}.py`, `*/pretrain_logmvn.py`, `*/run_logmvn.sh` | `Top-down/NovaPredict/NovaPredict_CNN_BiLSTM_LogMVN/` and required functions in `NovaPredict_Human_Tissue/` | Consolidated shared helpers, portable launchers, AMD native LSTM and gradient checks |
| `data/build_logmvn_hdf5.py` | Target logic from `NovaPredict_Human_Tissue/TD_repo_train/tdrepo_to_novapred_hdf5.py` and `NovaPredict_TDRepo_metadata_ablation/build_enriched_tdrepo_hdf5.py` | Streams input in order; retains original strongest-charge B/Y target and per-spectrum max normalization |
| `weights/cnn_bilstm_best.pt` | `NovaPredict_CNN_BiLSTM/log/best.pt` | Model tensors unchanged; optimizer and machine-specific config removed |
| `weights/logmvn_best.pt` | `NovaPredict_CNN_BiLSTM_LogMVN/log/best.pt` | Model tensors unchanged; optimizer and machine-specific config removed |

The Stage-1 model and Stage-2 direct decision decoder are the variants used in
the cross-model comparison (`hurdle_onestage` and `hurdle_direct_decoder`). The
older `TwoStage/train.py` also supports an expected-intensity Stage-2 option,
but this release's `run.sh` selects Stage 1 from that trainer and uses
`train_decoder.py` for the final Stage 2.

Data, new training outputs, logs, figures, local environments, comparison-model
code, and unpublished experiment outputs remain outside this release. The
original TopRepo validation HDF5 is named `test_multi_charge.hdf5` in the
workspace; the release uses `val_multi_charge.hdf5` to describe its role. The
pretraining HDF5 uses the older `[N,199,2]` target, whereas π-HurdleSpec uses
the later `[N,199,2,30]` target.
