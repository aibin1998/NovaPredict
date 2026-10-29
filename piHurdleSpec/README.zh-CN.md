# π-HurdleSpec 代码与运行说明

本目录整理了 π-HurdleSpec 的完整模型链：确定性 CNN–BiLSTM 预训练、LogMVN 预训练、**Stage 1 hurdle-MVN 分布模型**与**Stage 2 直接决策解码器**。随包提供两份预训练权重；NVIDIA/CUDA 和 AMD/ROCm 的训练与预测入口分别放在独立目录。

### 1. 目录结构

```text
piHurdleSpec_release/
├── README.md                  # 使用说明
├── SOURCE_MANIFEST.md         # 原始代码位置及发布版改动
├── requirements-common.txt    # 两个平台共同的 Python 依赖
├── weights/
│   ├── cnn_bilstm_best.pt      # LogMVN 的确定性预训练初始化
│   ├── logmvn_best.pt          # Stage 1 的 LogMVN 预训练初始化
│   ├── README.md, SHA256SUMS   # 权重来源与校验和
├── data/
│   ├── build_logmvn_hdf5.py          # annotated msalign → 旧版二维 B/Y HDF5
│   ├── logmvn_support/               # 两层预训练共用的数据与损失逻辑
│   ├── build_multi_charge_hdf5.py     # 已注释 msalign → 电荷分辨 HDF5
│   └── validate_multi_charge_hdf5.py  # 校验 HDF5 结构与强度归一化
├── nvidia/
│   ├── pretrain_cnn_bilstm.sh, run_logmvn.sh  # 两层预训练入口
│   ├── pretrain_cnn_bilstm.py, pretrain_logmvn.py
│   ├── deterministic_pretrain/       # 确定性模型训练实现
│   ├── run.sh                 # CUDA：依次训练 Stage 1 和 Stage 2
│   ├── train.py               # Stage 1 训练入口（指定 --stage2-epochs 0）
│   ├── train_decoder.py       # Stage 2 直接决策解码器训练
│   ├── predict.py             # Stage 2 批量预测与可选评估
│   ├── evaluate.py            # Stage 1 检查点评估
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

数据文件放在 `data/`，新训练检查点与预测结果由脚本写到 `outputs/nvidia/` 或 `outputs/amd/`。`.gitignore` 已排除 HDF5、msalign、新生成的权重、日志和缓存，但明确保留 `weights/` 中随包发布的两份检查点。原始谱图数据需另行提供。

主要文件说明：`model.py` 定义 CNN–BiLSTM 与 hurdle-MVN 的发生概率、条件均值和协方差；`decision_model.py` 由这些分布参数直接产生 Stage 2 强度谱；`dataset.py` 读取多电荷 HDF5；`predict.py` 将预测按输入顺序流式保存为 HDF5。两个硬件目录各自包含训练源码，不从另一硬件目录导入模块；旧二维 B/Y 数据处理逻辑由 `data/logmvn_support/` 共享。

### 2. 整体流程

1. 使用**已完成 proteoform 级训练/验证划分**的 annotated msalign 构建两种 HDF5：旧版 `[N,199,2]` B/Y 标签供两层预训练使用，新版 `[N,199,2,30]` 电荷分辨标签供 π-HurdleSpec 使用。构建脚本不负责数据划分。
2. 确定性 CNN–BiLSTM 在旧版 HDF5 上训练，最佳检查点作为 LogMVN 初始化；LogMVN 也在旧版 HDF5 上训练，最佳检查点作为 Stage 1 初始化。随包提供这两个历史最佳检查点，可直接跳到下一步。
3. Stage 1 学习碎片是否出现的概率，以及非零强度的低秩多元正态分布。按验证集 distribution loss 选择 `stage1_best.pt`。
4. Stage 2 从 Stage 1 检查点初始化，联合微调分布模型和直接决策解码器；按验证集 spectral-angle loss 选择 `best.pt`。
5. 加载 Stage 2 检查点，对同一 HDF5 格式的数据预测，并可选输出验证指标。

### 3. Quick Start

以下命令在本目录执行。推荐 Python 3.10；本地核对的依赖是 PyTorch 2.10.0、NumPy 1.26.4、h5py 3.16.0、pandas 2.3.3。请根据显卡和驱动选择**其中一套**环境。PyTorch 安装命令对应其[官方历史版本说明](https://pytorch.org/get-started/previous-versions/)。

**NVIDIA/CUDA 12.8：**

```bash
conda create -n pihurdlespec-cu python=3.10 -y
conda activate pihurdlespec-cu
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements-common.txt
python -c 'import torch; assert torch.version.cuda is not None'
```

**AMD/ROCm 7.1（Linux）：**

```bash
conda create -n pihurdlespec-rocm python=3.10 -y
conda activate pihurdlespec-rocm
python -m pip install torch==2.10.0 --index-url https://download.pytorch.org/whl/rocm7.1
python -m pip install -r requirements-common.txt
python -c 'import torch; assert torch.version.hip is not None'
```

准备数据；第二个文件是**验证集**，不是独立测试集：

```bash
python data/build_multi_charge_hdf5.py --input /path/to/annotated_train.msalign --output data/train_multi_charge.hdf5
python data/build_multi_charge_hdf5.py --input /path/to/annotated_val.msalign --output data/val_multi_charge.hdf5
python data/validate_multi_charge_hdf5.py data/train_multi_charge.hdf5 data/val_multi_charge.hdf5
```

`nvidia/run.sh` 与 `amd/run.sh` 默认读取随包提供的 `weights/logmvn_best.pt` 初始化 Stage 1。因此，只想使用原始 LogMVN 权重继续训练 π-HurdleSpec 时，无需先重训预训练模型。可用 `INIT_CHECKPOINT=/path/to/another.pt` 替换权重，或用 `USE_LOGMVN_PRETRAIN=0` 从随机初始化开始 Stage 1。

选择 NVIDIA 或 AMD 启动训练：

```bash
GPU=0 ./nvidia/run.sh
# 或在 ROCm 环境中：
GPU=0 ./amd/run.sh
```

Stage 2 推理及可选的验证指标：

```bash
CUDA_VISIBLE_DEVICES=0 python nvidia/predict.py \
  --checkpoint outputs/nvidia/stage2/best.pt \
  --hdf5 data/val_multi_charge.hdf5 \
  --output outputs/nvidia/val_predictions.hdf5 \
  --metrics-json outputs/nvidia/val_metrics.json

# AMD 环境使用：
ROCR_VISIBLE_DEVICES=0 python amd/predict.py \
  --checkpoint outputs/amd/stage2/best.pt \
  --hdf5 data/val_multi_charge.hdf5 \
  --output outputs/amd/val_predictions.hdf5 \
  --metrics-json outputs/amd/val_metrics.json
```

输出 HDF5 含 `prsm_id` 和 `[N,199,2,30]` 的 `predicted_intensity`；输入仍须符合本目录构建的 HDF5 schema。`--metrics-json` 使用输入中的实验标签评估；不指定该选项时只保存预测。

### 4. 策略

- **数据表示：** 每个谱图最多 200 个氨基酸，目标为 `[199 cleavages, 2 ion types, 30 fragment charges]`。N 端 b/c 与 C 端 y/z 共用断裂位置；实验强度以**每张谱图最大的电荷分辨离子强度**归一化。默认过滤条件为 E-value ≤ `1e-4`、碎片覆盖率 ≥ `0.2`、前体电荷 ≤ 30。
- **模型：** Stage 1 预测出现概率 `π`、条件均值 `μ` 和低秩协方差。Stage 2 的直接解码器接收这些分布参数，输出经非负变换、理论离子掩码和逐谱最大值归一化的强度；其目标为 spectral-angle loss 加 `0.01 ×` Stage 1 distribution loss。它与旧版直接使用 `π × μ` 的 Stage 2 实现不同。
- **数据隔离：** 训练与验证必须在 proteoform 层面先划分；本仓库的 HDF5 构建脚本只处理传入的已划分文件。原 TopRepo 项目中名为 `test_multi_charge.hdf5` 的文件实际对应 validation split。
- **硬件：** NVIDIA 版本使用 CUDA 构建的 PyTorch 和 `CUDA_VISIBLE_DEVICES`。AMD 版本使用 ROCm 构建的 PyTorch、`ROCR_VISIBLE_DEVICES`、FP32 与原生 PyTorch LSTM 后端；在 PyTorch 中 ROCm 显卡仍以 `cuda:0` 表示。本机 ROCm 7.1 / MIOpen 3.5.1 曾出现双层 LSTM dropout 反向梯度归零，因此 AMD 训练入口禁用对应加速后端，并在编码器训练开始后的前两个 batch 检查梯度。
- **预训练数据：** 确定性 CNN–BiLSTM 与 LogMVN 使用旧版 `[N,199,2]` B/Y 标签；同一离子类型和断裂位置只保留强度最大的电荷峰及对应电荷，再按逐谱最大强度线性归一化。它与 Stage 1/2 的 `[N,199,2,30]` 多电荷标签构建口径不同，不能直接复用多电荷 HDF5。
- **权重来源：** 原始 Stage 1 使用 LogMVN `best.pt` 进行 backbone warm-start；原 LogMVN 又从确定性 CNN–BiLSTM `best.pt` 初始化。本目录 `weights/` 中的两个检查点保留了原始模型张量，去除了优化器状态和本机绝对路径。Stage 1 会迁移兼容的编码器参数，新输出头在 Stage 1 中初始化。来源、epoch 与 SHA256 见 [weights/README.md](weights/README.md)。

### 5. Pipeline

常用可调参数由环境变量传递给两个启动脚本：`GPU`、`PYTHON`、`TRAIN_HDF5`、`VAL_HDF5`、`OUTPUT_DIR`、`BATCH_SIZE`、`NUM_WORKERS`、`STAGE1_EPOCHS`、`STAGE2_EPOCHS`、`INIT_CHECKPOINT` 和 `USE_LOGMVN_PRETRAIN`。例如：

```bash
BATCH_SIZE=1024 NUM_WORKERS=2 STAGE1_EPOCHS=20 STAGE2_EPOCHS=30 GPU=0 ./nvidia/run.sh
# ROCm：
BATCH_SIZE=1024 NUM_WORKERS=2 STAGE1_EPOCHS=20 STAGE2_EPOCHS=30 GPU=0 ./amd/run.sh
```

也可以分别运行两个阶段。下面的 NVIDIA 命令展示检查点传递方式；AMD 对应入口在 `amd/`，且需加入 `--no-amp`、使用 `ROCR_VISIBLE_DEVICES`：

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

快速检查核心损失与解码器：

```bash
python -m unittest discover -s nvidia/tests -v
python -m unittest discover -s amd/tests -v
```

训练产物：`outputs/<hardware>/stage1/stage1_best.pt`、`outputs/<hardware>/stage2/best.pt`。预测产物：指定的 HDF5 与可选 JSON；这些文件会被 `.gitignore` 排除。

### 6. 从数据重训两层预训练模型

对已在 proteoform 层面划分的训练、验证 annotated msalign 另建旧版二维 HDF5：

```bash
python data/build_logmvn_hdf5.py --input /path/to/annotated_train.msalign --output data/logmvn/train.hdf5
python data/build_logmvn_hdf5.py --input /path/to/annotated_val.msalign --output data/logmvn/val.hdf5
```

先运行确定性 CNN–BiLSTM，再用其输出初始化 LogMVN。以下是 NVIDIA 命令；AMD 将目录名换成 `amd/` 即可：

```bash
GPU=0 bash nvidia/pretrain_cnn_bilstm.sh
GPU=0 DETERMINISTIC_CHECKPOINT=outputs/nvidia/pretrain_cnn_bilstm/best.pt bash nvidia/run_logmvn.sh
GPU=0 INIT_CHECKPOINT=outputs/nvidia/logmvn/best.pt bash nvidia/run.sh
```

也可只重训 LogMVN：`run_logmvn.sh` 默认使用随包的 `weights/cnn_bilstm_best.pt`。原始 LogMVN 配置是 50 epoch、batch size 2048、rank 8、前 2 epoch 冻结 backbone、backbone/head 学习率分别为 `1e-4`/`5e-4`；最佳检查点为 epoch 7。预训练脚本根据验证损失选择最佳检查点。旧数据构建器按输入顺序写入，而历史构建器曾并行乱序写入；标签转换已对真实谱图逐字段核对，重新训练不保证得到与历史权重逐位相同的参数。直接使用 `weights/logmvn_best.pt` 可复用历史 Stage 1 的初始化。
