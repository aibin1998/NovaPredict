## NovaPredict

A Deep Learning Model for Predicting Top-Down MS/MS Spectra of Proteoforms.

### 当前已实现功能

#### 1. 训练任务

- 基于 proteoform 序列的碎裂位点级别建模，不再按氨基酸单点建模。
- 每个时间步对应一个 cleavage site，标签按 b/y 两类离子组织。
- 支持使用实验条件作为输入特征：
	- Fragmentation
	- Instrument
	- Energy bucket
	- Precursor charge
- 主模型采用 BiLSTM encoder + 多任务预测头。

#### 2. 预测头

当前模型包含 4 个输出头：

- intensity head
	- 预测 observed fragment 的相对强度
- dmass head
	- 预测 fragment 的质量偏差
- charge head
	- 预测 fragment charge 类别
- break probability head
	- 预测某个 cleavage site 的 b/y 离子是否可能被观测到

其中 break head 使用独立的 BCE loss 训练，用来补足 mask 训练下“未监督位置不能直接表示是否断裂”的问题。

#### 3. 标签与监督方式

- 标签来自 core tsv 和对应的 prsm CSV。
- 监督以 cleavage site 为单位组织，shape 为 `(max_len - 1, 2)`。
- `ion_mask` 表示某个 cleavage site 的某个离子类型是否有监督。
- `cleavage_mask` 表示该位置是否属于真实序列范围。
- intensity / dmass / charge 只在 `cleavage_mask * ion_mask` 上计算 loss。
- break head 在所有真实 cleavage 上训练，用于学习“是否可能断裂/是否可能有峰”。

#### 4. 同一离子分组的标签构造策略

`Train.py` 中支持通过 `group_label_strategy` 控制同一 `(ion_type, ion_position)` 的多条 PRSM 记录如何合并：

- `min_error_row`
	- 选择 `|mass_error|` 最小的记录
	- 若没有 `mass_error`，回退到 `|ppm|` 最小
- `aggregate`
	- intensity 取最大值
	- dmass 取平均值
	- charge 取众数

#### 5. 评估指标

`Evaluate.py` 当前支持以下评估：

- intensity PCC
	- `all_valid`
		- 在全部真实 cleavage 范围内计算 PCC
	- `observed_only`
		- 只在 `ion_mask=1` 的位置计算 PCC
- charge PCC
	- `charge_observed_only`
		- 在 observed-only 位置上，对预测 charge 与真实 charge 计算 PCC
- break metrics
	- precision
	- recall
	- f1
	- threshold 由 `break_prob_threshold` 控制

评估结果除了 summary 外，还会保留逐 PRSM 的 observed-only PCC，用于后续导出分布 CSV。

#### 6. 训练日志输出

`Train.py` 当前每个 epoch 会输出：

- `train_loss`
- `val_loss`
- `pcc_obs_med`
- `pcc_obs_mean`
- `pcc_obs_max`
- `pcc_obs_max_prsm`
- `charge_pcc_obs_med`
- `charge_pcc_obs_mean`
- `charge_pcc_obs_max`
- `charge_pcc_obs_max_prsm`
- `break_prec`
- `break_rec`
- `break_f1`

#### 7. 最优 PCC epoch 记录与 CSV 导出

训练过程中会跟踪 `pcc_obs_mean` 最高的 epoch。

当 `save_best_pcc_obs_csv=True` 时，训练结束后会导出该最佳 epoch 下验证集所有 PRSM 的 observed-only PCC 分布 CSV。

默认输出路径：

- `./outputs/best_pcc_obs_distribution.csv`

CSV 当前包含列：

- `epoch`
- `prsm_id`
- `pcc_obs`

这份 CSV 可直接用于绘制该最佳 epoch 的 PCC 分布直方图。

#### 8. 绘图脚本

当前提供两个绘图脚本。

##### `Draw.py`

用于读取训练日志 `result*.txt` 并绘制训练曲线：

- loss 曲线
	- 输出 `*_loss.png`
- observed-only intensity PCC 曲线
	- 输出 `*_pcc_obs.png`
- observed-only charge PCC 曲线
	- 输出 `*_charge_pcc_obs.png`
- break 指标曲线
	- 输出 `*_break.png`

示例：

```bash
python /ssd/wangab/Top-down/NovaPredict/Draw.py /ssd/wangab/Top-down/NovaPredict/result_v2_1.txt
```

##### `draw_pcc_distribution.py`

用于读取最佳 epoch 的 PCC 分布 CSV，并绘制 `pcc_obs` 直方图。

- 默认输出：`*_hist.png`
- 支持自定义输出路径和 bins

示例：

```bash
python /ssd/wangab/Top-down/NovaPredict/draw_pcc_distribution.py /ssd/wangab/Top-down/NovaPredict/outputs/best_pcc_obs_distribution.csv
```

#### 9. 当前主要文件

- `Train.py`
	- 训练主程序，包含数据编码、模型、loss、训练循环、最优 PCC CSV 导出
- `Evaluate.py`
	- 评估函数，包含 intensity PCC、charge PCC、break precision/recall/f1
- `Draw.py`
	- 从训练日志绘制 loss / PCC / break 曲线
- `draw_pcc_distribution.py`
	- 从最佳 epoch PCC CSV 绘制分布直方图

#### 10. 当前输出文件示例

项目目录下已经生成过如下图文件：

- `result_v2_1_loss.png`
- `result_v2_1_pcc_obs.png`
- `result_v2_1_charge_pcc_obs.png`
- `result_v2_1_break.png`

