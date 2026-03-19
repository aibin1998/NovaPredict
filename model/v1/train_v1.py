import os
import math
import random
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from Evaluate import evaluate_spectral_similarity


# -----------------------
# 0) 配置
# -----------------------
@dataclass
class TrainConfig:
    core_tsv: str = "/ssd/wangab/DeepDeconv/Result/train_set/core_train_dedup.tsv"
    prsm_dir: str = "/ssd/wangab/DeepDeconv/Result/prsm"
    prsm_pattern: str = "prsm{pid}.csv"
    precursor_charge_col: str = "Charge"
    instrument_col: Optional[str] = None
    energy_col: Optional[str] = None

    # 输入
    max_len: int = 350  # proteoform 最大长度（氨基酸数）；超过截断
    aa_embed_dim: int = 64
    cond_embed_dim: int = 16  # 实验条件 embedding（碎裂方式、仪器、能量、前体电荷）
    max_precursor_charge: int = 30

    # encoder
    lstm_hidden: int = 256
    lstm_layers: int = 2
    dropout: float = 0.2

    # heads
    head_hidden: int = 128
    charge_num_classes: int = 10  # peak_charge 1..9 => 9类(含0 padding)；这里先给 10
    group_label_strategy: str = "min_error_row"  # 可选: min_error_row | aggregate

    # 训练
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    num_workers: int = 4
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # loss 权重（按需调整）
    w_intensity: float = 1.0
    w_dmass: float = 0.2
    w_charge: float = 0.2

    # 强度是否做归一化
    intensity_log1p: bool = True
    intensity_per_prsm_norm: bool = True


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -----------------------
# 1) 字典与特征
# -----------------------
AA_VOCAB = list("ACDEFGHIKLMNPQRSTVWY")  # 20 标准氨基酸
AA_TO_ID = {aa: i + 1 for i, aa in enumerate(AA_VOCAB)}  # 0 留给 padding/unknown
PAD_ID = 0

FRAG_TO_ID = {
    "CID": 1,
    "HCD": 2,
    "ETD": 3,
    "ECD": 4,
    "EThcD": 5,
    "UVPD": 6,
    "UNKNOWN": 0,
}

INSTRUMENT_TO_ID = {
    "UNKNOWN": 0,
    "QE": 1,
    "LUMOS": 2,
    "VELOS": 3,
    "ELITE": 4,
    "FUSION": 5,
    "ORBITRAP": 6,
}

ION_TO_ID = {"B": 0, "Y": 1}
NUM_ION_TYPES = len(ION_TO_ID)


def normalize_proteoform_sequence(seq: str) -> str:
    seq = str(seq).strip()
    parts = seq.split(".")
    if len(parts) >= 3:
        seq = ".".join(parts[1:-1])
    seq = re.sub(r"\[[^\]]*\]", "", seq)
    seq = "".join(ch for ch in seq.upper() if ch.isalpha())
    return seq


def encode_cleavage_sequence(seq: str, max_len: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """返回 left_ids/right_ids/cleavage_mask，长度均为 max_len-1。"""
    seq = normalize_proteoform_sequence(seq)
    max_cleavages = max(0, max_len - 1)
    left_ids = np.zeros((max_cleavages,), dtype=np.int64)
    right_ids = np.zeros((max_cleavages,), dtype=np.int64)
    cleavage_mask = np.zeros((max_cleavages,), dtype=np.float32)

    seq_len = min(len(seq), max_len)
    num_cleavages = max(0, seq_len - 1)

    for i in range(num_cleavages):
        left_ids[i] = AA_TO_ID.get(seq[i], PAD_ID)
        right_ids[i] = AA_TO_ID.get(seq[i + 1], PAD_ID)
        cleavage_mask[i] = 1.0

    return left_ids, right_ids, cleavage_mask, seq_len

def encode_fragmentation(frag: str) -> int:
    if pd.isna(frag):
        return FRAG_TO_ID["UNKNOWN"]
    frag = str(frag).strip()
    return FRAG_TO_ID.get(frag, FRAG_TO_ID["UNKNOWN"])


def encode_instrument(instrument: Optional[str]) -> int:
    if instrument is None or pd.isna(instrument):
        return INSTRUMENT_TO_ID["UNKNOWN"]
    instrument = str(instrument).strip().upper()
    return INSTRUMENT_TO_ID.get(instrument, INSTRUMENT_TO_ID["UNKNOWN"])

def encode_energy_to_bucket(energy: Optional[float]) -> int:
    """
    碎裂能量离散桶。你当前 core_train_dedup.tsv 看起来没有能量列，
    所以这里允许 None -> 0。
    """
    if energy is None or pd.isna(energy):
        return 0
    e = float(energy)
    # 简单分桶示例：0..100 -> 1..10
    b = int(min(9, max(0, math.floor(e / 10.0))))
    return b + 1


def encode_precursor_charge(charge: Optional[float], max_charge: int) -> int:
    if charge is None or pd.isna(charge):
        return 0
    value = int(round(float(charge)))
    return max(0, min(max_charge, value))


# -----------------------
# 2) 从 prsmX.csv 构建 fragment-level 标签序列
# -----------------------
def build_targets_from_prsm_csv(
    df_pr: pd.DataFrame,
    seq_len: int,
    max_len: int,
    cfg: TrainConfig,
) -> Dict[str, np.ndarray]:
    """
    构造 shape=(max_len-1, 2) 的 labels，第二维对应 b/y 两类离子。
    每个时间步代表一个碎裂点，位置 i 对应切在 aa[i] 和 aa[i+1] 之间。
    当第二维顺序固定为 [B, Y] 时，第 k 行表示 [b_(k+1), y_(L-k-1)]。
    标签：
      intensity: float32, padding=0
      dmass: float32, padding=0
      charge: int64, padding=0 (ignore_index)
      ion_mask: float32, 表示该碎裂点/离子类型是否有监督
            同一 (ion_type, ion_position) 下多条匹配的聚合方式由 cfg.group_label_strategy 控制：
                - min_error_row: 选取 |mass_error| 最小的单条记录；若 mass_error 不存在，则退化为 |ppm| 最小。
                - aggregate: intensity 取 max，dmass 取 mean，charge 取众数。
    """
    L = min(seq_len, max_len)
    valid_T = max(0, L - 1)
    T = max(0, max_len - 1)

    intensity = np.zeros((T, NUM_ION_TYPES), dtype=np.float32)
    dmass = np.zeros((T, NUM_ION_TYPES), dtype=np.float32)
    charge = np.zeros((T, NUM_ION_TYPES), dtype=np.int64)
    ion_mask = np.zeros((T, NUM_ION_TYPES), dtype=np.float32)

    if valid_T == 0 or df_pr is None or len(df_pr) == 0:
        return dict(intensity=intensity, dmass=dmass, charge=charge, ion_mask=ion_mask)

    use_dmass_col = "mass_error" if "mass_error" in df_pr.columns else ("ppm" if "ppm" in df_pr.columns else None)

    grouped = df_pr.dropna(subset=["ion_position", "ion_type"]).copy()
    grouped["ion_position"] = pd.to_numeric(grouped["ion_position"], errors="coerce").astype("Int64")
    grouped["ion_type"] = grouped["ion_type"].astype(str).str.upper()
    grouped = grouped.dropna(subset=["ion_position"])
    grouped = grouped[grouped["ion_type"].isin(ION_TO_ID)]

    if cfg.group_label_strategy not in {"min_error_row", "aggregate"}:
        raise ValueError(
            f"不支持的 group_label_strategy={cfg.group_label_strategy!r}，可选值为 'min_error_row' 或 'aggregate'"
        )

    for (ion_type, pos), g in grouped.groupby(["ion_type", "ion_position"]):
        ##计算断裂点位置
        pos = int(pos)
        if ion_type == "B":
            cleavage_idx = pos - 1
        else:
            cleavage_idx = L - pos - 1
        if cleavage_idx < 0 or cleavage_idx >= valid_T:
            continue

        ion_idx = ION_TO_ID[ion_type]
        if cfg.group_label_strategy == "min_error_row":
            best_row = None
            if use_dmass_col is not None:
                selector = pd.to_numeric(g[use_dmass_col], errors="coerce")
                selector = selector.abs().dropna()
                if len(selector) > 0:
                    best_row = g.loc[selector.idxmin()]

            if best_row is None:
                best_row = g.iloc[0]

            if "peak_intensity" in g.columns:
                val = pd.to_numeric(pd.Series([best_row["peak_intensity"]]), errors="coerce").iloc[0]
                if not pd.isna(val):
                    intensity[cleavage_idx, ion_idx] = float(val)

            if use_dmass_col is not None:
                val = pd.to_numeric(pd.Series([best_row[use_dmass_col]]), errors="coerce").iloc[0]
                if not pd.isna(val):
                    dmass[cleavage_idx, ion_idx] = float(val)

            if "peak_charge" in g.columns:
                val = pd.to_numeric(pd.Series([best_row["peak_charge"]]), errors="coerce").iloc[0]
                if not pd.isna(val):
                    ch = int(round(float(val)))
                    ch = max(0, min(cfg.charge_num_classes - 1, ch))
                    charge[cleavage_idx, ion_idx] = ch
        else:
            if "peak_intensity" in g.columns:
                vals = pd.to_numeric(g["peak_intensity"], errors="coerce").dropna()
                if len(vals) > 0:
                    intensity[cleavage_idx, ion_idx] = float(vals.max())

            if use_dmass_col is not None:
                vals = pd.to_numeric(g[use_dmass_col], errors="coerce").dropna()
                if len(vals) > 0:
                    dmass[cleavage_idx, ion_idx] = float(vals.mean())

            if "peak_charge" in g.columns:
                vals = pd.to_numeric(g["peak_charge"], errors="coerce").dropna()
                if len(vals) > 0:
                    ch = int(vals.round().astype(int).mode().iloc[0])
                    ch = max(0, min(cfg.charge_num_classes - 1, ch))
                    charge[cleavage_idx, ion_idx] = ch

        ion_mask[cleavage_idx, ion_idx] = 1.0

    if cfg.intensity_log1p:
        intensity = np.log1p(intensity)

    if cfg.intensity_per_prsm_norm:
        m = intensity.max()
        if m > 0:
            intensity = intensity / m

    return dict(intensity=intensity, dmass=dmass, charge=charge, ion_mask=ion_mask)


# -----------------------
# 3) Dataset
# -----------------------
class CorePrsmDataset(Dataset):
    def __init__(self, core_df: pd.DataFrame, cfg: TrainConfig):
        self.core = core_df.reset_index(drop=True)
        self.cfg = cfg
        required = ["Prsm ID", "Proteoform", "Fragmentation", cfg.precursor_charge_col]
        for c in required:
            if c not in self.core.columns:
                raise KeyError(f"core_train_dedup 缺少列: {c}")

        self.core["Prsm ID"] = pd.to_numeric(self.core["Prsm ID"], errors="coerce").astype("Int64")
        self.core = self.core.dropna(subset=["Prsm ID"]).reset_index(drop=True)

    def __len__(self):
        return len(self.core)

    def _load_prsm_csv(self, pid: int) -> pd.DataFrame:
        path = os.path.join(self.cfg.prsm_dir, self.cfg.prsm_pattern.format(pid=pid))
        if not os.path.exists(path):
            return pd.DataFrame()
        return pd.read_csv(path, low_memory=False)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.core.iloc[idx]
        pid = int(row["Prsm ID"])
        seq = str(row["Proteoform"])
        frag = row["Fragmentation"]

        energy = row[self.cfg.energy_col] if self.cfg.energy_col and self.cfg.energy_col in self.core.columns else None
        instrument = row[self.cfg.instrument_col] if self.cfg.instrument_col and self.cfg.instrument_col in self.core.columns else None
        precursor_charge = row[self.cfg.precursor_charge_col]

        left_ids, right_ids, cleavage_mask, seq_len = encode_cleavage_sequence(seq, self.cfg.max_len)
        frag_id = encode_fragmentation(frag)
        instrument_id = encode_instrument(instrument)
        e_bucket = encode_energy_to_bucket(energy)
        precursor_charge_id = encode_precursor_charge(precursor_charge, self.cfg.max_precursor_charge)

        df_pr = self._load_prsm_csv(pid)
        targets = build_targets_from_prsm_csv(df_pr, seq_len=seq_len, max_len=self.cfg.max_len, cfg=self.cfg)

        return {
            "prsm_id": torch.tensor(pid, dtype=torch.long),
            "left_ids": torch.from_numpy(left_ids),                     # (T,)
            "right_ids": torch.from_numpy(right_ids),                   # (T,)
            "cleavage_mask": torch.from_numpy(cleavage_mask),           # (T,)
            "frag_id": torch.tensor(frag_id, dtype=torch.long),         # ()
            "instrument_id": torch.tensor(instrument_id, dtype=torch.long),
            "e_bucket": torch.tensor(e_bucket, dtype=torch.long),
            "precursor_charge": torch.tensor(precursor_charge_id, dtype=torch.long),

            "y_intensity": torch.from_numpy(targets["intensity"]),    # (T, 2)
            "y_dmass": torch.from_numpy(targets["dmass"]),            # (T, 2)
            "y_charge": torch.from_numpy(targets["charge"]),          # (T, 2)
            "ion_mask": torch.from_numpy(targets["ion_mask"]),        # (T, 2)
        }


def collate_batch(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    # 全部都已 pad 到 max_len，所以直接 stack
    out = {}
    for k in batch[0].keys():
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out


# -----------------------
# 4) 模型：BiLSTM encoder + 多头
# -----------------------
class BiLSTMFragmentPredictor(nn.Module):
    def __init__(self, cfg: TrainConfig, aa_vocab_size: int = 21):
        super().__init__()
        self.cfg = cfg

        self.aa_emb = nn.Embedding(aa_vocab_size, cfg.aa_embed_dim, padding_idx=PAD_ID)

        self.frag_emb = nn.Embedding(max(FRAG_TO_ID.values()) + 1, cfg.cond_embed_dim)
        self.instrument_emb = nn.Embedding(max(INSTRUMENT_TO_ID.values()) + 1, cfg.cond_embed_dim)
        self.energy_emb = nn.Embedding(11, cfg.cond_embed_dim)
        self.precursor_charge_emb = nn.Embedding(cfg.max_precursor_charge + 1, cfg.cond_embed_dim)

        in_dim = 2 * cfg.aa_embed_dim + 4 * cfg.cond_embed_dim

        self.encoder = nn.LSTM(
            input_size=in_dim,
            hidden_size=cfg.lstm_hidden,
            num_layers=cfg.lstm_layers,
            dropout=cfg.dropout if cfg.lstm_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        enc_dim = 2 * cfg.lstm_hidden

        self.intensity_head = nn.Sequential(
            nn.Linear(enc_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, NUM_ION_TYPES),
        )
        self.dmass_head = nn.Sequential(
            nn.Linear(enc_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, NUM_ION_TYPES),
        )
        self.charge_head = nn.Sequential(
            nn.Linear(enc_dim, cfg.head_hidden),
            nn.ReLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.head_hidden, NUM_ION_TYPES * cfg.charge_num_classes),
        )

    def forward(self, left_ids, right_ids, cleavage_mask, frag_id, instrument_id, e_bucket, precursor_charge):
        """
        left_ids/right_ids: (B,T)
        cleavage_mask: (B,T) 1=valid cleavage, 0=pad
        frag_id: (B,)
        instrument_id: (B,)
        e_bucket: (B,)
        precursor_charge: (B,)
        """
        B, T = left_ids.shape

        left = self.aa_emb(left_ids)
        right = self.aa_emb(right_ids)

        frag = self.frag_emb(frag_id).unsqueeze(1).expand(B, T, -1)
        inst = self.instrument_emb(instrument_id).unsqueeze(1).expand(B, T, -1)
        ene = self.energy_emb(e_bucket).unsqueeze(1).expand(B, T, -1)
        prec = self.precursor_charge_emb(precursor_charge).unsqueeze(1).expand(B, T, -1)

        x = torch.cat([left, right, frag, inst, ene, prec], dim=-1)

        lengths = cleavage_mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed_out, _ = self.encoder(packed)
        h, _ = nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=T)

        pred_int = self.intensity_head(h)
        pred_dm = self.dmass_head(h)
        pred_ch = self.charge_head(h).view(B, T, NUM_ION_TYPES, self.cfg.charge_num_classes)

        return pred_int, pred_dm, pred_ch


# -----------------------
# 5) Loss（mask）
# -----------------------
def masked_l1(pred, target, mask):
    diff = (pred - target).abs() * mask
    denom = mask.sum().clamp(min=1.0)
    return diff.sum() / denom

def masked_ce(logits, target, mask, ignore_index: int = 0):
    """
    target: (B,T,2) long，padding=0
    mask: (B,T,2) float，1=use this position
    """
    _, _, _, C = logits.shape
    logits = logits.reshape(-1, C)
    target = target.reshape(-1)
    mask = mask.reshape(-1)

    ce = F.cross_entropy(logits, target, reduction="none", ignore_index=ignore_index)
    ce = ce * mask
    denom = mask.sum().clamp(min=1.0)
    return ce.sum() / denom


# -----------------------
# 6) 训练循环
# -----------------------
def train_one_epoch(model, loader, optim, cfg: TrainConfig):
    model.train()
    total = 0.0
    nstep = 0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(cfg.device)

        pred_int, pred_dm, pred_ch = model(
            batch["left_ids"],
            batch["right_ids"],
            batch["cleavage_mask"],
            batch["frag_id"],
            batch["instrument_id"],
            batch["e_bucket"],
            batch["precursor_charge"],
        )

        mask = batch["cleavage_mask"].unsqueeze(-1) * batch["ion_mask"]

        loss_int = masked_l1(pred_int, batch["y_intensity"], mask)
        loss_dm = masked_l1(pred_dm, batch["y_dmass"], mask)
        loss_ch = masked_ce(pred_ch, batch["y_charge"], mask, ignore_index=0)

        loss = cfg.w_intensity * loss_int + cfg.w_dmass * loss_dm + cfg.w_charge * loss_ch

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optim.step()

        total += float(loss.item())
        nstep += 1

    return total / max(1, nstep)


@torch.no_grad()
def validate(model, loader, cfg: TrainConfig):
    model.eval()
    total = 0.0
    nstep = 0
    for batch in loader:
        for k in batch:
            batch[k] = batch[k].to(cfg.device)

        pred_int, pred_dm, pred_ch = model(
            batch["left_ids"],
            batch["right_ids"],
            batch["cleavage_mask"],
            batch["frag_id"],
            batch["instrument_id"],
            batch["e_bucket"],
            batch["precursor_charge"],
        )
        mask = batch["cleavage_mask"].unsqueeze(-1) * batch["ion_mask"]

        loss_int = masked_l1(pred_int, batch["y_intensity"], mask)
        loss_dm = masked_l1(pred_dm, batch["y_dmass"], mask)
        loss_ch = masked_ce(pred_ch, batch["y_charge"], mask, ignore_index=0)
        loss = cfg.w_intensity * loss_int + cfg.w_dmass * loss_dm + cfg.w_charge * loss_ch

        total += float(loss.item())
        nstep += 1
    return total / max(1, nstep)


def main():
    cfg = TrainConfig()
    seed_everything(cfg.seed)

    core = pd.read_csv(cfg.core_tsv, sep="\t", low_memory=False)

    # 简单划分 train/val（按行随机；更严谨可按 Protein accession 或文件名分层）
    idx = np.arange(len(core))
    np.random.shuffle(idx)
    split = int(0.9 * len(idx))
    train_df = core.iloc[idx[:split]].reset_index(drop=True)
    val_df = core.iloc[idx[split:]].reset_index(drop=True)

    train_ds = CorePrsmDataset(train_df, cfg)
    val_ds = CorePrsmDataset(val_df, cfg)
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        collate_fn=collate_batch,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=(cfg.device.startswith("cuda")),
        collate_fn=collate_batch,
    )

    model = BiLSTMFragmentPredictor(cfg).to(cfg.device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = float("inf")
    for epoch in range(1, cfg.epochs + 1):
        tr = train_one_epoch(model, train_loader, optim, cfg)
        va = validate(model, val_loader, cfg)
        sim = evaluate_spectral_similarity(model, val_loader, cfg)
        print(
            f"epoch {epoch:03d} train_loss={tr:.5f} val_loss={va:.5f} "
            f"group_label_strategy={cfg.group_label_strategy} "
            # f"pcc_all_valid_med={sim['all_valid']['pcc_median']:.4f} "
            # f"pcc_all_valid_mean={sim['all_valid']['pcc_mean']:.4f} "
            # f"pcc_all_valid_max={sim['all_valid']['pcc_max']:.4f} "
            # f"pcc_all_valid_max_prsm={sim['all_valid']['pcc_max_prsm']} "
            f"pcc_obs_med={sim['observed_only']['pcc_median']:.4f} "
            f"pcc_obs_mean={sim['observed_only']['pcc_mean']:.4f} "
            f"pcc_obs_max={sim['observed_only']['pcc_max']:.4f} "
            f"pcc_obs_max_prsm={sim['observed_only']['pcc_max_prsm']} "
            f"pcc_obs>0.75={sim['observed_only']['pcc_0.75']:.3f} "
            f"pcc_obs>0.90={sim['observed_only']['pcc_0.90']:.3f} "
            f"n_all={sim['all_valid']['count']} n_obs={sim['observed_only']['count']}"
        )

        if va < best:
            best = va
            os.makedirs("./checkpoints", exist_ok=True)
            torch.save(
                {"model": model.state_dict(), "cfg": cfg.__dict__},
                "./checkpoints/bilstm_frag_multitask.pt",
            )

    print("best_val_loss:", best)


if __name__ == "__main__":
    main()