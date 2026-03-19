import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


REQUIRED_COLUMNS = {"prsm_id", "pcc_obs"}


def load_pcc_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        missing_text = ", ".join(sorted(missing))
        raise ValueError(f"CSV 缺少必要列: {missing_text}")

    df = df[[col for col in ["epoch", "prsm_id", "pcc_obs"] if col in df.columns]].copy()
    df["pcc_obs"] = pd.to_numeric(df["pcc_obs"], errors="coerce")
    df = df.dropna(subset=["pcc_obs"])
    if df.empty:
        raise ValueError(f"在 {csv_path} 中未找到有效的 pcc_obs 数值")
    return df


def build_output_path(csv_path: Path, output_path: str | None) -> Path:
    if output_path:
        target = Path(output_path).expanduser().resolve()
        target.parent.mkdir(parents=True, exist_ok=True)
        return target
    return csv_path.with_name(f"{csv_path.stem}_hist.png")


def plot_pcc_histogram(df: pd.DataFrame, csv_path: Path, output_path: Path, bins: int) -> None:
    epoch_text = ""
    if "epoch" in df.columns and not df["epoch"].empty:
        epoch_text = f"\nEpoch {int(df['epoch'].iloc[0])}"

    values = df["pcc_obs"].to_numpy()
    mean_value = float(df["pcc_obs"].mean())
    median_value = float(df["pcc_obs"].median())

    plt.figure(figsize=(10, 6))
    plt.hist(values, bins=bins, color="#4C78A8", edgecolor="white", alpha=0.9)
    plt.axvline(mean_value, color="#F58518", linestyle="--", linewidth=2, label=f"mean={mean_value:.4f}")
    plt.axvline(median_value, color="#54A24B", linestyle=":", linewidth=2, label=f"median={median_value:.4f}")
    plt.xlabel("pcc_obs")
    plt.ylabel("Count")
    plt.title(f"Observed-only PCC Distribution{epoch_text}\n{csv_path.name}")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="读取最佳 epoch 的 PCC CSV 并绘制 pcc_obs 分布直方图")
    parser.add_argument("csv_path", help="包含 prsm_id 和 pcc_obs 列的 CSV 文件路径")
    parser.add_argument("--output", default=None, help="输出图片路径；默认保存为同目录下 *_hist.png")
    parser.add_argument("--bins", type=int, default=30, help="直方图分箱数，默认 30")
    args = parser.parse_args()

    csv_path = Path(args.csv_path).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV 文件不存在: {csv_path}")
    if args.bins <= 0:
        raise ValueError("--bins 必须为正整数")

    df = load_pcc_csv(csv_path)
    output_path = build_output_path(csv_path, args.output)
    plot_pcc_histogram(df, csv_path, output_path, args.bins)
    print(f"saved plot: {output_path}")


if __name__ == "__main__":
    main()