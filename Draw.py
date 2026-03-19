import argparse
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt


LOSS_PATTERN = re.compile(
	r"epoch\s+(?P<epoch>\d+)\s+train_loss=(?P<train_loss>[-+]?\d*\.?\d+)\s+val_loss=(?P<val_loss>[-+]?\d*\.?\d+)"
)

METRIC_PATTERNS = {
	"pcc_obs_med": [
		re.compile(r"pcc_obs_med=(?P<value>[-+]?\d*\.?\d+)"),
		re.compile(r"val_pcc_med=(?P<value>[-+]?\d*\.?\d+)"),
	],
	"pcc_obs_mean": [
		re.compile(r"pcc_obs_mean=(?P<value>[-+]?\d*\.?\d+)"),
		re.compile(r"val_pcc_mean=(?P<value>[-+]?\d*\.?\d+)"),
	],
	"pcc_obs_max": [
		re.compile(r"pcc_obs_max=(?P<value>[-+]?\d*\.?\d+)"),
	],
	"break_prec": [
		re.compile(r"break_prec=(?P<value>[-+]?\d*\.?\d+)"),
	],
	"break_rec": [
		re.compile(r"break_rec=(?P<value>[-+]?\d*\.?\d+)"),
	],
	"break_f1": [
		re.compile(r"break_f1=(?P<value>[-+]?\d*\.?\d+)"),
	],
	"charge_pcc_obs_med": [
		re.compile(r"charge_pcc_obs_med=(?P<value>[-+]?\d*\.?\d+)"),
	],
	"charge_pcc_obs_mean": [
		re.compile(r"charge_pcc_obs_mean=(?P<value>[-+]?\d*\.?\d+)"),
	],
	"charge_pcc_obs_max": [
		re.compile(r"charge_pcc_obs_max=(?P<value>[-+]?\d*\.?\d+)"),
	],
}


def parse_loss_file(file_path: Path) -> Tuple[List[int], List[float], List[float]]:
	epochs: List[int] = []
	train_losses: List[float] = []
	val_losses: List[float] = []

	with file_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			match = LOSS_PATTERN.search(line)
			if match is None:
				continue
			epochs.append(int(match.group("epoch")))
			train_losses.append(float(match.group("train_loss")))
			val_losses.append(float(match.group("val_loss")))

	if not epochs:
		raise ValueError(f"在 {file_path} 中未找到 epoch/train_loss/val_loss 记录")

	return epochs, train_losses, val_losses


def extract_metric(line: str, metric_name: str) -> float | None:
	for pattern in METRIC_PATTERNS[metric_name]:
		match = pattern.search(line)
		if match is not None:
			return float(match.group("value"))
	return None


def parse_metric_file(file_path: Path, metric_names: List[str]) -> Tuple[List[int], Dict[str, List[float]]]:
	epochs: List[int] = []
	metrics: Dict[str, List[float]] = {name: [] for name in metric_names}

	with file_path.open("r", encoding="utf-8") as handle:
		for line in handle:
			loss_match = LOSS_PATTERN.search(line)
			if loss_match is None:
				continue

			parsed_values = {name: extract_metric(line, name) for name in metric_names}
			if all(value is None for value in parsed_values.values()):
				continue

			epochs.append(int(loss_match.group("epoch")))
			for name, value in parsed_values.items():
				if value is not None:
					metrics[name].append(value)

	valid_metrics = {name: values for name, values in metrics.items() if len(values) == len(epochs) and values}
	if not epochs or not valid_metrics:
		joined = "/".join(metric_names)
		raise ValueError(f"在 {file_path} 中未找到 {joined} 记录")

	return epochs, valid_metrics


def parse_pcc_file(file_path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
	return parse_metric_file(file_path, ["pcc_obs_med", "pcc_obs_mean", "pcc_obs_max"])


def parse_break_file(file_path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
	return parse_metric_file(file_path, ["break_prec", "break_rec", "break_f1"])


def parse_charge_pcc_file(file_path: Path) -> Tuple[List[int], Dict[str, List[float]]]:
	return parse_metric_file(file_path, ["charge_pcc_obs_med", "charge_pcc_obs_mean", "charge_pcc_obs_max"])


def plot_loss_curve(file_path: Path, output_path: Path) -> None:
	epochs, train_losses, val_losses = parse_loss_file(file_path)

	plt.figure(figsize=(10, 6))
	plt.plot(epochs, train_losses, label="train_loss", linewidth=2)
	plt.plot(epochs, val_losses, label="val_loss", linewidth=2)
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.title(f"Training vs Validation Loss\n{file_path.name}")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def plot_pcc_curve(file_path: Path, output_path: Path) -> None:
	epochs, metrics = parse_pcc_file(file_path)

	plt.figure(figsize=(10, 6))
	style_map = {
		"pcc_obs_med": {"label": "pcc_obs_med", "linewidth": 2.2},
		"pcc_obs_mean": {"label": "pcc_obs_mean", "linewidth": 2.0},
		"pcc_obs_max": {"label": "pcc_obs_max", "linewidth": 2.0},
	}
	for metric_name in ["pcc_obs_med", "pcc_obs_mean", "pcc_obs_max"]:
		if metric_name not in metrics:
			continue
		plt.plot(epochs, metrics[metric_name], label=style_map[metric_name]["label"], linewidth=style_map[metric_name]["linewidth"])

	plt.xlabel("Epoch")
	plt.ylabel("PCC")
	plt.title(f"Observed-only PCC Metrics\n{file_path.name}")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def plot_break_curve(file_path: Path, output_path: Path) -> None:
	epochs, metrics = parse_break_file(file_path)

	plt.figure(figsize=(10, 6))
	style_map = {
		"break_prec": {"label": "break_precision", "linewidth": 2.2},
		"break_rec": {"label": "break_recall", "linewidth": 2.0},
		"break_f1": {"label": "break_f1", "linewidth": 2.0},
	}
	for metric_name in ["break_prec", "break_rec", "break_f1"]:
		if metric_name not in metrics:
			continue
		plt.plot(epochs, metrics[metric_name], label=style_map[metric_name]["label"], linewidth=style_map[metric_name]["linewidth"])

	plt.xlabel("Epoch")
	plt.ylabel("Score")
	plt.title(f"Break Head Metrics\n{file_path.name}")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def plot_charge_pcc_curve(file_path: Path, output_path: Path) -> None:
	epochs, metrics = parse_charge_pcc_file(file_path)

	plt.figure(figsize=(10, 6))
	style_map = {
		"charge_pcc_obs_med": {"label": "charge_pcc_obs_med", "linewidth": 2.2},
		"charge_pcc_obs_mean": {"label": "charge_pcc_obs_mean", "linewidth": 2.0},
		"charge_pcc_obs_max": {"label": "charge_pcc_obs_max", "linewidth": 2.0},
	}
	for metric_name in ["charge_pcc_obs_med", "charge_pcc_obs_mean", "charge_pcc_obs_max"]:
		if metric_name not in metrics:
			continue
		plt.plot(epochs, metrics[metric_name], label=style_map[metric_name]["label"], linewidth=style_map[metric_name]["linewidth"])

	plt.xlabel("Epoch")
	plt.ylabel("PCC")
	plt.title(f"Observed-only Charge PCC Metrics\n{file_path.name}")
	plt.grid(True, linestyle="--", alpha=0.4)
	plt.legend()
	plt.tight_layout()
	plt.savefig(output_path, dpi=200)
	plt.close()


def resolve_result_files(input_path: Path) -> Iterable[Path]:
	if input_path.is_file():
		return [input_path]
	if input_path.is_dir():
		result_files = sorted(input_path.glob("result*.txt"))
		if result_files:
			return result_files
		raise FileNotFoundError(f"目录 {input_path} 下未找到 result*.txt")
	raise FileNotFoundError(f"路径不存在: {input_path}")


def build_output_path(file_path: Path, output_dir: Path | None) -> Path:
	target_dir = output_dir if output_dir is not None else file_path.parent
	target_dir.mkdir(parents=True, exist_ok=True)
	return target_dir / f"{file_path.stem}_loss.png"


def build_pcc_output_path(file_path: Path, output_dir: Path | None) -> Path:
	target_dir = output_dir if output_dir is not None else file_path.parent
	target_dir.mkdir(parents=True, exist_ok=True)
	return target_dir / f"{file_path.stem}_pcc_obs.png"


def build_break_output_path(file_path: Path, output_dir: Path | None) -> Path:
	target_dir = output_dir if output_dir is not None else file_path.parent
	target_dir.mkdir(parents=True, exist_ok=True)
	return target_dir / f"{file_path.stem}_break.png"


def build_charge_pcc_output_path(file_path: Path, output_dir: Path | None) -> Path:
	target_dir = output_dir if output_dir is not None else file_path.parent
	target_dir.mkdir(parents=True, exist_ok=True)
	return target_dir / f"{file_path.stem}_charge_pcc_obs.png"


def main() -> None:
	parser = argparse.ArgumentParser(description="绘制训练日志中的 loss、observed-only intensity/charge PCC 和 break 指标曲线")
	parser.add_argument(
		"input_path",
		nargs="?",
		default=".",
		help="结果文件路径，或包含 result*.txt 的目录；默认当前目录",
	)
	parser.add_argument(
		"--output-dir",
		default=None,
		help="输出图片目录；默认和结果文件同目录",
	)
	args = parser.parse_args()

	input_path = Path(args.input_path).expanduser().resolve()
	output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else None

	for file_path in resolve_result_files(input_path):
		output_path = build_output_path(file_path, output_dir)
		pcc_output_path = build_pcc_output_path(file_path, output_dir)
		break_output_path = build_break_output_path(file_path, output_dir)
		charge_pcc_output_path = build_charge_pcc_output_path(file_path, output_dir)
		plot_loss_curve(file_path, output_path)
		plot_pcc_curve(file_path, pcc_output_path)
		print(f"saved plot: {output_path}")
		print(f"saved plot: {pcc_output_path}")
		try:
			plot_charge_pcc_curve(file_path, charge_pcc_output_path)
		except ValueError:
			pass
		else:
			print(f"saved plot: {charge_pcc_output_path}")
		try:
			plot_break_curve(file_path, break_output_path)
		except ValueError:
			pass
		else:
			print(f"saved plot: {break_output_path}")


if __name__ == "__main__":
	main()
