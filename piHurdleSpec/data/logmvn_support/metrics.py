from typing import Any, Dict, List

import numpy as np
import torch


def pearson_corrcoef(pred: np.ndarray, target: np.ndarray) -> float:
	pred = np.asarray(pred, dtype=np.float64)
	target = np.asarray(target, dtype=np.float64)
	if pred.size < 2 or target.size < 2:
		return float("nan")

	pred = pred - pred.mean()
	target = target - target.mean()
	denom = np.linalg.norm(pred) * np.linalg.norm(target)
	if denom <= 1e-12:
		return float("nan")
	return float(np.dot(pred, target) / denom)


def summarize_pcc(pcc_values: List[float], prsm_ids: List[int]) -> Dict[str, float]:
	if len(pcc_values) == 0:
		return {
			"count": 0,
			"pcc_mean": float("nan"),
			"pcc_median": float("nan"),
			"pcc_0.75": float("nan"),
			"pcc_0.90": float("nan"),
			"pcc_max": float("nan"),
			"pcc_max_prsm": -1,
		}

	pcc = np.asarray(pcc_values, dtype=np.float64)
	max_idx = int(np.argmax(pcc))
	return {
		"count": int(pcc.size),
		"pcc_mean": float(np.mean(pcc)),
		"pcc_median": float(np.median(pcc)),
		"pcc_0.75": float(np.mean(pcc > 0.75)),
		"pcc_0.90": float(np.mean(pcc > 0.90)),
		"pcc_max": float(pcc[max_idx]),
		"pcc_max_prsm": int(prsm_ids[max_idx]),
	}


def batch_intensity_pcc(pred_int, target_int, cleavage_mask, ion_mask, prsm_ids) -> Dict[str, List[float]]:
	pred_int = pred_int.detach().cpu().numpy()
	target_int = target_int.detach().cpu().numpy()
	cleavage_mask = cleavage_mask.detach().cpu().numpy()
	ion_mask = ion_mask.detach().cpu().numpy()
	prsm_ids = prsm_ids.detach().cpu().numpy()

	pcc_all_valid = []
	pcc_observed_only = []
	prsm_all_valid = []
	prsm_observed_only = []
	batch_size = pred_int.shape[0]
	for i in range(batch_size):
		prsm_id = int(prsm_ids[i])
		valid_len = int(cleavage_mask[i].sum())
		if valid_len <= 0:
			continue

		pred_all_valid = np.clip(pred_int[i, :valid_len], a_min=0.0, a_max=None).reshape(-1)
		target_all_valid = target_int[i, :valid_len].reshape(-1)
		pcc = pearson_corrcoef(pred_all_valid, target_all_valid)
		if not np.isnan(pcc):
			pcc_all_valid.append(pcc)
			prsm_all_valid.append(prsm_id)

		valid_ion_mask = ion_mask[i, :valid_len].astype(bool)
		if valid_ion_mask.sum() <= 0:
			continue

		pred_vec = np.clip(pred_int[i, :valid_len], a_min=0.0, a_max=None)[valid_ion_mask]
		target_vec = target_int[i, :valid_len][valid_ion_mask]
		pcc = pearson_corrcoef(pred_vec, target_vec)
		if not np.isnan(pcc):
			pcc_observed_only.append(pcc)
			prsm_observed_only.append(prsm_id)

	return {
		"all_valid": pcc_all_valid,
		"observed_only": pcc_observed_only,
		"all_valid_prsm": prsm_all_valid,
		"observed_only_prsm": prsm_observed_only,
	}


def batch_break_vec_pcc(pred_int, pred_break, target_int, cleavage_mask, ion_mask, prsm_ids) -> Dict[str, List[float]]:
	pred_int = pred_int.detach().cpu().numpy()
	pred_break = torch.sigmoid(pred_break).detach().cpu().numpy()
	target_int = target_int.detach().cpu().numpy()
	cleavage_mask = cleavage_mask.detach().cpu().numpy()
	ion_mask = ion_mask.detach().cpu().numpy()
	prsm_ids = prsm_ids.detach().cpu().numpy()

	pcc_all_valid = []
	pcc_observed_only = []
	prsm_all_valid = []
	prsm_observed_only = []
	batch_size = pred_int.shape[0]
	for i in range(batch_size):
		prsm_id = int(prsm_ids[i])
		valid_len = int(cleavage_mask[i].sum())
		if valid_len <= 0:
			continue

		pred_break_vec = np.clip(pred_int[i, :valid_len], a_min=0.0, a_max=None) * pred_break[i, :valid_len]
		target_break_vec = target_int[i, :valid_len] * ion_mask[i, :valid_len]

		pcc = pearson_corrcoef(pred_break_vec.reshape(-1), target_break_vec.reshape(-1))
		if not np.isnan(pcc):
			pcc_all_valid.append(pcc)
			prsm_all_valid.append(prsm_id)

		valid_ion_mask = ion_mask[i, :valid_len].astype(bool)
		if valid_ion_mask.sum() <= 0:
			continue

		pcc = pearson_corrcoef(pred_break_vec[valid_ion_mask], target_break_vec[valid_ion_mask])
		if not np.isnan(pcc):
			pcc_observed_only.append(pcc)
			prsm_observed_only.append(prsm_id)

	return {
		"all_valid": pcc_all_valid,
		"observed_only": pcc_observed_only,
		"all_valid_prsm": prsm_all_valid,
		"observed_only_prsm": prsm_observed_only,
	}


def batch_charge_pcc(pred_ch, target_ch, cleavage_mask, ion_mask, prsm_ids) -> Dict[str, List[float]]:
	pred_charge = pred_ch.detach().argmax(dim=-1).cpu().numpy()
	target_ch = target_ch.detach().cpu().numpy()
	cleavage_mask = cleavage_mask.detach().cpu().numpy()
	ion_mask = ion_mask.detach().cpu().numpy()
	prsm_ids = prsm_ids.detach().cpu().numpy()

	pcc_observed_only = []
	prsm_observed_only = []
	batch_size = pred_charge.shape[0]
	for i in range(batch_size):
		prsm_id = int(prsm_ids[i])
		valid_len = int(cleavage_mask[i].sum())
		if valid_len <= 0:
			continue

		valid_ion_mask = ion_mask[i, :valid_len].astype(bool)
		if valid_ion_mask.sum() <= 1:
			continue

		pred_vec = pred_charge[i, :valid_len][valid_ion_mask]
		target_vec = target_ch[i, :valid_len][valid_ion_mask]
		pcc = pearson_corrcoef(pred_vec, target_vec)
		if not np.isnan(pcc):
			pcc_observed_only.append(pcc)
			prsm_observed_only.append(prsm_id)

	return {
		"observed_only": pcc_observed_only,
		"observed_only_prsm": prsm_observed_only,
	}


def summarize_break_metrics(tp: int, fp: int, fn: int, tn: int, threshold: float) -> Dict[str, float]:
	precision = float(tp / (tp + fp)) if (tp + fp) > 0 else float("nan")
	recall = float(tp / (tp + fn)) if (tp + fn) > 0 else float("nan")
	f1 = float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) > 0 else float("nan")
	return {
		"threshold": float(threshold),
		"tp": int(tp),
		"fp": int(fp),
		"fn": int(fn),
		"tn": int(tn),
		"precision": precision,
		"recall": recall,
		"f1": f1,
		"support": int(tp + fn),
		"predicted_positive": int(tp + fp),
	}


def batch_break_counts(pred_break, cleavage_mask, ion_mask, threshold: float) -> Dict[str, int]:
	pred_break = torch.sigmoid(pred_break).detach().cpu().numpy()
	cleavage_mask = cleavage_mask.detach().cpu().numpy().astype(bool)
	ion_mask = ion_mask.detach().cpu().numpy().astype(bool)

	tp = 0
	fp = 0
	fn = 0
	tn = 0
	batch_size = pred_break.shape[0]
	for i in range(batch_size):
		valid_len = int(cleavage_mask[i].sum())
		if valid_len <= 0:
			continue

		pred_vec = pred_break[i, :valid_len] >= threshold
		target_vec = ion_mask[i, :valid_len]

		tp += int(np.logical_and(pred_vec, target_vec).sum())
		fp += int(np.logical_and(pred_vec, np.logical_not(target_vec)).sum())
		fn += int(np.logical_and(np.logical_not(pred_vec), target_vec).sum())
		tn += int(np.logical_and(np.logical_not(pred_vec), np.logical_not(target_vec)).sum())

	return {"tp": tp, "fp": fp, "fn": fn, "tn": tn}


@torch.no_grad()
def evaluate_spectral_similarity(model, loader, cfg: Any) -> Dict[str, float]:
	model.eval()
	all_valid_pcc = []
	observed_only_pcc = []
	all_valid_prsm = []
	observed_only_prsm = []
	charge_observed_only_pcc = []
	charge_observed_only_prsm = []
	break_vec_all_valid_pcc = []
	break_vec_observed_only_pcc = []
	break_vec_all_valid_prsm = []
	break_vec_observed_only_prsm = []
	break_tp = 0
	break_fp = 0
	break_fn = 0
	break_tn = 0
	threshold = float(getattr(cfg, "break_prob_threshold", 0.5))
	eval_break_vec_pcc = bool(getattr(cfg, "eval_break_vec_pcc", False))
	for batch in loader:
		for k in batch:
			batch[k] = batch[k].to(cfg.device)

		pred_int, pred_ch, pred_break = model(
			batch["left_ids"],
			batch["right_ids"],
			batch["cleavage_mask"],
			batch["frag_id"],
			batch["instrument_id"],
			batch["e_bucket"],
			batch["precursor_charge"],
			batch["n_term_form_id"],
		)
		batch_pcc = batch_intensity_pcc(
			pred_int,
			batch["y_intensity"],
			batch["cleavage_mask"],
			batch["ion_mask"],
			batch["prsm_id"],
		)
		all_valid_pcc.extend(batch_pcc["all_valid"])
		observed_only_pcc.extend(batch_pcc["observed_only"])
		all_valid_prsm.extend(batch_pcc["all_valid_prsm"])
		observed_only_prsm.extend(batch_pcc["observed_only_prsm"])

		batch_charge = batch_charge_pcc(
			pred_ch,
			batch["y_charge"],
			batch["cleavage_mask"],
			batch["ion_mask"],
			batch["prsm_id"],
		)
		charge_observed_only_pcc.extend(batch_charge["observed_only"])
		charge_observed_only_prsm.extend(batch_charge["observed_only_prsm"])

		if eval_break_vec_pcc:
			batch_break_vec = batch_break_vec_pcc(
				pred_int,
				pred_break,
				batch["y_intensity"],
				batch["cleavage_mask"],
				batch["ion_mask"],
				batch["prsm_id"],
			)
			break_vec_all_valid_pcc.extend(batch_break_vec["all_valid"])
			break_vec_observed_only_pcc.extend(batch_break_vec["observed_only"])
			break_vec_all_valid_prsm.extend(batch_break_vec["all_valid_prsm"])
			break_vec_observed_only_prsm.extend(batch_break_vec["observed_only_prsm"])

		break_counts = batch_break_counts(
			pred_break,
			batch["cleavage_mask"],
			batch["ion_mask"],
			threshold,
		)
		break_tp += break_counts["tp"]
		break_fp += break_counts["fp"]
		break_fn += break_counts["fn"]
		break_tn += break_counts["tn"]

	result = {
		"all_valid": summarize_pcc(all_valid_pcc, all_valid_prsm),
		"observed_only": summarize_pcc(observed_only_pcc, observed_only_prsm),
		"charge_observed_only": summarize_pcc(charge_observed_only_pcc, charge_observed_only_prsm),
		"break": summarize_break_metrics(break_tp, break_fp, break_fn, break_tn, threshold),
		"raw": {
			"all_valid_pcc": list(all_valid_pcc),
			"all_valid_prsm": list(all_valid_prsm),
			"observed_only_pcc": list(observed_only_pcc),
			"observed_only_prsm": list(observed_only_prsm),
			"charge_observed_only_pcc": list(charge_observed_only_pcc),
			"charge_observed_only_prsm": list(charge_observed_only_prsm),
		},
	}
	if eval_break_vec_pcc:
		result["break_vec_all_valid"] = summarize_pcc(break_vec_all_valid_pcc, break_vec_all_valid_prsm)
		result["break_vec_observed_only"] = summarize_pcc(break_vec_observed_only_pcc, break_vec_observed_only_prsm)
		result["raw"]["break_vec_all_valid_pcc"] = list(break_vec_all_valid_pcc)
		result["raw"]["break_vec_all_valid_prsm"] = list(break_vec_all_valid_prsm)
		result["raw"]["break_vec_observed_only_pcc"] = list(break_vec_observed_only_pcc)
		result["raw"]["break_vec_observed_only_prsm"] = list(break_vec_observed_only_prsm)
	return result
