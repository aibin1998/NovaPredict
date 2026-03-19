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


@torch.no_grad()
def evaluate_spectral_similarity(model, loader, cfg: Any) -> Dict[str, float]:
	model.eval()
	all_valid_pcc = []
	observed_only_pcc = []
	all_valid_prsm = []
	observed_only_prsm = []
	for batch in loader:
		for k in batch:
			batch[k] = batch[k].to(cfg.device)

		pred_int, _, _, _ = model(
			batch["left_ids"],
			batch["right_ids"],
			batch["cleavage_mask"],
			batch["frag_id"],
			batch["instrument_id"],
			batch["e_bucket"],
			batch["precursor_charge"],
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

	return {
		"all_valid": summarize_pcc(all_valid_pcc, all_valid_prsm),
		"observed_only": summarize_pcc(observed_only_pcc, observed_only_prsm),
	}
