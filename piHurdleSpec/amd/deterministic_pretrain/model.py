"""Residual CNN-BiLSTM NovaPredict backbone with baseline conditions only."""
from __future__ import annotations

import os
import sys
import torch
import torch.nn as nn

RELEASE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, RELEASE_ROOT)
from data.logmvn_support.compat import (  # noqa: E402
    FRAG_TO_ID, INSTRUMENT_TO_ID, N_TERM_FORM_TO_ID, NUM_ION_TYPES, PAD_ID,
)


class CNNBiLSTMFragmentPredictor(nn.Module):
    """Local residual CNN before the existing global BiLSTM encoder."""

    effective_receptive_field = 7

    def __init__(self, cfg, aa_vocab_size=21, encoder_layers=None):
        super().__init__()
        encoder_layers = cfg.lstm_layers if encoder_layers is None else encoder_layers
        if encoder_layers < 1:
            raise ValueError("encoder_layers must be positive")
        self.cfg = cfg
        self.encoder_layers = encoder_layers
        self.backbone_name = f"residual_cnn_bilstm_e{encoder_layers}_rf7"
        d = cfg.cond_embed_dim
        self.aa_emb = nn.Embedding(aa_vocab_size, cfg.aa_embed_dim, padding_idx=PAD_ID)
        self.frag_emb = nn.Embedding(max(FRAG_TO_ID.values()) + 1, d)
        self.instrument_emb = nn.Embedding(max(INSTRUMENT_TO_ID.values()) + 1, d)
        self.energy_emb = nn.Embedding(11, d)
        self.precursor_charge_emb = nn.Embedding(cfg.max_precursor_charge + 1, d)
        self.n_term_form_emb = nn.Embedding(max(N_TERM_FORM_TO_ID.values()) + 1, d)

        self.input_dim = 2 * cfg.aa_embed_dim + 5 * d
        self.cnn_input_norm = nn.LayerNorm(self.input_dim)
        self.local_cnn = nn.Sequential(
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Conv1d(self.input_dim, self.input_dim, kernel_size=5, padding=2),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
        )
        self.encoder = nn.LSTM(
            input_size=self.input_dim, hidden_size=cfg.lstm_hidden, num_layers=encoder_layers,
            dropout=cfg.dropout if encoder_layers > 1 else 0.0,
            bidirectional=True, batch_first=True,
        )
        enc_dim = 2 * cfg.lstm_hidden
        self.intensity_head = nn.Sequential(nn.Linear(enc_dim, cfg.head_hidden), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.head_hidden, NUM_ION_TYPES))
        self.charge_head = nn.Sequential(nn.Linear(enc_dim, cfg.head_hidden), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.head_hidden, NUM_ION_TYPES * cfg.charge_num_classes))
        self.break_head = nn.Sequential(nn.Linear(enc_dim, cfg.head_hidden), nn.ReLU(), nn.Dropout(cfg.dropout), nn.Linear(cfg.head_hidden, NUM_ION_TYPES))

    def forward(self, left_ids, right_ids, cleavage_mask, frag_id, instrument_id,
                e_bucket, precursor_charge, n_term_form_id):
        batch, steps = left_ids.shape
        expand = lambda value: value.unsqueeze(1).expand(batch, steps, -1)
        x = torch.cat([
            self.aa_emb(left_ids), self.aa_emb(right_ids),
            expand(self.frag_emb(frag_id)), expand(self.instrument_emb(instrument_id)),
            expand(self.energy_emb(e_bucket)), expand(self.precursor_charge_emb(precursor_charge)),
            expand(self.n_term_form_emb(n_term_form_id)),
        ], dim=-1)

        position_mask = cleavage_mask.unsqueeze(-1).to(dtype=x.dtype)
        cnn_input = self.cnn_input_norm(x) * position_mask
        local = self.local_cnn(cnn_input.transpose(1, 2)).transpose(1, 2) * position_mask
        x = (x + local) * position_mask

        lengths = cleavage_mask.sum(dim=1).long().clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        packed, _ = self.encoder(packed)
        hidden, _ = nn.utils.rnn.pad_packed_sequence(packed, batch_first=True, total_length=steps)
        pred_intensity = self.intensity_head(hidden)
        pred_charge = self.charge_head(hidden).view(batch, steps, NUM_ION_TYPES, self.cfg.charge_num_classes)
        return pred_intensity, pred_charge, self.break_head(hidden)
