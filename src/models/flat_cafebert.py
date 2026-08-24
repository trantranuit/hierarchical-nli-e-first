"""Flat CafeBERT: P+H -> CafeBERT -> Linear(3) -> E/C/N"""
from __future__ import annotations
import torch, torch.nn as nn
from transformers import AutoModel, AutoConfig

LABELS = ["E","C","N"]

class FlatCafeBERT(nn.Module):
    def __init__(self, model_name: str, fallback_models=None, dropout: float=0.1, num_labels: int=3,
                 gradient_checkpointing: bool=False, label_smoothing: float=0.0):
        super().__init__()
        self.num_labels = num_labels
        self.label_smoothing = label_smoothing
        self.backbone = self._load_backbone(model_name, fallback_models or [])
        if gradient_checkpointing:
            try:
                self.backbone.gradient_checkpointing_enable()
                print("[FlatCafeBERT] gradient checkpointing enabled (trades compute for VRAM, exact gradients)")
            except Exception as e:
                print(f"[FlatCafeBERT] gradient checkpointing not supported: {e}")
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden, num_labels)

    def _load_backbone(self, primary, fallbacks):
        for name in [primary] + fallbacks:
            try:
                print(f"[FlatCafeBERT] trying {name}...")
                m = AutoModel.from_pretrained(name, trust_remote_code=True)
                print(f"[FlatCafeBERT] loaded {name} ✓ hidden={m.config.hidden_size}")
                self.model_name_used = name
                return m
            except Exception as e:
                print(f"[FlatCafeBERT] {name} failed: {e}")
                continue
        raise RuntimeError("No backbone could be loaded. Check internet / model names.")

    def forward(self, input_ids, attention_mask, token_type_ids=None, labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids if token_type_ids is not None else None)
        # CLS pooling
        h = out.last_hidden_state[:, 0]  # [B, H]
        h = self.dropout(h)
        logits = self.classifier(h)  # [B, 3]
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=self.label_smoothing)(logits, labels)
        return {"logits": logits, "loss": loss, "hidden": h}
