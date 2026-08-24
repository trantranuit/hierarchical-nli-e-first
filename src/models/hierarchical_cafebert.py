"""Hierarchical CafeBERT 2-head: shared h -> Head1(E/Non-E) + Head2(C/N)"""
from __future__ import annotations
import torch, torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel

class HierarchicalCafeBERT(nn.Module):
    """P+H -> CafeBERT -> h -> Head1(E/Non-E) + Head2(C/N)
    Loss: Gold=E      => L = L_coarse
          Gold=C/N    => L = L_coarse + lambda * L_fine
    """
    def __init__(self, model_name: str, fallback_models=None, dropout: float=0.1, lambda_fine: float=1.0,
                 gradient_checkpointing: bool=False, label_smoothing: float=0.0):
        super().__init__()
        self.lambda_fine = lambda_fine
        self.label_smoothing = label_smoothing
        self.backbone = self._load_backbone(model_name, fallback_models or [])
        if gradient_checkpointing:
            try:
                self.backbone.gradient_checkpointing_enable()
                print("[HierCafeBERT] gradient checkpointing enabled (trades compute for VRAM, exact gradients)")
            except Exception as e:
                print(f"[HierCafeBERT] gradient checkpointing not supported: {e}")
        hidden = self.backbone.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.head_coarse = nn.Linear(hidden, 2)  # E / Non-E
        self.head_fine   = nn.Linear(hidden, 2)  # C / N  (conditioned on Non-E)

    def _load_backbone(self, primary, fallbacks):
        for name in [primary] + fallbacks:
            try:
                print(f"[HierCafeBERT] trying {name}...")
                m = AutoModel.from_pretrained(name, trust_remote_code=True)
                print(f"[HierCafeBERT] loaded {name} ✓ hidden={m.config.hidden_size}")
                self.model_name_used = name
                return m
            except Exception as e:
                print(f"[HierCafeBERT] {name} failed: {e}")
                continue
        raise RuntimeError("No backbone could be loaded.")

    def forward(self, input_ids, attention_mask, token_type_ids=None,
                coarse_labels=None, fine_labels=None):
        out = self.backbone(input_ids=input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids if token_type_ids is not None else None)
        h = out.last_hidden_state[:, 0]
        h = self.dropout(h)
        coarse_logits = self.head_coarse(h)  # [B,2]
        fine_logits   = self.head_fine(h)    # [B,2]

        loss = None
        if coarse_labels is not None and fine_labels is not None:
            # per-example loss (Run A: giảm weight fine C/N qua lambda_fine, không phải qua batch-mean gộp
            # 2 loss có support size khác nhau — coarse tính trên cả batch, fine chỉ tính trên phần Non-E)
            coarse_loss = F.cross_entropy(coarse_logits, coarse_labels, reduction="none",
                                          label_smoothing=self.label_smoothing)
            fine_loss = torch.zeros_like(coarse_loss)
            mask = fine_labels != -100  # chỉ C/N
            if mask.any():
                fine_loss[mask] = F.cross_entropy(fine_logits[mask], fine_labels[mask], reduction="none",
                                                  label_smoothing=self.label_smoothing)
            loss = (coarse_loss + self.lambda_fine * fine_loss).mean()
        return {
            "coarse_logits": coarse_logits,
            "fine_logits": fine_logits,
            "loss": loss,
            "hidden": h,
        }

    @torch.no_grad()
    def predict_hard_soft(self, coarse_logits, fine_logits):
        """Compute hard & soft predictions from logits (offline use as well)."""
        # coarse: [B,2], fine: [B,2]
        p_coarse = F.softmax(coarse_logits, dim=-1)  # [B,2]  0:E 1:NonE
        p_fine   = F.softmax(fine_logits, dim=-1)    # [B,2]  0:C  1:N

        p_E = p_coarse[:, 0]
        p_NonE = p_coarse[:, 1]
        p_C_given = p_fine[:, 0]
        p_N_given = p_fine[:, 1]

        p_C = p_NonE * p_C_given
        p_N = p_NonE * p_N_given

        # stack for soft argmax over 3 classes order E,C,N
        p_soft = torch.stack([p_E, p_C, p_N], dim=1)  # [B,3]
        soft_pred = p_soft.argmax(dim=1)  # 0:E 1:C 2:N

        # hard routing: if coarse argmax == E -> E else fine argmax -> C/N
        coarse_pred = coarse_logits.argmax(dim=1)  # 0:E 1:NonE
        fine_pred   = fine_logits.argmax(dim=1)    # 0:C 1:N
        # map to flat ids: E=0, C=1, N=2
        hard_pred = torch.where(coarse_pred==0,
                                torch.zeros_like(soft_pred),  # E
                                torch.where(fine_pred==0,
                                            torch.ones_like(soft_pred),   # C
                                            torch.full_like(soft_pred, 2) # N
                                ))
        return {
            "p_E": p_E, "p_C": p_C, "p_N": p_N,
            "p_soft": p_soft,
            "soft_pred": soft_pred,
            "hard_pred": hard_pred,
            "coarse_pred": coarse_pred,
            "fine_pred": fine_pred,
        }
