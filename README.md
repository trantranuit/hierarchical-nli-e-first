# Feasibility Test — Hierarchical NLI (E-first) — CafeBERT

> **ALL DATA MUST IN A FOLDER** — toàn bộ data, checkpoint, prediction, metrics đều nằm trong `~/hierarchical-nli-e-first/`.

## Mục tiêu
So sánh Flat `E/C/N` vs Hierarchical `Head1: E/Non-E + Head2: C/N` trên cùng backbone CafeBERT. Vòng đầu chỉ **2 training runs**.

```
Flat:         P+H → CafeBERT → Linear(3) → E/C/N
Hierarchical: P+H → CafeBERT → h → Head1(E/Non-E) + Head2(C/N)
```

## Cấu trúc folder
```
hierarchical-nli-e-first/
├── configs/config.yaml           # single source of truth (λ=1, lr, batch, seeds)
├── data/
│   ├── raw/{train,dev,test}.jsonl   # {id, premise, hypothesis, label∈E/C/N}
│   └── processed/
├── src/
│   ├── data/dataset.py           # NLIDataset + synthetic generator + GOLD→coarse/fine mapping
│   ├── models/{flat,hierarchical}_cafebert.py
│   ├── training/{train_flat,train_hierarchical}.py
│   └── evaluation/{metrics,hard_soft_inference,compare}.py
├── scripts/{run_pipeline.sh, evaluate_offline.py}
└── outputs/
    ├── flat_checkpoints/best/    # Run1
    ├── hier_checkpoints/best/    # Run2
    ├── predictions/{flat,hier}/  # CSVs per spec
    ├── metrics/                  # JSON per split
    ├── diagnostics/              # CM + report
    └── comparison/               # comparison_report.md + comparison_metrics.csv
```

## Mapping & Loss (vòng đầu λ=1)
| Gold | Head1 | Head2 |
|------|-------|-------|
| E    | E     | bỏ qua (-100) |
| C    | Non-E | C |
| N    | Non-E | N |

```
Gold=E:        L = L_coarse
Gold=C hoặc N: L = L_coarse + λ·L_fine   (λ=1)
```

## Hard vs Soft (cùng checkpoint, không train thêm)
- **Hard:** `Head1=E → E ; Head1=Non-E → Head2→C/N`
- **Soft:** `P(E)=P(E) ; P(C)=P(Non-E)*P(C|Non-E) ; P(N)=P(Non-E)*P(N|Non-E)` → argmax

## File prediction cần lưu
### Flat
`sample_id, gold_label, logit_E, logit_C, logit_N, pred_flat`
### Hierarchical
`sample_id, gold_label, coarse_logit_E, coarse_logit_nonE, fine_logit_C, fine_logit_N`  
→ offline tính `hard_pred, p_E, p_C, p_N, soft_pred`

## Tracking (W&B) & Storage (Hugging Face Hub)
- **W&B — chỉ tracking**: hyperparameters, seed, lr, batch_size, train/dev loss, Accuracy, Macro-F1, F1 từng lớp, confusion matrix, run metadata. Project: `hierarchical-nli-e-first` (entity mặc định theo API key).
- **HF Hub — lưu trữ bắt buộc** (private repo riêng cho mỗi run, để tải lại model / chạy Hard-Soft / error analysis / reproduce mà không train lại):
  - `configs/config.yaml -> hf_hub.flat_repo_id` = `trinhtrantran122/hier-nli-e-first-flat-cafebert`
  - `configs/config.yaml -> hf_hub.hier_repo_id` = `trinhtrantran122/hier-nli-e-first-hier-cafebert`
  - Mỗi repo chứa: `checkpoint/` (pytorch_model.bin + tokenizer/config), `predictions/{dev,test}_predictions.csv` (per-sample logits đúng schema spec), `run_metadata.json`.
  - W&B run summary log kèm `hf_repo_id` + `hf_revision` (commit SHA) để trace đúng model/predictions ứng với từng run.
- Credentials nằm trong `.env` (đã gitignore, KHÔNG commit) — `WANDB_API_KEY`, `HF_TOKEN`. `src/utils/env.py` tự load khi import training scripts.
- `--mock` mode (test khung, không cần model/GPU) **không** push lên W&B/HF, chỉ ghi local CSV để verify pipeline.

## Chạy nhanh (không cần GPU — mock pipeline để verify khung, local only)
```bash
cd ~/hierarchical-nli-e-first
pip install -r requirements.txt  # hoặc uv pip
bash scripts/run_pipeline.sh --mock   # Steps 1-5 mock, ra đủ CSV/MD (local, không lên W&B/HF)
# hoặc step-by-step
python3 -m src.data.dataset  # gen synthetic nếu thiếu data/raw/*.jsonl
python3 -m src.training.train_flat --mock
python3 -m src.training.train_hierarchical --mock
python3 scripts/evaluate_offline.py
cat outputs/comparison/comparison_report.md
```

## Chạy thật (cần internet để pull CafeBERT + push W&B/HF; GPU khuyến nghị, CPU vẫn chạy được nhưng chậm)
```bash
bash scripts/run_pipeline.sh --real   # train thật 2 runs, tự log W&B + push HF Hub
# hoặc
python3 -m src.training.train_flat --config configs/config.yaml
python3 -m src.training.train_hierarchical --config configs/config.yaml
python3 scripts/evaluate_offline.py
```
CafeBERT priority: `uitnlp/CafeBERT` → `vinai/phobert-base` → `xlm-roberta-base` (auto fallback nếu model chính không tải được).  
Tokenizer fallback tương tự.

**⚠️ GPU cũ (kiến trúc Pascal trở về trước)**: `requirements.txt` **không** liệt kê `torch` — cố tình, vì torch mới nhất trên PyPI có thể đã bỏ hỗ trợ compute capability của GPU đời cũ (`pip install torch` sẽ ghi đè bản torch máy/image đã cài sẵn khớp GPU, gây lỗi `CUDA error: no kernel image is available`). Dùng đúng torch có sẵn trong môi trường (Colab tự cài khớp GPU runtime đã chọn); nếu máy chưa có torch, tự cài theo đúng CUDA của GPU tại pytorch.org.

## Chạy trên Google Colab (khuyến nghị — free GPU T4)
Mở `notebooks/colab_train.ipynb` bằng Google Colab (File → Upload notebook, hoặc mở thẳng từ GitHub qua `File → Open notebook → GitHub` rồi dán URL repo). Notebook tự: clone repo → cài deps → nạp secrets từ **Colab Secrets** (🔑 ở sidebar trái, thêm `WANDB_API_KEY` + `HF_TOKEN`, bật "Notebook access") → in toàn bộ config → kéo data thật ViANLI → train Flat + Hierarchical thật (log W&B + push HF Hub) → eval offline → in bảng so sánh cuối. `Runtime → Change runtime type → GPU (T4)` trước khi chạy. Mỗi cell shell đều `assert _exit_code == 0` nên training crash sẽ dừng notebook ngay tại chỗ thay vì âm thầm chạy tiếp.

## Các bước thực nghiệm (5 bước)
| Bước | Thực nghiệm | Đánh giá | Đã implement |
|------|-------------|----------|--------------|
| 1 | Flat 3-class | Accuracy, Macro-F1, F1 từng lớp | `train_flat.py` |
| 2 | Diagnostic Flat | `E↔C, E↔N, C↔N` CM | `evaluate_offline.py` + `metrics.py` |
| 3 | Hierarchical Multi-task | Macro-F1, F1 coarse/fine | `train_hierarchical.py` |
| 4A | Hard inference | Macro-F1, CM, error propagation | `hard_soft_inference.py` |
| 4B | Soft inference | same | `hard_soft_inference.py` |
| 5 | So sánh cuối | Flat vs Hard vs Soft | `compare.py` → `comparison_report.md` |

## GO / STOP
- **GO** nếu Hier Macro-F1 > Flat, F1 C/N tăng hoặc `C↔N` giảm, Head1 E/Non-E tốt.
- **STOP** nếu Hier kém Flat rõ, Head1 nhầm nhiều `C/N→E`, không cải thiện `C↔N`.

## Data: synthetic (mặc định local) vs ViANLI thật
Nếu `data/raw/*.jsonl` chưa có, pipeline tự gen synthetic (Vi template, 3000/500/800, seed 42) — chỉ để test khung nhanh, không dùng cho kết quả thật.

Để train trên data thật **ViANLI** (`uitnlp/ViANLI` trên HF Hub, train 8012 / dev 1000 / test 1000, không gated):
```bash
python3 scripts/prepare_vianli.py --config configs/config.yaml
```
Script này tải `vianli_{train,dev,test}.jsonl`, đổi `uid→id` và map nhãn (`entailment→E, contradiction→C, neutral→N`), rồi **ghi đè** `data/raw/*.jsonl`. Sau đó `load_or_generate()` thấy file đã có nên sẽ không sinh synthetic nữa — chạy `train_flat.py`/`train_hierarchical.py` bình thường là dùng đúng ViANLI. Notebook Colab (`notebooks/colab_train.ipynb`) đã tự động gọi script này trước khi train.

## Requirements
`torch, transformers, datasets, scikit-learn, pandas, numpy, tqdm, pyyaml, accelerate, matplotlib, seaborn, wandb, huggingface_hub, python-dotenv`

---
*Updated: 2026-08-24 — E-first feasibility · 2 training runs · W&B tracking + HF Hub mandatory storage · Hard/Soft offline from same checkpoint*
