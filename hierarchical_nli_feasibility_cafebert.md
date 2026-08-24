# Feasibility Test --- Hierarchical NLI (E-first)

## Mục tiêu

So sánh NLI 3 lớp truyền thống với mô hình phân tầng:

``` text
Flat:
Premise + Hypothesis → CafeBERT → E / C / N

Hierarchical:
Premise + Hypothesis → CafeBERT → shared representation
                                  ↙                  ↘
                        Head 1: E / Non-E       Head 2: C / N
```

Vòng đầu **chỉ train 2 model**: 1. Flat CafeBERT. 2. Hierarchical
CafeBERT 2-head.

Hard/Soft chỉ dùng output của model Hierarchical, không train thêm.

------------------------------------------------------------------------

## Các bước thực nghiệm

  --------------------------------------------------------------------------------------------------------------------------
  Bước        Thực nghiệm      Pipeline / Implement                                 Đánh giá    Tại sao cần?  Cần lưu
  ----------- ---------------- ---------------------------------------------------- ----------- ------------- --------------
  **1**       **Flat 3-class   `P+H → CafeBERT → Linear(3) → E/C/N`                 Accuracy,   Baseline      checkpoint,
              baseline**                                                            Macro-F1,   chuẩn và xem  gold, 3
                                                                                    F1 từng lớp model đang    logits,
                                                                                                nhầm lớp nào  prediction,
                                                                                                              metrics

  **2**       **Diagnostic     Đọc output Bước 1 → confusion matrix                 `E↔C`,      Kiểm tra      confusion
              Flat**                                                                `E↔N`, đặc  E-first       matrix, lỗi
                                                                                    biệt `C↔N`  hierarchy có  từng cặp, F1
                                                                                                hợp lý không  từng lớp

  **3**       **Hierarchical   `P+H → CafeBERT → h → Head1(E/Non-E) + Head2(C/N)`   Macro-F1,   Test          checkpoint,
              Multi-task**                                                          F1 từng     hierarchy có  gold, 2 coarse
                                                                                    lớp, F1     tốt hơn Flat  logits, 2 fine
                                                                                    coarse, F1  không         logits, losses
                                                                                    fine                      

  **4A**      **Hard           Head1 = E → E; Head1 = Non-E → Head2 chọn C/N        Macro-F1,   Kiểm tra      `hard_pred`,
              inference**                                                           F1 E/C/N,   routing cứng  metrics
                                                                                    confusion   và error      
                                                                                    matrix      propagation   

  **4B**      **Soft           `P(E)=P(E)`; `P(C)=P(NonE)×P(C\|NonE)`;              Macro-F1,   Xem soft      `p_E`, `p_C`,
              inference**      `P(N)=P(NonE)×P(N\|NonE)`                            F1 E/C/N,   routing có    `p_N`,
                                                                                    confusion   giảm lỗi      `soft_pred`,
                                                                                    matrix      không         metrics

  **5**       **So sánh cuối** `Flat vs Hier-Hard vs Hier-Soft`                     Macro-F1    Quyết định có bảng kết quả +
                                                                                    chính,      tiếp tục      error analysis
                                                                                    Accuracy,   hướng này     
                                                                                    F1 từng     không         
                                                                                    lớp, `C↔N`                
  --------------------------------------------------------------------------------------------------------------------------

------------------------------------------------------------------------

## Train Hierarchical

Mapping:

  Gold   Head 1   Head 2
  ------ -------- --------
  E      E        bỏ qua
  C      Non-E    C
  N      Non-E    N

Loss:

``` text
Gold = E:
L = L_coarse

Gold = C hoặc N:
L = L_coarse + λ × L_fine
```

Vòng đầu dùng:

``` text
λ = 1
```

------------------------------------------------------------------------

## Hard vs Soft

### Hard

``` text
Head1
├── E → final E
└── Non-E
      ↓
    Head2
    ├── C → final C
    └── N → final N
```

### Soft

Ví dụ:

``` text
Head1:
P(E) = 0.40
P(Non-E) = 0.60

Head2:
P(C|Non-E) = 0.70
P(N|Non-E) = 0.30
```

Final:

``` text
P(E) = 0.40
P(C) = 0.60 × 0.70 = 0.42
P(N) = 0.60 × 0.30 = 0.18
```

→ chọn class có probability cao nhất.

**Hard và Soft dùng cùng checkpoint, không train lại.**

------------------------------------------------------------------------

## File prediction cần lưu

### Flat

``` text
sample_id
gold_label
logit_E
logit_C
logit_N
pred_flat
```

### Hierarchical

``` text
sample_id
gold_label
coarse_logit_E
coarse_logit_nonE
fine_logit_C
fine_logit_N
```

Từ logits Hierarchical có thể tính offline:

``` text
hard_pred
p_E
p_C
p_N
soft_pred
```

------------------------------------------------------------------------

## GO / STOP

**GO** nếu Hierarchical: - Macro-F1 tốt hơn Flat; - F1 C/N tăng hoặc
`C↔N` confusion giảm; - Head1 `E/Non-E` đủ tốt.

**STOP / reconsider** nếu: - Hierarchical kém Flat rõ ràng; - Head1 nhầm
nhiều `C/N → E`; - hierarchy không cải thiện lỗi `C↔N`.

------------------------------------------------------------------------

## Resource vòng đầu

Chỉ có **2 training runs**:

``` text
Run 1: Flat CafeBERT
Run 2: E-first Hierarchical CafeBERT
```

Diagnostic, Hard, Soft và Comparison đều làm offline từ
prediction/logits đã lưu.
