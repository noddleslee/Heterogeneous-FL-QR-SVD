# PathVQA 1-Epoch LoRA Rank Comparison Report
**Date:** 2026-04-08  
**Model:** Qwen3.5-0.8B (Mamba-hybrid)  
**Dataset:** PathVQA (19,654 train / 6,259 val)  
**Eval:** 2000 balanced samples (1000 yes/no + 1000 open)

---

## Training Configuration
| Config | Value |
|--------|-------|
| Base model | Qwen/Qwen3.5-0.8B |
| Epochs | 1 |
| Batch size | 64 (per device, no grad accum) |
| Eval batch size | 8 |
| Learning rate | 2e-4, linear schedule |
| Optimizer | adamw_8bit |
| Max seq length | 512 |
| Steps | ~308 |

---

## Results

| Metric | rank=4 | rank=16 | rank=64 | Δ (64 vs 4) |
|--------|--------|---------|---------|---|
| Trainable params | 3.3M (0.38%) | 13.2M (1.52%) | 52.7M (5.82%) | — |
| Train loss (final) | 0.9763 | 0.8421 | 0.6824 | -0.294 |
| Eval loss | 0.9827 | 0.9249 | 0.9039 | -0.079 |
| **Yes/No accuracy** | 0.807 | 0.832 | **0.869** | +0.062 |
| **Open exact match** | 0.212 | 0.261 | **0.319** | +0.107 |
| **Open BLEU-1** | 0.2695 | 0.3177 | **0.3713** | +0.102 |
| **Overall accuracy** | 0.5095 | 0.5465 | **0.5940** | +0.085 |

---

## Federated Learning Results (10 Rounds)

Comparison of different ranks and heterogeneous LoRA algorithms after 10 communication rounds.

| Metric | Fixed Rank 4 | Fixed Rank 16 | FlexLoRA (4-64) | HetLoRA (4-64) |
| :--- | :---: | :---: | :---: | :---: |
| **Yes/No accuracy** | 0.836 | 0.832 | **0.844** | 0.812 |
| **Open exact match** | 0.280 | **0.336** | 0.252 | 0.248 |
| **Open BLEU-1** | 0.3339 | **0.3940** | 0.3160 | 0.3044 |
| **Overall accuracy** | 0.5580 | **0.5840** | 0.5480 | 0.5300 |

---

## Key Findings

1. **Monotonic performance scaling with rank**: Performance consistently improves as LoRA rank increases (4 → 16 → 64) across all categories. Specifically, rank=16 successfully bridge the gap, capturing about 44% of the accuracy gain between rank=4 and rank=64 while using only 25% of the parameters of rank=64.

2. **Train loss fitting**: rank=64 fits training data much better than rank=16/4. However, the diminishing returns in eval loss (0.92 for r16 vs 0.90 for r64) suggest that rank=16 provides a very strong balance between parameter counts and generalization.

3. **Open-ended questions**: The advantage of higher-rank LoRA is particularly pronounced in open-ended VQA, where rank=64 and rank=16 significantly outperform rank=4. In Federated Learning, **Fixed Rank 16** continues to lead in open-ended accuracy (0.336 EM), suggesting that higher rank is essential for complex reasoning tasks even in a collaborative setting.

4. **Federated vs. Local**: Federated Learning significantly improves performance for lower-rank models. Fixed Rank 4 accuracy increased from **0.5095** (local) to **0.5580** (after 10 rounds of FL).

5. **Heterogeneous Algorithms**: In this 4-client setup, **FlexLoRA** (0.5480) and **HetLoRA** (0.5300) achieved comparable performance but were slightly outperformed by the Fixed Rank 4/16 baselines. This may be due to the small number of clients affecting the benefits of rank adaptation.

---

## Simulation Plots
![Federated Accuracy Comparison](file:///mimer/NOBACKUP/groups/bloom/shenghui/hfedlora/federated_accuracy_comparison.png)

---

## Infrastructure Optimizations Applied

| Issue | Fix | Impact |
|-------|-----|--------|
| Missing `causal_conv1d` | Installed with GCCcore/12.3.0 | 9s → 3.6s/step (2.5×) |
| Large batch | train_batch=64, no grad accum | Further ~20% speedup |
| Dataset caching | Arrow cache (~12GB train, ~4GB val) | Skip 60min preprocessing on rerun |
| `flash_attn` | Not fixable (GLIBC 2.28 < 2.32) | GPU util ~75% (not 100%) |

---

## Notes
- VQA eval uses logit-based scoring for yes/no questions (P(yes) vs P(no) at first token)
- Open questions use greedy generation (max_new_tokens=32) with exact match + BLEU-1
- Images in Arrow cache stored as PNG bytes; must be decoded with `Image.open(BytesIO(...))`
