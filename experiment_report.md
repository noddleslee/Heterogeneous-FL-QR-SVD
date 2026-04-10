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

Comparison of different ranks and heterogeneous LoRA algorithms after 10 communication rounds (Round 9 results).

| Metric | Fixed Rank 4 | Fixed Rank 16 | FlexLoRA (4-64) | HetLoRA (4-64) |
| :--- | :---: | :---: | :---: | :---: |
| **Yes/No accuracy** | 0.836 | **0.856** | 0.828 | 0.792 |
| **Open exact match** | 0.268 | **0.324** | 0.260 | 0.252 |
| **Open BLEU-1** | 0.3184 | **0.3885** | 0.3063 | 0.3088 |
| **Overall accuracy** | 0.5520 | **0.5900** | 0.5440 | 0.5220 |

---

## Key Findings

1. **Monotonic performance scaling with rank**: Performance consistently improves as LoRA rank increases (4 → 16 → 64) across all categories. In the local setting, rank=64 is superior, while in Federated Learning, **Fixed Rank 16** consistently outperforms other configurations, reaching a peak overall accuracy of **0.6040** (Round 6).

2. **Open-ended reasoning**: Higher rank continues to be critical for open-ended questions. Fixed Rank 16 maintains a significant lead in Open Exact Match (0.324) and BLEU-1 (0.3885) compared to Rank 4 and heterogeneous methods.

3. **Federated vs. Local**: Federated Learning significantly improves performance for lower-rank models. Fixed Rank 4 accuracy increased from **0.5095** (local) to **0.5520** (after 10 rounds of FL).

4. **Heterogeneous Algorithms**: 
   - **FlexLoRA** achieved an overall accuracy of **0.5440**, which is competitive with the Fixed Rank 4 baseline but slightly lower than Fixed Rank 16.
   - **HetLoRA** (with sparsity-weighted aggregation) reached **0.5220**. In this 4-client setup, the clients were assigned static ranks (4, 16, 32, 64) and pruning was not triggered as the weight norms remained stable.
   - Both methods show promise but may require more clients or more heterogeneous resource constraints to demonstrate clear advantages over fixed-rank baselines in this specific dataset.

5. **Stability**: Fixed Rank 16 showed the most stable improvement, while FlexLoRA and HetLoRA exhibited more fluctuations across rounds, likely due to the dynamic nature of their parameter updates.

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
