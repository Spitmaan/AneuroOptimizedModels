# Stage 6 — Knowledge Distillation (Teacher-Student Pipeline)

**Project:** ANeurologic Phase 5 — Advanced Optimization
**Hardware:** NVIDIA Jetson Orin Nano 8 GB
**Date:** 2026-03-29 00:42
**Domain:** Aerospace Telemetry Classification

## Overview

Knowledge distillation (Hinton et al., 2015) transfers 'dark knowledge'
from a large teacher model into a tiny student model. The key insight:
a model's full probability distribution encodes richer information than
hard class labels.

```
Teacher (Llama-3.1-70B, remote API)
    │  Soft targets: [P(normal)=0.82, P(caution)=0.15, P(emergency)=0.03]
    │  (higher temperature T=4 → softer, more informative distributions)
    ▼
KL-Divergence Loss: L_KL = T² · KL(teacher_soft/T || student_soft/T)
    +
Cross-Entropy Loss: L_CE = CE(student_logits, hard_labels)
    ↓
Total: L = α·L_KL + (1-α)·L_CE    (α=0.7)
    ▼
Student (LFM2.5-1.2B or Llama-3.2-1B)
    Backbone: frozen + LoRA (rank=8, q_proj + v_proj)
    Head: ClassificationHead (linear 768→128→3)
```

## Dataset: Aerospace Telemetry

15 curated telemetry scenarios covering:
- Normal flight operations
- Caution-level anomalies (sensor faults, weather, fuel imbalance)
- Emergency conditions (stall, fire, hydraulic failure, overspeed)

Labels: `normal` / `caution` / `emergency`

## Configuration

| Parameter | Value |
|-----------|-------|
| Temperature (T) | 4.0 |
| Alpha (α) | 0.7 |
| LoRA rank | 8 |
| LoRA target | q_proj, v_proj |
| Optimizer | AdamW (lr=1e-4) |

## Results

### Qwen2.5-0.5B

| Metric | Before Distillation | After Distillation |
|--------|--------------------|--------------------|
| Accuracy (aerospace) | 26.67% | **53.33%** |
| Training loss (final) | — | 0.3549 |
| KL loss (final) | — | 0.0455 |
| Trainable params | — | 657667 |

**Training curve:**

| Epoch | Loss | KL | CE | Accuracy |
|-------|------|----|----|---------|
| 1 | 0.4027 | 0.0811 | 1.1532 | 26.67% |
| 2 | 0.3803 | 0.0593 | 1.1294 | 26.67% |
| 3 | 0.3636 | 0.0484 | 1.0991 | 26.67% |
| 4 | 0.3607 | 0.0477 | 1.0911 | 40.0% |
| 5 | 0.3549 | 0.0455 | 1.0768 | 40.0% |

## Dark Knowledge Transfer

The teacher (70B) provides probability distributions like:
- Emergency scenario: `[normal: 0.02, caution: 0.08, emergency: 0.90]`
- Ambiguous caution: `[normal: 0.15, caution: 0.72, emergency: 0.13]`

These soft targets tell the student:
1. How certain the teacher is (high entropy = ambiguous case)
2. Which confusable classes are related (caution/emergency often confused)
3. The relative severity of misclassifications

The student absorbs this via KL-divergence loss, learning not just
'is this an emergency?' but 'how emergency-like is this, relatively?'

## How to Reproduce

```bash
# With synthetic teacher (no API key needed):
docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage6_distillation/distill.py

# With real Llama-3.1-70B teacher via Together.ai:
TOGETHER_API_KEY=<key> docker exec aneurologic_phase5 python3 \
    /workspace/scripts/stage6_distillation/distill.py \
    --teacher-api openai \
    --api-base https://api.together.xyz/v1 \
    --teacher-model meta-llama/Llama-3.1-70B-Instruct-Turbo
```
