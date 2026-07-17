# RA4KT
## Overview

This file implements the core RA4KT framework, the first reverse attribution method tailored for Knowledge Tracing (KT) scenarios. It breaks the black-box nature of deep learning-based KT models by generating causal, actionable, and pedagogically compliant reverse attribution sequences.

The output directly maps to a set of knowledge concepts (KCs) that students need to review to flip the KT model's prediction from *unmastered* to *mastered*, bridging the gap between model prediction and real-world teaching intervention.

Due to the intellectual property protection agreement of the author's institution, all model parameters and specific details will not be made public until the paper is officially published.

## Version

This repository corresponds to the experiments reported in the IEEE Access manuscript.

Release version:
RA4KT-v1.0

## Core Design Principles

RA4KT strictly follows the 5 core properties of high-quality reverse attribution for KT defined in the paper:

1. **Minimalism**: Achieves prediction flip with the minimum number of modifications to the original learning sequence, minimizing students' learning burden (implemented via Constraint Loss).
2. **Reality**: 100% guarantees that all modifications are executable in real educational scenarios (implemented via the Actionability Mask hard constraint mechanism).
3. **Generality**: Outputs explanations at the knowledge concept (KC) level, not individual question items, aligning with frontline teaching rules (implemented via deduplication of modified KCs).
4. **Coherence**: Ensures all intervention KCs have strong pedagogical correlation with the target KC (implemented via Coherence Loss based on the KC correlation graph).
5. **Ordinality**: Supports conversion of unordered KC sets to ordered teaching paths (paired with `postprocess.py` for TSPP-based path generation).

Quick Start Example

## Quick Start Example

```
import torch
import networkx as nx
from model.kt_models.dkt import DKT
from model.ra4kt import RA4KT

# ---------------------- 1. Load Pre-trained KT Model ----------------------
num_skills = 1175  # For XES3G5M dataset
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pre-trained DKT model (trained via train_kt.py)
kt_model = DKT(num_skills=num_skills, hidden_dim=100).to(device)
kt_model.load_state_dict(torch.load("checkpoints/dkt_xes3g5m.pth", map_location=device))
kt_model.eval()

# ---------------------- 2. Load KC Correlation Graph ----------------------
kc_graph = nx.read_gpickle("data/kc_graph_xes3g5m.gpkl")

# ---------------------- 3. Initialize RA4KT ----------------------
ra4kt = RA4KT(
    kt_model=kt_model,
    kc_graph=kc_graph,
    num_skills=num_skills,
    lambda_cons=,
    lambda_coh=,
    max_iter=,
    lr=,
    target_pred_threshold=,
    device=device
)

# ---------------------- 4. Prepare Student Sequence ----------------------
# Format: [(kc_id, correct), ..., (target_kc, 0)]
# Example: Student's learning sequence, with incorrect response on target KC 123
student_sequence = [
    (45, 1), (67, 0), (89, 1), (101, 0), (123, 0)  # Target KC: 123, incorrect
]

# ---------------------- 5. Generate Reverse Attribution ----------------------
cf_sequence, review_kcs = ra4kt.generate(student_sequence)

# ---------------------- 6. Output Results ----------------------
print(f"Original response sequence: {[r for kc, r in student_sequence]}")
print(f"Optimized counterfactual sequence: {cf_sequence}")
print(f"Knowledge concepts to review: {review_kcs}")
```

