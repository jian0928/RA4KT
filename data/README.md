# Dataset Preparation

This directory contains the dataset preprocessing information, dataset split configurations, and knowledge concept graph construction resources used in the RA4KT experiments.

RA4KT is evaluated on two publicly available knowledge tracing benchmark datasets:

1. XES3G5M
2. ASSISTments2017

The original datasets are maintained and distributed by their respective providers. Due to the data usage policies of the original providers, the raw datasets are not redistributed in this repository. Users should download the datasets from the official sources and place them into the corresponding directories.

---

## 1. Directory Structure

The expected data directory structure is:

data/
 │
 ├── raw/
 │   ├── XES3G5M/
 │   └── ASSISTments2017/
 │
 ├── processed/
 │   ├── xes3g5m.pkl
 │   └── assist2017.pkl
 │
 ├── splits/
 │   ├── xes3g5m_split.json
 │   └── assist2017_split.json
 │
 └── kc_graph/
 ├── xes3g5m_graph.pkl
 └── assist2017_graph.pkl

---

# 2. Dataset Description

## 2.1 XES3G5M

XES3G5M is a large-scale primary school mathematics knowledge tracing dataset collected from a real online education platform.

Dataset statistics:

- Students: 18,066
- Interaction records: 5,549,635
- Knowledge concepts: 1,175
- Knowledge concept dependency edges: 1,304

The dataset contains:

- student identifiers
- exercise identifiers
- answering results
- knowledge concept annotations
- expert-provided knowledge concept relationships

The dataset is used to evaluate RA4KT in large-scale mathematics learning scenarios.

---

## 2.2 ASSISTments2017

ASSISTments2017 is a widely used knowledge tracing benchmark dataset collected from the ASSISTments educational platform.

Dataset statistics:

- Students: 9,428
- Interaction records: 1,495,104
- Knowledge concepts: 102

The dataset contains:

- student answering sequences
- correctness labels
- knowledge concept annotations

The dataset is used to evaluate the cross-scenario adaptability of RA4KT.

---

# 3. Data Preprocessing

The preprocessing pipeline follows the standard knowledge tracing protocol.

The preprocessing procedure includes:

1. Removing invalid samples with sequence length smaller than 5.
2. Truncating or padding sequences to maximum length 50.
3. Splitting datasets at the student level to avoid information leakage.
4. Constructing knowledge concept dependency graphs.
5. Selecting test samples with sufficient incorrect responses for reverse attribution evaluation.

The preprocessing script is: preprocess.py

Example:

```bash
python preprocess.py \
    --dataset XES3G5M \
    --input_dir data/raw/XES3G5M \
    --output_dir data/processed
```