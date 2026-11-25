# Conditional Discrimination and Numeric Fidelity in a Frozen In-Context Autoencoder Memory

YANG LUOXIAO · aSSIST University, Seoul, Republic of Korea · 2026

This repository contains the controlled question-answering datasets, model outputs and analysis code used to compare candidate scoring with exact numeric generation. It includes the source data for both figures and all 23 manuscript tables. [Supplementary guide and file inventory](../README.md)

## Reproduce the results

With Python 3.10 or newer, run the commands below. The Git option preserves the original line endings used for file verification, including on Windows.

```shell
git clone -c core.autocrlf=false https://github.com/tendouarisu520/icae-numeric-fidelity.git
cd icae-numeric-fidelity
python -B reproduce.py
```

The standard check verifies the archived files and recalculates the digit-distance analysis and figure data. To run the additional CPU analyses, use `python -B reproduce.py --full`. [Reproduction guide](../README.md#reproduce-the-results)

## Licenses and attribution

The author's original training, evaluation and analysis code is available under the MIT License. This includes the original scripts distributed inside the evidence archives. [Software license](../LICENSE)

The experimental datasets, predictions, candidate scores, training logs and result tables are available under Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0) to the extent copyright or database rights apply. This also covers data inside the evidence archives. Credit YANG LUOXIAO for the study contributions, retain the source credits and identify changes. Shared adaptations use the same license or a compatible license as provided in the terms. [Data license](../LICENSE-DATA)

The natural-language passages and questions are adapted from SQuAD 2.0 by Pranav Rajpurkar, Robin Jia and Percy Liang, using articles by Wikipedia contributors. The study selected disjoint article sets, edited supporting answers and paired source passages with generated entity, condition and code records. The source records retain article titles and identifiers. [SQuAD 2.0 and its license](https://rajpurkar.github.io/SQuAD-explorer/) · [Source credits and modifications](../experiment_data/data/ATTRIBUTION.txt)

The ICAE reference implementation, model weights and external software dependencies have their own upstream terms. The asset manifest identifies the versions used in the study. [ICAE reference revision](https://github.com/getao/icae/tree/469a46886a92dd5e76b2d12a8bac0fb7ed7d4cdd) · [Asset manifest](../experiment_data/reproduction/upstream_assets.json)
