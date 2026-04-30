# BIOS 740: Final Project
Cassi Chen

## Named Entity Recognition (NER) and Relation Extraction (RE) on ADKG and MDKG

This work is based off the following repository: https://github.com/lavis-nlp/spert

### Abstract

Extraction from biomedical text remains challenging due to complex entity structures and long-range dependencies. In this work, we evaluate a span-based joint entity and relation extraction model (SpERT) on two PubMed-derived datasets, ADKG and MDKG. We first analyze dataset characteristics, noting class imbalance, and then assess model performance across entity and relation types. We further investigate cross-domain transfer learning by incorporating MDKG as auxiliary training data for ADKG. Results show that performance is dictated by high-frequency patterns but degrades on overlapping entities and long-distance relations. In addition, simple dataset merging does not improve the model in a cross-domain transfer learning setting.


Please read the paper for more information.


### Steps for Replication:

1. Get the datasets and types.
    python scripts/conversion/convert_adkg_mdkg.py
    python scripts/conversion/types_adkg_mdkg.py
    python scripts/conversion/combine_adkg_mdkg.py

2. EDA.
    eda.ipynb

3. Train data.
    python ./spert.py train --config configs/adkg_train.conf
    python ./spert.py train --config configs/mdkg_train.conf
    python ./spert.py train --config configs/adkg_mdkg_train.conf

4. Ensure compatability. Make sure paths are correct.
    python scripts/get_bin.py

5. Test data. Make sure paths are correct.
    python ./spert.py eval --config configs/adkg_eval.conf
    python ./spert.py eval --config configs/mdkg_eval.conf
    python ./spert.py eval --config configs/adkg_mdkg_eval.conf

6. Analysis. Make sure paths are correct.
    adkg_analysis.ipynb
    mdkg_analysis.ipynb



