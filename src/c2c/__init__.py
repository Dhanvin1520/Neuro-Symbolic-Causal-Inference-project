"""Shared code for the Corr2Cause pipeline. Each module belongs to one notebook:

* parsing.py  - notebook 02, read premises and hypotheses with regex
* graphs.py   - notebook 02, PC algorithm, equivalence class and claim rules
* features.py - notebook 03, structured features
* bayes.py    - notebook 05, DAG building and intervention effects

Phase 2:
* reader.py      - notebooks 06 and 07, rename variables to A-F and the wordings for the claim readers
* transformer.py - notebooks 07 and 08, fine-tune DistilBERT and predict
"""
