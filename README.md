# ML_based_Intrusion_Detection-System

A supervised machine learning pipeline for network intrusion detection, trained on the CIC-IDS2017 dataset from the Canadian Institute for Cybersecurity. Detects 14 attack categories in real time from Zeek network logs.

Pipeline:

CIC-IDS2017 CSVs (8 files, 2.8M rows)
        ↓
Preprocessing — deduplicate, drop inf/NaN, strip identifiers
        ↓
Feature Selection — zero-variance → correlation filter → mutual information (top 20)
        ↓
Class Balancing — undersample BENIGN + SMOTE minority classes
        ↓
Train / Val / Test split (70 / 15 / 15, stratified)
        ↓
RandomForest vs XGBoost — best model selected by validation F1
        ↓
ids_model.pkl — used by live_ids.py for real-time inference

Project Structure:

ml-ids/
├── ids_full_pipeline.py      # training — preprocessing to evaluation
├── live_ids.py               # real-time Zeek log inference + alerts
├── setup_defender.sh         # defender VM setup (Ubuntu 22.04)
├── attack_commands.sh        # attacker scripts (Kali Linux)
└── ids_output/
    ├── ids_model.pkl
    ├── selected_features.txt
    ├── confusion_matrix.png
    ├── per_class_f1.png
    └── feature_importance.png

    

