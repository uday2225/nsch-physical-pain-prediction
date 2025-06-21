# Predicting Physical Pain in Children Using Random Forest Models

This project uses the 2018 NSCH dataset to predict physical pain outcomes in children using a Random Forest classification model.

## 🔍 Overview
- Dataset: 2018 National Survey of Children's Health (NSCH)
- Model: Random Forest with 1000 trees and cross-validated tuning
- Features: 25 top-ranked predictors (excluding pain-related leakage variables)
- Outcome: Binary classification of physical pain (yes/no)

## 📁 Repository Contents
- `scripts/`: R script for model training and evaluation
- `data/`: Original CSV used
- `outputs/`: Cleaned dataset, trained model, plots
- `report/`: Final polished report (PDF version)

## 📊 Key Results
- **Accuracy:** 92.68%
- **AUC:** 0.821
- **Sensitivity:** 10.27%
- **Specificity:** 99.66%

## 📦 Required Packages
- `tidyverse`
- `caret`
- `randomForest`
- `ROCR`
- `ggplot2`
- `readxl`

## 📜 License
MIT License (optional, if you're open to reuse)
## 👨‍💻 Author

**Uday Kiran Gogineni** – Clustering & Modeling Lead  
_M.S. in Bioinformatics_  
[LinkedIn](https://www.linkedin.com/in/udaykiran01)
