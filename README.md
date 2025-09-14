# ConvNeXt with Histopathology-Specific Augmentations for Mitotic Figure Classification

This repository contains the official code used in our solution to **Track 2 of the [MIDOG 2025 Challenge](https://midog2025.grand-challenge.org/)**, where the task is to distinguish **atypical mitotic figures (AMFs)** from **normal mitotic figures (NMFs)** in histopathology images.

🔗 This repository accompanies our preprint:  
[**ConvNeXt with Histopathology-Specific Augmentations for Mitotic Figure Classification**](https://arxiv.org/abs/2509.02595)  

---

## 📌 Overview

- **Architecture**: Lightweight ConvNeXt backbone (`ConvNeXt`-based)  
- **Datasets**:
  - AMi-Br  
  - AtNorM-Br  
  - AtNorM-MD  
  - OMG-Octo  
- **Key strategies**:
  - Histopathology-specific augmentations (elastic deformations, stain-based transformations, etc.)  
  - Balanced sampling to address class imbalance  
  - Grouped 5-fold cross-validation for robust evaluation  

📊 **Leaderboard performance**:  
**Balanced Accuracy = 0.8961** (Preliminary leaderboard)

---

## 🗂️ Repository Structure
```
├── data/ # Dataset utilities
│ └── dataset.py
├── models/ # Model definitions
│ └── convnextv2.py
├── utils/ # Custom augmentations & helper functions
│ └── augmentations.py
├── trainer.py # Training loop logic
├── train.py # Entry point for training
├── requirements.txt # Dependencies
└── README.md # Project documentation

```
## ⚙️ Installation

Clone the repository and install the required dependencies:

```bash
git clone https://github.com/HanaFEKI/ConvNeXt-with-Histopathology-Specific-Augmentations-for-Mitotic-Figure-Classification.git
cd ConvNeXt-with-Histopathology-Specific-Augmentations-for-Mitotic-Figure-Classification
pip install -r requirements.txt
