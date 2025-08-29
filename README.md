# ConvNeXt with Histopathology-Specific Augmentations for Mitotic Figure Classification
This repository contains our solution to **Track 2 of the MIDOG 2025 Challenge**, focusing on distinguishing **atypical mitotic figures (AMFs)** from **normal mitotic figures (NMFs)** in histopathology images.

---

## 📌 Overview
- **Architecture:** Lightweight ConvNeXt backbone
- **Datasets used:** 
  - AMi-Br
  - AtNorM-Br
  - AtNorM-MD
  - OMG-Octo
- **Key strategies:**
  - Histopathology-specific augmentations (elastic, stain-based)
  - Balanced sampling for class imbalance
  - Grouped 5-fold cross-validation
- **Leaderboard performance:** Balanced Accuracy = **0.8961** (Preliminary leaderboard)

---

## 🗂️ Repository Structure
