# 🧬 QSAR Toxicity Predictor

![Python](https://img.shields.io/badge/Python-3.10-blue)
![RDKit](https://img.shields.io/badge/RDKit-Chemoinformatics-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)
![Status](https://img.shields.io/badge/Status-Active-brightgreen)

Applying QSAR machine learning to predict molecular toxicity — bridging
environmental contaminant research and computational drug safety screening.

**Author:** Agbailu Adejoke Adewumi  
**Background:** Environmental Science and Engineering / Molecular Dynamics  
**Field:** Cheminformatics / Computational Toxicology / QSAR Modelling

---

##  Motivation

My Master's thesis investigates how environmental pollutants (3-BHA)
penetrate and disrupt lipid bilayer membranes using Molecular Dynamics
simulations. This project extends that work computationally — applying
machine learning to rapidly screen molecular toxicity before conducting
more computationally intensive molecular simulations.

QSAR (Quantitative Structure-Activity Relationship) modelling is widely
used in both environmental risk assessment and pharmaceutical drug safety
screening (ADMET). This project sits at that intersection.

---

##  Pipeline
```
SMILES String
      ↓
RDKit Morgan Fingerprints (radius=2, 2048 bits)
      ↓
Feature Matrix (~8,000 molecules × 2048 features)
      ↓
Random Forest classifiers (one model trained per toxicity target)
      ↓
Toxicity Predictions + ROC-AUC Evaluation
      ↓
Interactive Streamlit Web App
```

---

##  What This Does

- Converts molecules (SMILES strings) into Morgan fingerprints using **RDKit**
- Trains and benchmarks **Random Forest, SVM, and XGBoost** classifiers
- Evaluates model performance using **ROC-AUC score**
- Visualises results with ROC curves and feature importance plots
- Interactive **Streamlit web app** for predicting toxicity of any molecule

---

##  Model Performance

Mean ROC-AUC across 12 toxicity targets: **0.793**

| Target | Biological Meaning | ROC-AUC |
|--------|--------------------|---------|
| NR-AR | Androgen receptor | 0.819 |
| NR-AR-LBD | Androgen receptor ligand binding | 0.841 |
| NR-AhR | Aryl hydrocarbon receptor | 0.877 |
| NR-Aromatase | Aromatase inhibition | 0.811 |
| NR-ER | Estrogen receptor | 0.733 |
| NR-ER-LBD | Estrogen receptor ligand binding | 0.766 |
| NR-PPAR-gamma | Peroxisome proliferator receptor | 0.698 |
| SR-ARE | Oxidative stress response | 0.752 |
| SR-ATAD5 | Genotoxicity | 0.772 |
| SR-HSE | Heat shock response | 0.790 |
| SR-MMP | Mitochondrial membrane disruption | 0.858 |
| SR-p53 | DNA damage response | 0.821 |
| **Mean** | | **0.793** |
```

---

##  Example Prediction

Input SMILES: `COc1ccc(O)cc1C(C)(C)C`  
Compound: **3-BHA** (Butylated Hydroxyanisole — my Masters thesis compound)

| Target | Toxicity Probability |
|--------|---------------------|
| NR-AhR | 0.54 |
| NR-ER | 0.63 |
| NR-ER-LBD | 0.61 |
| SR-ARE | 0.55 |
| SR-MMP | 0.74 |
| SR-p53 | 0.52 |

> 🔴 High toxicity predicted for SR-MMP (mitochondrial membrane disruption) 
```
---

##  Toxicity Targets (Tox21)

| Target | Biological Meaning |
|--------|--------------------|
| NR-AR | Androgen receptor |
| NR-AhR | Aryl hydrocarbon receptor |
| NR-ER | Estrogen receptor |
| NR-PPAR-gamma | Peroxisome proliferator receptor |
| SR-ARE | Oxidative stress response |
| SR-p53 | DNA damage response |
| SR-MMP | Mitochondrial membrane disruption |

---

##  Installation
```bash
git clone https://github.com/adejoke-adewumi/qsar-toxicity-predictor.git
cd qsar-toxicity-predictor

python -m venv qsar_env
qsar_env\Scripts\activate        # Windows
source qsar_env/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

---

##  Usage
```bash
# Step 1: Download Tox21 dataset
python -c "
import urllib.request, gzip, shutil, os
os.makedirs('data', exist_ok=True)
urllib.request.urlretrieve(
    'https://deepchemio.s3-us-west-1.amazonaws.com/datasets/tox21.csv.gz',
    'data/tox21.csv.gz')
with gzip.open('data/tox21.csv.gz','rb') as f_in:
    with open('data/tox21.csv','wb') as f_out:
        shutil.copyfileobj(f_in, f_out)
print('Dataset ready!')
"

# Step 2: Train and benchmark models
python src/model_rf.py

# Step 3: Launch the web app
streamlit run app/streamlit_app.py
```

---

##  Project Structure
```
qsar-project/
├── src/
│   ├── features.py        # SMILES → Morgan fingerprint conversion (RDKit)
│   └── model_rf.py        # Model training + benchmarking + ROC-AUC evaluation
├── app/
│   └── streamlit_app.py   # Interactive toxicity prediction web app
├── data/                  # Tox21 dataset (downloaded via usage instructions)
├── models/                # Saved trained models (.pkl)
├── notebooks/             # Exploratory analysis notebooks
├── requirements.txt       # Python dependencies
└── README.md
```

---

##  Future Work

- Extend to **PFAS and endocrine disruptor** datasets
- Add **Graph Neural Network (GNN)** using PyTorch Geometric
- Extend to **protein-ligand interaction** predictions
- Deploy web app on **Streamlit Cloud**

---

##  References

- Tox21 Dataset: https://tox21.gov
- RDKit Documentation: https://www.rdkit.org
- QSAR Modelling: Roy et al. (2015), *A Primer on QSAR/QSPR Modeling*

---

##  License

This project is licensed under the MIT License.

---

*Built with RDKit · scikit-learn · XGBoost · Streamlit · Tox21 (NIH)*