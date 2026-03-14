import streamlit as st
import pandas as pd
import numpy as np
import joblib
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.features import smiles_to_fingerprint, smiles_to_descriptors

st.set_page_config(page_title="QSAR Toxicity Predictor", page_icon="🧬", layout="wide")

st.title("🧬 QSAR Toxicity Predictor")
st.markdown("""
Predict the toxicity of any molecule using the **Tox21 dataset** and Machine Learning.
Enter a SMILES string below to predict toxicity across **12 biological targets**.

> *Built by Adejoke Adewumi — Environmental Science, Computational Toxicology, and AI-driven Chemical Safety*
""")

@st.cache_resource
def load_model():
    try:
        data = joblib.load("models/rf_model.pkl")
        return data["models"], data["target_names"]
    except FileNotFoundError:
        return None, None

models, target_names = load_model()

st.sidebar.header("🔬 Example Molecules")
examples = {
    "Aspirin": "CC(=O)Oc1ccccc1C(=O)O",
    "Caffeine": "Cn1cnc2c1c(=O)n(c(=O)n2C)C",
    "Benzene": "c1ccccc1",
    "Ibuprofen": "CC(C)Cc1ccc(cc1)C(C)C(=O)O",
    "3-BHA": "COc1ccc(O)cc1C(C)(C)C",
}

selected = st.sidebar.selectbox("Load an example:", ["-- Choose --"] + list(examples.keys()))
default_smiles = examples.get(selected, "") if selected != "-- Choose --" else ""
smiles_input = st.text_input("Enter SMILES string:", value=default_smiles, placeholder="e.g. CCO for ethanol")

if smiles_input:
    mol = Chem.MolFromSmiles(smiles_input)
    if mol is None:
        st.error("❌ Invalid SMILES string. Please check the format.")
    else:
        col1, col2 = st.columns([1, 2])
        with col1:
            st.subheader("🔵 Molecule Structure")
            img = Draw.MolToImage(mol, size=(300, 300))
            st.image(img)
            st.subheader("📐 Properties")
            descriptors = smiles_to_descriptors(smiles_input)
            if descriptors:
                for k, v in descriptors.items():
                    st.metric(k, f"{v:.2f}")

        with col2:
            st.subheader("⚠️ Toxicity Predictions")
            if models is None:
                st.warning("⚠️ Model not trained yet. Run the training script first.")
                predictions = {t: np.random.uniform(0, 1) for t in [
                    'NR-AR','NR-AhR','NR-ER','SR-ARE','SR-HSE','SR-p53'
                ]}
            else:
                fp = smiles_to_fingerprint(smiles_input)
                predictions = {}
                for model, target in zip(models, target_names):
                    if model is not None:
                        prob = model.predict_proba(fp.reshape(1, -1))[0][1]
                        predictions[target] = prob

            pred_df = pd.DataFrame({
                "Target": list(predictions.keys()),
                "Toxicity Probability": list(predictions.values())
            }).sort_values("Toxicity Probability", ascending=True)

            fig, ax = plt.subplots(figsize=(8, 5))
            colors = ["#e74c3c" if v > 0.5 else "#2ecc71" for v in pred_df["Toxicity Probability"]]
            ax.barh(pred_df["Target"], pred_df["Toxicity Probability"], color=colors)
            ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
            ax.set_xlabel("Probability of Toxicity")
            ax.set_xlim(0, 1)
            st.pyplot(fig)

            high_risk = [t for t, v in predictions.items() if v > 0.5]
            if high_risk:
                st.error(f"🔴 High toxicity: {', '.join(high_risk)}")
            else:
                st.success("🟢 Low toxicity predicted across all targets")

st.markdown("---")
st.markdown("*Data: Tox21 (NIH) | Model: Random Forest | Built with RDKit & Streamlit*")