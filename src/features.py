from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors
import numpy as np
import pandas as pd

def smiles_to_fingerprint(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    return np.array(fp)

def smiles_to_descriptors(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return {
        "MolecularWeight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBondDonors": Descriptors.NumHDonors(mol),
        "HBondAcceptors": Descriptors.NumHAcceptors(mol),
        "TPSA": Descriptors.TPSA(mol),
        "RotatableBonds": Descriptors.NumRotatableBonds(mol),
        "AromaticRings": Descriptors.NumAromaticRings(mol),
    }

def load_and_featurize(csv_path, smiles_col="smiles", target_cols=None):
    df = pd.read_csv(csv_path)
    if target_cols is None:
        target_cols = [
            'NR-AR','NR-AR-LBD','NR-AhR','NR-Aromatase',
            'NR-ER','NR-ER-LBD','NR-PPAR-gamma',
            'SR-ARE','SR-ATAD5','SR-HSE','SR-MMP','SR-p53'
        ]
    fingerprints, valid_idx = [], []
    for i, smiles in enumerate(df[smiles_col]):
        fp = smiles_to_fingerprint(str(smiles))
        if fp is not None:
            fingerprints.append(fp)
            valid_idx.append(i)
    X = np.array(fingerprints)
    y = df[target_cols].iloc[valid_idx].values
    print(f"✅ Loaded {len(valid_idx)} valid molecules")
    print(f"✅ Feature matrix: {X.shape}")
    return X, y, valid_idx, df.iloc[valid_idx]