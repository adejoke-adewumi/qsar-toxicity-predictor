import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import joblib
import os

def train_random_forest(X, y, target_names=None, save_path="models/rf_model.pkl"):
    n_targets = y.shape[1]
    if target_names is None:
        target_names = [f"Target_{i}" for i in range(n_targets)]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    results = {}
    models = []

    print("Training Random Forest models...\n")

    for i, target in enumerate(target_names):
        y_col_train = y_train[:, i]
        y_col_test = y_test[:, i]
        train_mask = ~np.isnan(y_col_train.astype(float))
        test_mask = ~np.isnan(y_col_test.astype(float))

        if train_mask.sum() < 50:
            print(f"⚠️  {target}: Not enough data, skipping")
            models.append(None)
            continue

        model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight='balanced',
            random_state=42,
            n_jobs=-1
        )
        model.fit(X_train[train_mask], y_col_train[train_mask])
        y_pred = model.predict_proba(X_test[test_mask])[:, 1]
        auc = roc_auc_score(y_col_test[test_mask], y_pred)
        results[target] = auc
        models.append(model)
        print(f"✅ {target}: ROC-AUC = {auc:.3f}")

    mean_auc = np.mean(list(results.values()))
    print(f"\n📊 Mean ROC-AUC: {mean_auc:.3f}")

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    joblib.dump({"models": models, "target_names": target_names}, save_path)
    print(f"💾 Model saved to {save_path}")

    return results, models
if __name__ == "__main__":
    import sys
    import os

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if project_root not in sys.path:
        sys.path.append(project_root)

    from src.features import load_and_featurize

    TARGET_COLS = [
        "NR-AR", "NR-AR-LBD", "NR-AhR", "NR-Aromatase",
        "NR-ER", "NR-ER-LBD", "NR-PPAR-gamma",
        "SR-ARE", "SR-ATAD5", "SR-HSE", "SR-MMP", "SR-p53"
    ]

    print("Loading dataset...")
    X, y, _, _ = load_and_featurize("data/tox21.csv", target_cols=TARGET_COLS)

    print("Starting training...\n")
    results, models = train_random_forest(X, y, target_names=TARGET_COLS)