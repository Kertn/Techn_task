import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.neighbors import KernelDensity
from sklearn.metrics import roc_auc_score, matthews_corrcoef, roc_curve
import xgboost as xgb
import joblib
import os

SEED = 42
np.random.seed(SEED)


def engineer_features(X_df):

    X = X_df.copy()
    a = X.iloc[:, 0].values
    b = X.iloc[:, 1].values

    s = a + b
    d = a - b
    prod = a * b
    ratio = a / (b + 1e-8)
    ratio2 = b / (a + 1e-8)
    a2 = a ** 2
    b2 = b ** 2
    a3 = a ** 3
    b3 = b ** 3
    sa = np.sign(a) * np.log1p(np.abs(a))
    sb = np.sign(b) * np.log1p(np.abs(b))

    four_a = np.column_stack([np.sin(a), np.cos(a), np.sin(2*a), np.cos(2*a)])
    four_b = np.column_stack([np.sin(b), np.cos(b), np.sin(2*b), np.cos(2*b)])

    kde = KernelDensity(kernel='gaussian', bandwidth=0.1)
    kde.fit(a.reshape(-1, 1))
    dens_a = np.exp(kde.score_samples(a.reshape(-1, 1)))
    kde.fit(b.reshape(-1, 1))
    dens_b = np.exp(kde.score_samples(b.reshape(-1, 1)))

    kb = KBinsDiscretizer(n_bins=8, encode='onehot-dense', strategy='quantile')
    bins = kb.fit_transform(np.vstack([a, b]).T)

    pca = PCA(n_components=2)
    pca_xy = pca.fit_transform(np.vstack([a, b]).T)

    X_feats = np.column_stack([
        a, b, s, d, prod, ratio, ratio2,
        a2, b2, a3, b3, sa, sb, dens_a, dens_b,
        pca_xy
    ])
    X_feats = np.hstack([X_feats, four_a, four_b, bins])

    return X_feats


def best_threshold_mcc(y_true, y_prob):
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    mccs = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        mccs.append(matthews_corrcoef(y_true, y_pred))

    max_mcc = max(mccs)
    best_index = mccs.index(max_mcc)
    return float(thresholds[best_index]), float(max_mcc)


def train_and_evaluate(X, y, params):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof_proba = np.zeros(len(y), dtype=float)
    fold_aucs, fold_mccs, fold_thrs = [], [], []

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X, y), 1):
        X_tr, X_va = X[tr_idx], X[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)

        proba_va = model.predict_proba(X_va)[:, 1]
        oof_proba[va_idx] = proba_va

        auc = roc_auc_score(y_va, proba_va)
        thr_best, mcc_best = best_threshold_mcc(y_va, proba_va)

        fold_aucs.append(auc)
        fold_mccs.append(mcc_best)
        fold_thrs.append(thr_best)

        print(f"Fold {fold}: ROC-AUC={auc:.6f}, MCC={mcc_best:.6f}, Thr={thr_best:.6f}")

    oof_auc = roc_auc_score(y, oof_proba)
    oof_thr_best, oof_mcc_best = best_threshold_mcc(y, oof_proba)

    print("\nCV Summary:")
    print(f"Mean ROC-AUC: {np.mean(fold_aucs):.6f}, OOF ROC-AUC: {oof_auc:.6f}")
    print(f"Mean MCC: {np.mean(fold_mccs):.6f}, OOF MCC: {oof_mcc_best:.6f}")
    print(f"OOF Optimal Threshold: {oof_thr_best:.6f}")

    final_model = xgb.XGBClassifier(**params)
    final_model.fit(X, y)

    return {
        "model": final_model,
        "oof_proba": oof_proba,
        "oof_auc": oof_auc,
        "oof_mcc": oof_mcc_best,
        "oof_threshold": oof_thr_best
    }


def main():

    TRAIN_PATH = "data/train_data.csv"
    LABELS_PATH = "data/train_labels.csv"
    TEST_PATH = "data/test_data.csv"

    df_train = pd.read_csv(TRAIN_PATH, header=None)
    df_labels = pd.read_csv(LABELS_PATH, header=None)


    y = df_labels[0].astype(int)

    if set(np.unique(y)) <= {-1, 1}:
        y = y.copy()
        y = y.replace(-1, 0) if isinstance(y, pd.Series) else np.where(y == -1, 0, 1)
    elif set(np.unique(y)) <= {0, 1}:
        pass

    X = df_train[[5432, 1017]]

    df_test = pd.read_csv(TEST_PATH, header=None)

    X_train = engineer_features(X)
    X_test = engineer_features(df_test)

    xgb_params = {
        'learning_rate': 0.0012660,
        'max_depth': 3,
        'n_estimators': 883,
        'subsample': 0.64898,
        'colsample_bytree': 0.62617,
        'gamma': 4.75288,
        'random_state': SEED,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'verbosity': 0,
        'n_jobs': -1,
    }

    results = train_and_evaluate(X_train, y, xgb_params)
    threshold = results['oof_threshold']

    predictions_proba = results["model"].predict_proba(X_test)

    prob_onnx = predictions_proba[:, 1] if predictions_proba.ndim == 2 else predictions_proba.ravel()

    y_pred = (prob_onnx > threshold).astype(int)

    predictions_df = pd.DataFrame({'test_predictions': y_pred})

    predictions_df.to_csv('outputs/test_labels.csv')

    os.makedirs("models", exist_ok=True)
    joblib.dump(results["model"], "models/xgb_final_model.joblib")
    print("Model saved to models/xgb_final_model.joblib")


if __name__ == "__main__":
    main()
