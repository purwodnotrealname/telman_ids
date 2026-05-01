import os
import argparse
import logging
import pickle

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEFAULT_TRAIN_PATH = 'data/raw/UNSW_NB15_training-set.csv'
DEFAULT_TEST_PATH = 'data/raw/UNSW_NB15_testing-set.csv'
OUTPUT_DIR = "data/processed"

# pre configuration
ID_COLS = ["id"]
TARGET_COL = "label"
CAT_COL = "attack_cat"

CATEGORICAL_FEATURES = ["proto", "service", "state"]

DEFAULT_N_FEATURES = 10
SMOTE_RANDOM_STATE = 42


#load 
def load_data(train_path: str, test_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("loading training %s", train_path)
    train = pd.read_csv(train_path)

    log.info("loading testing %s", test_path)
    test = pd.read_csv(test_path)

    log.info("training data shape: %s", train.shape)
    log.info("testing data shape: %s", test.shape)
    return train, test

#clean
def clean_data(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    log.info("cleaning data")
    
    for df in (train,test):
        for col in df.select_dtypes(include="object").columns:
            df[col] = df[col].str.strip()
    
    before = len(train)
    train = train.drop_duplicates()
    removed = before - len(train)
    if removed: 
        log.info("removed %d duplicate rows", removed)
    else:
        log.info("no duplicate rows")
    
    feature_cols = [c for c in train.columns if c not in ID_COLS + [TARGET_COL, CAT_COL] + CATEGORICAL_FEATURES]
    numeric_train = train[feature_cols].select_dtypes(include=[np.number])
    constant_cols = numeric_train.columns[numeric_train.nunique() <= 1].tolist()
    if constant_cols:
        log.info("removing %d constant columns: %s", len(constant_cols), constant_cols)
        train = train.drop(columns=constant_cols)
        test = test.drop(columns=constant_cols)
    else:
        log.info("no constant columns found")
    return train, test

#encode categorical features
def encode_categorical(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    log.info("encoding categorical features %s", CATEGORICAL_FEATURES)
    encoders: dict[str, LabelEncoder] = {}

    for col in CATEGORICAL_FEATURES:
        if col not in train.columns:
            log.warning("column %s not found in training data, skipping encoding", col)
            continue

        le = LabelEncoder()
        le.fit(train[col].astype(str))
        encoders[col] = le

        n_classes = len(le.classes_)
        train[col] = le.transform(train[col].astype(str))

        def safe_transform(series, encoder=le, n=n_classes):
            known = set(encoder.classes_)
            return series.astype(str).map(lambda v: encoder.transform([v])[0] if v in known else n)
        
        test[col] = safe_transform(test[col])
        log.info(" %-12s -> %d classes", col, n_classes)

    return train, test, encoders

#split features 
def split_and_normalize(train: pd.DataFrame, test: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], MinMaxScaler]:
    log.info("normalizing")

    drop_for_x = ID_COLS + [TARGET_COL, CAT_COL]
    feature_names = [c for c in train.columns if c not in drop_for_x]    

    x_train = train[feature_names].values.astype(np.float32)
    x_test = test[feature_names].values.astype(np.float32)
    y_train = train[TARGET_COL].values.astype(np.int32)
    y_test = test[TARGET_COL].values.astype(np.int32)

    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(x_train).astype(np.float32)
    X_test_scaled = scaler.transform(x_test).astype(np.float32)

    log.info("  X_train: %s   X_test: %s", X_train_scaled.shape, X_test_scaled.shape)
    log.info("  y_train  — Normal: %d  Attack: %d",(y_train == 0).sum(), (y_train == 1).sum())
    log.info("  y_test   — Normal: %d  Attack: %d",
             (y_test  == 0).sum(), (y_test  == 1).sum())
 
    return X_train_scaled, X_test_scaled, y_train, y_test, feature_names, scaler

    
#feature selection
def select_features(X_train: np.array, X_test: np.array, y_train: np.array, feature_names: list[str], n_features: int = DEFAULT_N_FEATURES, random_state: int = 42) -> tuple[np.ndarray, np.ndarray, list[str], pd.Series]:

    log.info("selecting top %d features", n_features)

    mi_raw = mutual_info_classif(X_train, y_train, random_state=random_state, n_jobs=-1)
    mi_scores = pd.Series(mi_raw, index=feature_names).sort_values(ascending=False)

    log.info("top features %d ", n_features)
    for rank, (feat, score) in enumerate(mi_scores.head(n_features).items(), 1):
        log.info("  %2d. %-20s  MI: %.4f", rank, feat, score)
    
    selected_names = mi_scores.head(n_features).index.tolist()
    col_indices = [feature_names.index(f) for f in selected_names]

    X_train_sel = X_train[:, col_indices]
    X_test_sel = X_test[:, col_indices]

    return X_train_sel, X_test_sel, selected_names, mi_scores

# class weighing

def compute_class_weights(y_train: np.ndarray) -> dict:
    classes = np.unique(y_train)
    weights = compute_class_weight(
        class_weight="balanced",
        classes=classes,
        y=y_train,
    )
    class_weights = dict(zip(classes.tolist(), weights.tolist()))
    for cls, w in class_weights.items():
        label = "Normal" if cls == 0 else "Attack"
        log.info("  Class %d (%s): weight = %.4f", cls, label, w)
    log.info("  Usage → model.fit(..., class_weight=%s)", class_weights)
    return class_weights
 

# smote for class imbalance
def apply_smote(X_train: np.ndarray, y_train: np.ndarray, random_state: int = SMOTE_RANDOM_STATE,) -> tuple[np.ndarray, np.ndarray]:
    try:
        from imblearn.over_sampling import SMOTE
    except ImportError:
        log.error("imblearn is required for SMOTE. Please install it with 'pip install imblearn'")
        raise

    log.info("applying SMOTE")
    before = dict(zip(*np.unique(y_train, return_counts=True)))
    log.info(
        "  Before — Normal: %d  Attack: %d  (ratio 1:%.2f)",
        before[0], before[1], before[1] / before[0],
    )

    smote = SMOTE(random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    after = dict(zip(*np.unique(y_res, return_counts=True)))
    log.info(
        "  After  — Normal: %d  Attack: %d  (balanced)",
        after[0], after[1],
    )
    log.info(
        "  Synthetic samples added: %d",
        after[0] - before[0],
    )
    return X_res.astype(np.float32), y_res.astype(np.int32)

def reshape_for_cnn(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Conv1D in Keras/TF expects shape  (batch, steps, channels).
    Each feature is treated as one time-step with a single channel:
        (n_samples, n_features)  →  (n_samples, n_features, 1)
    """
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn  = X_test.reshape(X_test.shape[0],  X_test.shape[1],  1)
    log.info("=== CNN reshape ===")
    log.info("  X_train: %s   X_test: %s", X_train_cnn.shape, X_test_cnn.shape)
    return X_train_cnn, X_test_cnn
 

def save_artefacts(
    output_dir: str,
    X_train: np.ndarray,
    X_test: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    selected_features: list[str],
    mi_scores: pd.Series,
    scaler: MinMaxScaler,
    encoders: dict,
    class_weights: dict,
) -> None:
    
    os.makedirs(output_dir, exist_ok=True)
 
    np.save(os.path.join(output_dir, "X_train.npy"), X_train)
    np.save(os.path.join(output_dir, "X_test.npy"),  X_test)
    np.save(os.path.join(output_dir, "y_train.npy"), y_train)
    np.save(os.path.join(output_dir, "y_test.npy"),  y_test)
 
    with open(os.path.join(output_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
 
    with open(os.path.join(output_dir, "encoders.pkl"), "wb") as f:
        pickle.dump(encoders, f)
 
    with open(os.path.join(output_dir, "class_weights.pkl"), "wb") as f:
        pickle.dump(class_weights, f)
 
    with open(os.path.join(output_dir, "selected_features.txt"), "w") as f:
        f.write("\n".join(selected_features))
 
    mi_scores.to_csv(os.path.join(output_dir, "mi_scores.csv"), header=["mi_score"])
 
    log.info("=== Artefacts saved to '%s' ===", output_dir)
    log.info("  Arrays : X_train%s  X_test%s", X_train.shape, X_test.shape)
    log.info("  Files  : scaler.pkl, encoders.pkl, class_weights.pkl, selected_features.txt, mi_scores.csv")

#run
def run_preprocessing(
    train_path: str  = DEFAULT_TRAIN_PATH,
    test_path:  str  = DEFAULT_TEST_PATH,
    output_dir: str  = OUTPUT_DIR,
    n_features: int  = DEFAULT_N_FEATURES,
    use_smote:  bool = False,
) -> dict:
    """
    Returns
    -------
    {
        "X_train":        np.ndarray,   shape (n, n_features, 1)
        "X_test":         np.ndarray,   shape (m, n_features, 1)
        "y_train":        np.ndarray,
        "y_test":         np.ndarray,
        "class_weights":  dict,         {0: float, 1: float}
        "selected_features": list[str],
        "mi_scores":      pd.Series,
        "scaler":         MinMaxScaler,
        "encoders":       dict,
        "smote_applied":  bool,
    }
    """

    log.info(" IDS TELMAN — Preprocessing")

 
    train, test                              = load_data(train_path, test_path)
    train, test                              = clean_data(train, test)
    train, test, encoders                    = encode_categorical(train, test)
    X_tr, X_te, y_tr, y_te, feat_names, scaler = split_and_normalize(train, test)
    X_tr_sel, X_te_sel, sel_feats, mi_scores = select_features(
                                                    X_tr, X_te, y_tr,
                                                    feat_names, n_features)
 
    # ── Strategy A: class weights (always computed) ──
    class_weights = compute_class_weights(y_tr)
 
    # ── Strategy B: SMOTE (opt-in, training set only) ──
    smote_applied = False
    if use_smote:
        X_tr_sel, y_tr = apply_smote(X_tr_sel, y_tr)
        smote_applied = True
 
    X_tr_cnn, X_te_cnn                       = reshape_for_cnn(X_tr_sel, X_te_sel)
 
    save_artefacts(output_dir, X_tr_cnn, X_te_cnn, y_tr, y_te,
                   sel_feats, mi_scores, scaler, encoders, class_weights)
 
    log.info(" Imbalance handling:")
    log.info("   Class weights : YES (always)")
    log.info("   SMOTE         : %s", "YES" if smote_applied else "NO (pass --use_smote to enable)")
 
    return {
        "X_train":           X_tr_cnn,
        "X_test":            X_te_cnn,
        "y_train":           y_tr,
        "y_test":            y_te,
        "class_weights":     class_weights,
        "selected_features": sel_feats,
        "mi_scores":         mi_scores,
        "scaler":            scaler,
        "encoders":          encoders,
        "smote_applied":     smote_applied,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IDS TELMAN — Preprocessing")
    parser.add_argument("--train",      default=DEFAULT_TRAIN_PATH)
    parser.add_argument("--test",       default=DEFAULT_TEST_PATH)
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    parser.add_argument("--n_features", type=int, default=DEFAULT_N_FEATURES,
                        help="Number of top MI features to keep (default: 10)")
    parser.add_argument("--use_smote",  action="store_true",
                        help="Apply SMOTE oversampling to training set "
                             "(requires: pip install imbalanced-learn)")
    args = parser.parse_args()
 
    run_preprocessing(
        train_path = args.train,
        test_path  = args.test,
        output_dir = args.output_dir,
        n_features = args.n_features,
        use_smote  = args.use_smote,
    )