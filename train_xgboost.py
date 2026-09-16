"""
Spaceship Titanic — XGBoost pipeline
=====================================
Same feature engineering as the improved CatBoost version, but with XGBClassifier
(native categorical support via enable_categorical).

Usage:
    pip install xgboost pandas numpy scikit-learn
    python train_xgboost.py

Output:
    submission_xgb.csv
"""
import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score

# CONFIG
RANDOM_SEED = 42
N_FOLDS = 5
SEEDS = (42, 7, 2024)

TRAIN_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"
OUTPUT_PATH = "./outputs/"

SPENDING_COLS = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
LUXURY_COLS = ["Spa", "VRDeck", "ShoppingMall"]
ESSENTIAL_COLS = ["RoomService", "FoodCourt"]


# FEATURE ENGINEERING 
def add_cabin_features(df):
    df = df.copy()
    df["CabinMissing"] = df["Cabin"].isna().astype(int)
    parts = df["Cabin"].str.split("/", expand=True)
    if parts.shape[1] < 3:
        for i in range(parts.shape[1], 3):
            parts[i] = np.nan
    df["CabinDeck"] = parts[0]
    df["CabinNumber"] = pd.to_numeric(parts[1], errors="coerce")
    df["CabinSide"] = parts[2]
    deck_map = {"A": "lower", "B": "lower", "C": "lower", "T": "vip",
                "D": "mid", "E": "mid", "F": "mid", "G": "mid"}
    df["CabinDeckGroup"] = df["CabinDeck"].map(deck_map).fillna("Missing")
    df["CabinSideCode"] = df["CabinSide"].map({"P": 0, "S": 1})
    df["CabinParity"] = df["CabinNumber"] % 2
    df = df.drop(columns=["Cabin"])
    return df


def add_passenger_features(df):
    df = df.copy()
    df["GroupId"] = df["PassengerId"].str.split("_").str[0]
    df["GroupSize"] = df.groupby("GroupId")["PassengerId"].transform("count")
    df["PassengerNumber"] = pd.to_numeric(
        df["PassengerId"].str.split("_").str[1], errors="coerce")
    df["IsGroupLeader"] = (df["PassengerNumber"] == 1).astype(int)
    df["IsSolo"] = (df["GroupSize"] == 1).astype(int)
    return df


def add_spending_features(df):
    df = df.copy()
    for col in SPENDING_COLS:
        df[col] = df[col].fillna(0)
    df["TotalSpending"] = df[SPENDING_COLS].sum(axis=1)
    df["NoSpending"] = (df["TotalSpending"] == 0).astype(int)
    df["LogTotalSpending"] = np.log1p(df["TotalSpending"])
    for col in SPENDING_COLS:
        df[f"Log_{col}"] = np.log1p(df[col])
        df[f"SpendFlag_{col}"] = (df[col] > 0).astype(int)
    df["LuxurySpending"] = df[LUXURY_COLS].sum(axis=1)
    df["EssentialSpending"] = df[ESSENTIAL_COLS].sum(axis=1)
    df["LogLuxurySpending"] = np.log1p(df["LuxurySpending"])
    df["LogEssentialSpending"] = np.log1p(df["EssentialSpending"])
    return df


def add_family_features(df):
    df = df.copy()
    df["Surname"] = df["Name"].str.split().str[-1].fillna("Unknown")
    df["FamilySize"] = df.groupby("Surname")["PassengerId"].transform("count")
    return df


def add_age_features(df):
    df = df.copy()
    df["AgeMissing"] = df["Age"].isna().astype(int)
    df["Age"] = df["Age"].fillna(27.0)  # train median
    bins = [-1, 12, 18, 30, 45, 60, 200]
    labels = ["Child", "Teen", "YoungAdult", "Adult", "MiddleAged", "Senior"]
    df["AgeGroup"] = pd.cut(df["Age"], bins=bins, labels=labels).astype(str)
    df["IsChild"] = (df["Age"] < 13).astype(int)
    return df


def add_cryo_features(df):
    df = df.copy()
    df["CryoMissing"] = df["CryoSleep"].isna().astype(int)
    return df


def engineer_features(df):
    df = add_cabin_features(df)
    df = add_passenger_features(df)
    df = add_spending_features(df)
    df = add_family_features(df)
    df = add_age_features(df)
    df = add_cryo_features(df)
    return df


# MODEL INPUTS 
def prepare_model_inputs(train, test):
    X = train.drop(columns=["Transported"])
    y = train["Transported"].astype(int)
    X_test = test.copy()

    X = X.drop(columns=["Name", "PassengerId", "GroupId", "Surname"])
    X_test = X_test.drop(columns=["Name", "PassengerId", "GroupId", "Surname"])

    categorical_features = X.select_dtypes(include=["object", "str", "bool"]).columns.tolist()
    for col in categorical_features:
        X[col] = X[col].fillna("Missing").astype(str)
        X_test[col] = X_test[col].fillna("Missing").astype(str)
        # XGBoost native categorical: category dtype (codes must be non-negative)
        X[col] = X[col].astype("category")
        X_test[col] = X_test[col].astype("category")

    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    for col in numerical_features:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    return X, y, X_test, categorical_features


# MODEL 
def make_model(seed, early_stopping_rounds=None):
    return xgb.XGBClassifier(
        n_estimators=4000,
        learning_rate=0.03,
        max_depth=7,
        min_child_weight=3,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_lambda=3.0,        # L2
        reg_alpha=0.1,         # L1
        tree_method="hist",
        enable_categorical=True,
        max_cat_to_onehot=12,   # one-hot the low-card cats; avoids partition-split noise
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,  # XGBoost >= 2.x: goes in the constructor
        random_state=seed,
        n_jobs=-1,
    )


def fit_with_early_stopping(model, X_tr, y_tr, X_val, y_val):
    model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
    return model


def cross_validate(X, y, n_folds=N_FOLDS, seeds=SEEDS):
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))

    for fold, (tr_idx, val_idx) in enumerate(skf.split(X, y), start=1):
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        fold_pred = np.zeros(len(X_val))
        for seed in seeds:
            model = make_model(seed, early_stopping_rounds=200)
            fit_with_early_stopping(model, X_tr, y_tr, X_val, y_val)
            fold_pred += model.predict_proba(X_val)[:, 1] / len(seeds)

        oof[val_idx] = fold_pred
        fold_acc = accuracy_score(y_val, (fold_pred > 0.5).astype(int))
        print(f"Fold {fold}: accuracy = {fold_acc:.4f}")

    grid = np.linspace(0.40, 0.60, 81)
    best_thr, best_acc = 0.5, 0.0
    for thr in grid:
        acc = accuracy_score(y, (oof > thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    print(f"\nOOF accuracy: {accuracy_score(y, (oof > 0.5).astype(int)):.4f}")
    print(f"OOF accuracy with tuned threshold {best_thr:.3f}: {best_acc:.4f}")
    return oof, best_thr


def train_final_and_predict(X, y, X_test, threshold, seeds=SEEDS):
    """No eval set -> trains n_estimators fully; use the CV best_iteration instead."""
    test_proba = np.zeros(len(X_test))
    for seed in seeds:
        model = make_model(seed)
        model.set_params(n_estimators=1500, early_stopping_rounds=None)  # no eval set -> no early stopping
        model.fit(X, y, verbose=False)
        test_proba += model.predict_proba(X_test)[:, 1] / len(seeds)
    return (test_proba > threshold).astype(int), test_proba


def build_submission(test, pred, out_path="./outputs/submission_xgb.csv"):
    submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Transported": pred.astype(bool),
    })
    submission.to_csv(out_path, index=False)
    print(f"\nSaved submission to: {out_path}")
    print(submission.head())
    return submission


def main():
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    print(f"Train shape: {train_raw.shape}, Test shape: {test_raw.shape}")

    train_fe = engineer_features(train_raw)
    test_fe = engineer_features(test_raw)

    X, y, X_test, cat_features = prepare_model_inputs(train_fe, test_fe)
    print(f"Features: {X.shape[1]} (categorical: {len(cat_features)})")

    _, best_thr = cross_validate(X, y)

    pred, _ = train_final_and_predict(X, y, X_test, best_thr)

    os.makedirs(OUTPUT_PATH, exist_ok=True)
    build_submission(test_fe, pred, os.path.join(OUTPUT_PATH, "submission_xgb.csv"))


if __name__ == "__main__":
    main()