"""
Spaceship Titanic — improved pipeline
=====================================
Changes vs v1 (CV ~0.8117):
  1. Log-transformed spending + per-amenity spend flags (spending is heavily skewed)
  2. More features: cabin deck groups, cabin parity, family size, group leader flag,
     age groups, missing-value indicators, CryoSleep x spending signals
  3. Tuned CatBoost (lower lr, more iterations, depth 8, regularization)
  4. 5-fold CV with out-of-fold (OOF) predictions averaged over 3 seeds
  5. OOF-based probability threshold tuning (tiny gain on a balanced target)
"""
import os
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

# CONFIG
RANDOM_SEED = 42
N_FOLDS = 5
SEEDS = (42, 7, 2024)          # seed-averaging for stability
TRAIN_PATH = "./dataset/train.csv"
TEST_PATH = "./dataset/test.csv"

OUTPUT_PATH = "./outputs"

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
    # decks cluster by ship section
    deck_map = {"A": "lower", "B": "lower", "C": "lower", "T": "vip",
                "D": "mid", "E": "mid", "F": "mid", "G": "mid"}
    df["CabinDeckGroup"] = df["CabinDeck"].map(deck_map).fillna("Missing")
    df["CabinSideCode"] = df["CabinSide"].map({"P": 0, "S": 1})
    df["CabinParity"] = df["CabinNumber"] % 2      # NaN propagates -> filled later
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

    # skewed distributions -> log1p makes them usable
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
    df["Age"] = df["Age"].fillna(27.0)          # train median (see v1 stats)
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


def prepare_model_inputs(train, test):
    X = train.drop(columns=["Transported"])
    y = train["Transported"].astype(int)
    X_test = test.copy()

    X = X.drop(columns=["Name", "PassengerId"])
    X_test = X_test.drop(columns=["Name", "PassengerId"])

    # 'str' dtype included explicitly (pandas 2/3 compatible, kills the warning)
    categorical_features = X.select_dtypes(include=["object", "str", "bool"]).columns.tolist()
    for col in categorical_features:
        X[col] = X[col].fillna("Missing").astype(str)
        X_test[col] = X_test[col].fillna("Missing").astype(str)

    numerical_features = X.select_dtypes(include=["number"]).columns.tolist()
    for col in numerical_features:
        median_val = X[col].median()
        X[col] = X[col].fillna(median_val)
        X_test[col] = X_test[col].fillna(median_val)

    return X, y, X_test, categorical_features


def make_model(seed):
    return CatBoostClassifier(
        iterations=4000,
        learning_rate=0.03,
        depth=8,
        l2_leaf_reg=3,
        min_data_in_leaf=20,
        random_strength=1.0,
        bagging_temperature=1.0,
        loss_function="Logloss",
        eval_metric="Accuracy",
        verbose=False,
        random_seed=seed,
        allow_writing_files=False,
    )


def cross_validate(X, y, cat_features, n_folds=N_FOLDS, seeds=SEEDS):
    """5-fold CV with seed-averaged OOF predictions + threshold tuning."""
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_SEED)
    oof = np.zeros(len(X))

    fold_bar = tqdm(
        enumerate(skf.split(X, y), start=1),
        total=n_folds,
        desc="CV folds",
        unit="fold",
    )

    for fold, (tr_idx, val_idx) in fold_bar:
        X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

        fold_pred = np.zeros(len(X_val))
        seed_bar = tqdm(
            seeds,
            desc=f"  Fold {fold} seeds",
            unit="seed",
            leave=False,
        )
        for seed in seed_bar:
            model = make_model(seed)
            model.fit(X_tr, y_tr, cat_features=cat_features,
                      eval_set=(X_val, y_val), early_stopping_rounds=200,
                      use_best_model=True)
            fold_pred += model.predict_proba(X_val)[:, 1] / len(seeds)

        oof[val_idx] = fold_pred
        fold_acc = accuracy_score(y_val, (fold_pred > 0.5).astype(int))
        fold_bar.set_postfix(acc=f"{fold_acc:.4f}")
        tqdm.write(f"Fold {fold}: accuracy = {fold_acc:.4f}")

    # threshold search on OOF (target is balanced, so expect ~0.5)
    grid = np.linspace(0.40, 0.60, 81)
    best_thr, best_acc = 0.5, 0.0
    for thr in tqdm(grid, desc="Threshold tuning", unit="thr", leave=False):
        acc = accuracy_score(y, (oof > thr).astype(int))
        if acc > best_acc:
            best_acc, best_thr = acc, thr

    print(f"\nOOF accuracy: {accuracy_score(y, (oof > 0.5).astype(int)):.4f}")
    print(f"OOF accuracy with tuned threshold {best_thr:.3f}: {best_acc:.4f}")
    return oof, best_thr


def train_final_and_predict(X, y, X_test, cat_features, threshold, seeds=SEEDS):
    test_proba = np.zeros(len(X_test))
    seed_bar = tqdm(seeds, desc="Final training", unit="seed")
    for seed in seed_bar:
        model = make_model(seed)
        model.fit(X, y, cat_features=cat_features, verbose=False)
        test_proba += model.predict_proba(X_test)[:, 1] / len(seeds)
    return (test_proba > threshold).astype(int), test_proba


def build_submission(test, pred, out_path="./outputs/submission.csv"):
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

    _, best_thr = cross_validate(X, y, cat_features)

    pred, _ = train_final_and_predict(X, y, X_test, cat_features, best_thr)
    
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    build_submission(test_fe, pred, os.path.join(OUTPUT_PATH, "submission_xgb.csv"))


if __name__ == "__main__":
    main()