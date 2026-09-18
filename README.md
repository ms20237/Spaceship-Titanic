# Spaceship-Titanic

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange)
![Pandas](https://img.shields.io/badge/Pandas-2.0%2B-green)
![CatBoost](https://img.shields.io/badge/CatBoost-1.2%2B-yellow)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-red)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## 📖 Overview

This repository tackles the [Kaggle Spaceship Titanic](https://www.kaggle.com/competitions/spaceship-titanic/overview) competition — a binary classification problem where the goal is to predict whether each passenger was **`Transported`** to another dimension, based on their travel and spending records.

The project follows a full ML workflow: a deep exploratory data analysis (EDA) notebook, feature engineering shared across models, and two gradient-boosting pipelines (**CatBoost** and **XGBoost**) trained with stratified cross-validation, seed-averaging, and probability-threshold tuning.

## 📂 Repository Structure

```
Spaceship-Titanic/
├── dataset/
│   ├── train.csv
│   ├── test.csv
│   └── sample_submission.csv
├── plots/                       # EDA figures (histograms, KDEs, signal charts, correlation matrix)
├── outputs/
│   ├── submission.csv           # CatBoost submission
│   └── submission_xgb.csv       # XGBoost submission
├── 01_EDA.ipynb
├── train_classifier.py          # CatBoost pipeline
├── train_xgboost.py             # XGBoost pipeline
└── README.md
```

## 📊 Dataset

*   **`train.csv`** — 8,693 passengers with features + the `Transported` target.
*   **`test.csv`** — 4,277 passengers, features only.
*   **`sample_submission.csv`** — expected submission format (`PassengerId`, `Transported`).

The dataset is not included in this repository. Download it from the [Kaggle competition page](https://www.kaggle.com/competitions/spaceship-titanic/data) and place the files in a `./dataset/` directory.

### Attribute Information

| Column | Type | Meaning |
| :--- | :--- | :--- |
| `PassengerId` | ID | `gggg_pp` — group and passenger number |
| `HomePlanet` | Categorical | Planet of origin |
| `CryoSleep` | Boolean | Whether the passenger was in suspended animation |
| `Cabin` | Text | `deck/number/side` |
| `Destination` | Categorical | Destination planet |
| `Age` | Numerical | Passenger age |
| `VIP` | Boolean | VIP service |
| `RoomService`, `FoodCourt`, `ShoppingMall`, `Spa`, `VRDeck` | Numerical | Amenity spending |
| `Name` | Text | Passenger name |
| `Transported` | Target | `True` / `False` |

## 🔍 Key Findings from the EDA (`01_EDA.ipynb`)

### 1. Shape & quality
*   Train: **8,693 × 14**, Test: **4,277 × 13**.
*   No duplicate rows or duplicate `PassengerId`s in either set.
*   Target is well balanced: **49.6% not transported / 50.4% transported**.

### 2. Missing values
Every feature has missing values, roughly 2–2.5% of rows each (`CryoSleep` 217, `ShoppingMall` 208, `VIP` 203, `HomePlanet` 201, `Name` 200, `Cabin` 199, down to `Age` 179 in train). None of the columns are missing at a rate that requires dropping — indicator flags + imputation are used instead.

### 3. What actually predicts `Transported`
Ranked by mutual information with the target:

| Rank | Feature | Signal |
| :--- | :--- | :--- |
| 1 | `CryoSleep` | Strongest single feature — passengers in cryosleep were transported **82%** of the time vs. **33%** awake (Cramér's V = 0.47) |
| 2 | `Spa` spending | Higher spend ↔ lower transport odds |
| 3 | `RoomService` spending | Same inverse pattern |
| 4 | `ShoppingMall` spending | Weaker, non-monotonic |
| 5 | `VRDeck` spending | Inverse pattern |
| 6 | `FoodCourt` spending | Weakest of the spending features |
| — | `Deck` (from `Cabin`) | Decks B/C transported most (~70–73%), E least (~36%) |
| — | `HomePlanet` | Europa 66% → Mars 52% → Earth 42% |
| — | `Age` | Children (<5) transported ~77% of the time vs. ~45–50% for adults |
| — | `VIP` | Very weak signal (Cramér's V = 0.04) |

Spending columns are **heavily right-skewed** (skew 5–11, thousands of IQR outliers) — `log1p` transforms are used before feeding them to the models.

### 4. Interactions
`CryoSleep × HomePlanet/Deck/Side` interactions carry extra signal beyond either feature alone (e.g. Europa passengers in cryosleep are transported ~99% of the time). `HomePlanet`, `CryoSleep`, `Destination`, and `VIP` categories are fully consistent between train and test — no unseen-category risk.

### 5. Train vs. test distribution
A KS-test pass is included in the notebook to check for covariate shift between train and test spending distributions; no material shift was found.

## 🛠️ Feature Engineering (shared by both models)

Implemented identically in `train_classifier.py` and `train_xgboost.py`:

*   **Cabin** → `CabinDeck`, `CabinNumber`, `CabinSide`, `CabinDeckGroup` (lower/mid/vip cluster), `CabinSideCode`, `CabinParity`, `CabinMissing` flag.
*   **PassengerId** → `GroupId`, `GroupSize`, `PassengerNumber`, `IsGroupLeader`, `IsSolo`.
*   **Spending** → `TotalSpending`, `NoSpending`, log1p of every spend column, per-amenity `SpendFlag_*`, `LuxurySpending` (Spa+VRDeck+ShoppingMall), `EssentialSpending` (RoomService+FoodCourt), and their log versions.
*   **Name** → `Surname`, `FamilySize` (passengers sharing a surname).
*   **Age** → `AgeMissing` flag, median imputation, `AgeGroup` bins (Child/Teen/YoungAdult/Adult/MiddleAged/Senior), `IsChild`.
*   **CryoSleep** → `CryoMissing` flag.

Missing categoricals are filled with `"Missing"`; missing numerics are imputed with the train median.

## 🤖 Models & Results

Both pipelines use 5-fold `StratifiedKFold` cross-validation, average predictions over 3 seeds (42, 7, 2024) for stability, and tune the classification threshold on out-of-fold (OOF) predictions.

### CatBoost (`train_classifier.py`)
`iterations=4000, learning_rate=0.03, depth=8, l2_leaf_reg=3` with native categorical support.

| Fold | Accuracy |
| :--- | :--- |
| 1 | — |
| 5 | 0.8113 |
| **OOF (thr=0.5)** | **0.8133** |
| **OOF (tuned thr=0.502)** | **0.8142** |

Output: `outputs/submission.csv`

### XGBoost (`train_xgboost.py`)
`n_estimators=4000, learning_rate=0.03, max_depth=7`, `enable_categorical=True` with one-hot for low-cardinality categoricals (`max_cat_to_onehot=12`).

```
Train shape: (8693, 14), Test shape: (4277, 13)
Features: 43 (categorical: 8)
Fold 1: accuracy = 0.8154
Fold 2: accuracy = 0.8074
Fold 3: accuracy = 0.8125
Fold 4: accuracy = 0.8136
Fold 5: accuracy = 0.7969

OOF accuracy: 0.8092
OOF accuracy with tuned threshold 0.495: 0.8101
```

Output: `outputs/submission_xgb.csv`

### Comparison

| Model | OOF accuracy | Tuned OOF accuracy |
| :--- | :--- | :--- |
| CatBoost | 0.8133 | **0.8142** |
| XGBoost | 0.8092 | 0.8101 |

CatBoost currently edges out XGBoost, mainly from its native handling of high-cardinality categoricals (`CabinNumber`, `Surname`-derived features) without manual encoding.

## 🛠️ Technologies Used

*   **Python 3.9+**
*   **Jupyter Notebook** for EDA
*   **Pandas / NumPy** for data manipulation
*   **Matplotlib / Seaborn** for visualization
*   **scikit-learn** for cross-validation and metrics
*   **CatBoost** and **XGBoost** for modeling
*   **tqdm** for training progress bars

## 🚀 Getting Started

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ms20237/Spaceship-Titanic.git
    cd Spaceship-Titanic
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn catboost xgboost tqdm jupyter scipy statsmodels
    ```

3.  **Download the data** from Kaggle and place it under `./dataset/`:
    ```
    dataset/
    ├── train.csv
    ├── test.csv
    └── sample_submission.csv
    ```

4.  **Run the EDA notebook:**
    ```bash
    jupyter notebook 01_EDA.ipynb
    ```

5.  **Train a model and generate a submission:**
    ```bash
    python train_classifier.py     # CatBoost -> outputs/submission.csv
    python train_xgboost.py        # XGBoost  -> outputs/submission_xgb.csv
    ```

## 🔮 Next Steps

*   **Blend** CatBoost and XGBoost OOF probabilities (or stack with a meta-learner) to push past the current ~0.814 ceiling.
*   **Hyperparameter search** (Optuna) instead of the current hand-tuned settings.
*   **Group-aware imputation** — infer missing `HomePlanet`/`Destination`/`CryoSleep` from other members of the same `GroupId`, which the EDA shows are highly consistent within groups.
*   **Target encoding** for `Surname` / `CabinNumber` with proper out-of-fold computation to avoid leakage.
*   **LightGBM** as a third base model for the ensemble.

## License

This project is licensed under the [MIT License](https://choosealicense.com/licenses/mit/).