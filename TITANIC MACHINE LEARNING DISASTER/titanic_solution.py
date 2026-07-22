"""
Titanic - Machine Learning from Disaster
A complete, solid baseline solution (~0.80-0.83 accuracy range).

HOW TO USE:
1. Download train.csv and test.csv from the Kaggle "Data" tab:
   https://www.kaggle.com/c/titanic/data
2. Put them in the same folder as this script.
3. Run: python titanic_solution.py
4. It will produce submission.csv - upload that on the "Submit Prediction" page.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------
train = pd.read_csv("train.csv")
test = pd.read_csv("test.csv")

test_ids = test["PassengerId"]
full = pd.concat([train, test], sort=False)  # engineer features on both together

# ---------------------------------------------------------
# 2. FEATURE ENGINEERING
# ---------------------------------------------------------

# --- Title from Name ---
full["Title"] = full["Name"].str.extract(r",\s*([^\.]*)\.")
title_map = {
    "Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs",
    "Lady": "Rare", "Countess": "Rare", "Capt": "Rare", "Col": "Rare",
    "Don": "Rare", "Dr": "Rare", "Major": "Rare", "Rev": "Rare",
    "Sir": "Rare", "Jonkheer": "Rare", "Dona": "Rare",
}
full["Title"] = full["Title"].replace(title_map)
full.loc[~full["Title"].isin(["Mr", "Mrs", "Miss", "Master", "Rare"]), "Title"] = "Rare"

# --- Family features ---
full["FamilySize"] = full["SibSp"] + full["Parch"] + 1
full["IsAlone"] = (full["FamilySize"] == 1).astype(int)

# --- Deck from Cabin (missing cabin -> 'M' for missing) ---
full["Deck"] = full["Cabin"].str[0]
full["Deck"] = full["Deck"].fillna("M")
# Group rare decks together
deck_map = {"A": "ABC", "B": "ABC", "C": "ABC", "D": "DE", "E": "DE",
            "F": "FG", "G": "FG", "T": "ABC", "M": "M"}
full["Deck"] = full["Deck"].map(deck_map)

# --- Fill missing Age using median by Title + Pclass group ---
full["Age"] = full.groupby(["Title", "Pclass"])["Age"].transform(
    lambda x: x.fillna(x.median())
)
full["Age"] = full["Age"].fillna(full["Age"].median())

# --- Fill missing Fare (1 missing in test) using median by Pclass ---
full["Fare"] = full.groupby("Pclass")["Fare"].transform(
    lambda x: x.fillna(x.median())
)

# --- Fill missing Embarked with mode ---
full["Embarked"] = full["Embarked"].fillna(full["Embarked"].mode()[0])

# --- Age and Fare bins (helps tree models find splits) ---
full["AgeBin"] = pd.qcut(full["Age"], 5, labels=False, duplicates="drop")
full["FareBin"] = pd.qcut(full["Fare"], 5, labels=False, duplicates="drop")

# --- Ticket frequency (people traveling on same ticket often survived together) ---
ticket_counts = full["Ticket"].value_counts()
full["TicketFreq"] = full["Ticket"].map(ticket_counts)

# ---------------------------------------------------------
# 3. ENCODE CATEGORICALS
# ---------------------------------------------------------
cat_cols = ["Sex", "Embarked", "Title", "Deck"]
full = pd.get_dummies(full, columns=cat_cols, drop_first=True)

feature_cols = [
    "Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone",
    "AgeBin", "FareBin", "TicketFreq"
] + [c for c in full.columns if c.startswith(("Sex_", "Embarked_", "Title_", "Deck_"))]

# ---------------------------------------------------------
# 4. SPLIT BACK INTO TRAIN / TEST
# ---------------------------------------------------------
X = full.loc[full["Survived"].notna(), feature_cols]
y = full.loc[full["Survived"].notna(), "Survived"].astype(int)
X_test = full.loc[full["Survived"].isna(), feature_cols]

# ---------------------------------------------------------
# 5. MODEL: Random Forest (tuned) — reliably strong on this dataset
# ---------------------------------------------------------
rf = RandomForestClassifier(random_state=42)
param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [4, 5, 6, 7],
    "min_samples_split": [2, 4, 6],
    "min_samples_leaf": [1, 2, 3],
}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
grid = GridSearchCV(rf, param_grid, cv=cv, scoring="accuracy", n_jobs=-1)
grid.fit(X, y)

best_model = grid.best_estimator_
print("Best params:", grid.best_params_)

cv_scores = cross_val_score(best_model, X, y, cv=cv, scoring="accuracy")
print(f"Cross-val accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ---------------------------------------------------------
# 6. FEATURE IMPORTANCE (sanity check)
# ---------------------------------------------------------
importances = pd.Series(best_model.feature_importances_, index=feature_cols)
print("\nTop 10 features:")
print(importances.sort_values(ascending=False).head(10))

# ---------------------------------------------------------
# 7. PREDICT & SAVE SUBMISSION
# ---------------------------------------------------------
best_model.fit(X, y)  # fit on all training data
predictions = best_model.predict(X_test).astype(int)

submission = pd.DataFrame({"PassengerId": test_ids, "Survived": predictions})
submission.to_csv("submission.csv", index=False)
print("\nSaved submission.csv — ready to upload to Kaggle!")
