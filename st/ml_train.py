import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    VotingClassifier,
)
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel

import joblib
import json
from scipy.stats import randint, uniform

LEAGUES_PATH = "./leagues.json"
features1 = [
    "HomeTeam",
    "AwayTeam",
    "HTeamEloScore",
    "ATeamEloScore",
    "HTdaysSinceLastMatch",
    "ATdaysSinceLastMatch",
    "HTW_rate",
    "ATW_rate",
    "ATD_rate",
    "HTD_rate",
    "7_HTW_rate",
    "12_HTW_rate",
    "7_ATW_rate",
    "12_ATW_rate",
    "7_HTD_rate",
    "12_HTD_rate",
    "7_ATD_rate",
    "12_ATD_rate",
    "7_HTL_rate",
    "12_HTL_rate",
    "7_ATL_rate",
    "12_ATL_rate",
    "5_HTHW_rate",
    "5_ATAW_rate",
    "ODDS1",
    "ODDSX",
    "ODDS2",
]
features2 = [
    "HomeTeam",
    "AwayTeam",
    "HTeamEloScore",
    "ATeamEloScore",
    "HTdaysSinceLastMatch",
    "ATdaysSinceLastMatch",
    "HTW_rate",
    "ATW_rate",
    "7_HTW_rate",
    "12_HTW_rate",
    "7_ATW_rate",
    "12_ATW_rate",
    "7_HTL_rate",
    "12_HTL_rate",
    "7_ATL_rate",
    "12_ATL_rate",
    "5_HTHW_rate",
    "5_ATAW_rate",
    "ODDS1",
    "ODDS2",
]


# Create a pipeline with imputer, scaler, feature selection, and model
def create_pipeline(model):
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="mean")),
            ("scaler", StandardScaler()),
            (
                "feature_selection",
                SelectFromModel(
                    estimator=RandomForestClassifier(n_estimators=100, random_state=42)
                ),
            ),
            ("model", model),
        ]
    )


# Function to print feature importance
def print_feature_importance(pipeline, feature_names):
    # Get the final estimator (model) from the pipeline
    model = pipeline.named_steps["model"]

    if hasattr(model, "feature_importances_"):
        # Get the feature selector from the pipeline
        feature_selector = pipeline.named_steps["feature_selection"]
        # Get the mask of selected features
        feature_mask = feature_selector.get_support()
        # Filter the feature names
        selected_features = [
            f for f, selected in zip(feature_names, feature_mask) if selected
        ]

        importances = model.feature_importances_

        # Sort features by importance
        feature_importance = sorted(zip(importances, selected_features), reverse=True)

        print("Top 10 most important features:")
        for i, (importance, feature) in enumerate(feature_importance, 1):
            print(f"{i}. {feature} ({importance:.6f})")
    else:
        print("This model doesn't have feature importances.")


# Load league information
with open(LEAGUES_PATH, "r") as file:
    leagues = json.load(file)
scores = []

for league in leagues:
    if league["is_active"] == False:
        continue
    # Define models with hyperparameter distributions for random search

    rf_params = {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": [None] + list(randint(10, 50).rvs(4)),
        "model__min_samples_split": randint(2, 20),
        "model__min_samples_leaf": randint(1, 10),
        "feature_selection__max_features": randint(
            8, len(features1) if league["can_draw"] else len(features2)
        ),
    }

    xgb_params = {
        "model__n_estimators": randint(100, 500),
        "model__max_depth": randint(3, 10),
        "model__learning_rate": uniform(0.01, 0.3),
        "model__subsample": uniform(0.5, 0.5),
        "model__colsample_bytree": uniform(0.5, 0.5),
        "feature_selection__max_features": randint(
            8, len(features1) if league["can_draw"] else len(features2)
        ),
    }

    models = {
        "RandomForest": (
            create_pipeline(RandomForestClassifier(random_state=42)),
            rf_params,
        ),
        "XGBoost": (create_pipeline(XGBClassifier(random_state=42)), xgb_params),
    }

    data_path = f"./match_infos/{league['folder_name']}/preprocess_data.csv"
    model_path = f"./models/{league['folder_name']}/label_encoder.joblib"
    df = pd.read_csv(data_path, encoding="utf-8")
    le = joblib.load(model_path)
    encoded_teams = set(le.classes_)

    df["HomeTeam"] = df["HomeTeam"].apply(
        lambda x: le.transform([x])[0] if x in encoded_teams else None
    )
    df["AwayTeam"] = df["AwayTeam"].apply(
        lambda x: le.transform([x])[0] if x in encoded_teams else None
    )
    if league["can_draw"]:
        df["FTR"] = np.select(
            [df["FTR"] == "H", df["FTR"] == "A", df["FTR"] == "D"], [2, 1, 0]
        )
    else:
        df = df[df["FTR"] != "D"]
        df["FTR"] = np.select([df["FTR"] == "H", df["FTR"] == "A"], [1, 0])
    if league["can_draw"]:
        df = df[(df["ODDS1"] != "-") & (df["ODDSX"] != "-") & (df["ODDS2"] != "-")]
    else:
        df = df[(df["ODDS1"] != "-") & (df["ODDS2"] != "-")]
    if league["can_draw"]:
        X = df[features1].fillna(0)
    else:
        X = df[features2].fillna(0)
    Y = df["FTR"]

    # XY preprocessing
    imputer = SimpleImputer()
    X_imputed = imputer.fit_transform(X)

    x_train = X_imputed
    y_train = Y

    # Perform random search for each model (except SVC)
    best_models = {}
    for name, (pipeline, params) in models.items():
        print(f"### Tuning {name}...")
        random_search = RandomizedSearchCV(
            pipeline, params, n_iter=80, cv=5, n_jobs=-1, verbose=1, random_state=42
        )
        random_search.fit(x_train, y_train)
        best_models[name] = random_search.best_estimator_
        print(f"Best parameters for {name}: {random_search.best_params_}")
        print(f"Best score for {name}: {random_search.best_score_}")
        print(f"\nFeature importance for {name}:")
        if league["can_draw"]:
            print_feature_importance(random_search.best_estimator_, features1)
        else:
            print_feature_importance(random_search.best_estimator_, features2)

    # Create ensemble model for result prediction
    ensemble = VotingClassifier(
        estimators=[(name, model) for name, model in best_models.items()], voting="soft"
    )

    ensemble.fit(x_train, y_train)
    model_path = f"./models/{league['folder_name']}/ml_model.joblib"
    # Save the ensemble model, label encoder, and score models
    joblib.dump(ensemble, model_path)
    print(f"{league['folder_name']} is completed!")

print("ML training is DONE!")
