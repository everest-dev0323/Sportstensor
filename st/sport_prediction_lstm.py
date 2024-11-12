from abc import ABC, abstractmethod
from common.data import MatchPrediction
import pandas as pd
import os
import numpy as np
import bittensor as bt


class SportPredictionModel(ABC):
    def __init__(self, prediction):
        self.prediction = prediction
        self.huggingface_model = None

    @abstractmethod
    def make_prediction(self):
        pass

    def set_default_scores(self):
        self.prediction.homeTeamScore = 0
        self.prediction.awayTeamScore = 0


def make_match_prediction(prediction: MatchPrediction):
    # Lazy import to avoid circular dependency
    from st.lstm_predictor import Predictor

    predictor = Predictor()
    bt.logging.warning(prediction)
    predictor = Predictor()

    predict, conf_scores, odds, result, league, can_draw = predictor.predict_winner(
        prediction
    )

    home_team = prediction.homeTeamName
    away_team = prediction.awayTeamName

    os.makedirs("./st/logging_lstm", exist_ok=True)
    file_path = f"./st/logging_lstm/{league}.csv"
    pred_match = pd.DataFrame()
    match_date = prediction.matchDate.strftime("%Y-%m-%d")
    # Check if file exists and is non-empty
    if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
        df = pd.read_csv(file_path, encoding="utf-8")
        pred_match = df[
            (df["Date"] == match_date)
            & (df["HomeTeam"] == home_team)
            & (df["AwayTeam"] == away_team)
        ]
    else:
        df = pd.DataFrame()

    if pred_match.empty:
        if can_draw:
            pred_match = pd.DataFrame(
                {
                    "Date": [match_date],
                    "HomeTeam": [home_team],
                    "AwayTeam": [away_team],
                    "Prediction": [predict],
                    "ProbHome": [conf_scores[0]],
                    "ProbDraw": [conf_scores[2]],
                    "ProbAway": [conf_scores[1]],
                    "ODDS1": [odds[0]],
                    "ODDSX": [odds[1]],
                    "ODDS2": [odds[2]],
                }
            )
        else:
            pred_match = pd.DataFrame(
                {
                    "Date": [match_date],
                    "HomeTeam": [home_team],
                    "AwayTeam": [away_team],
                    "Prediction": [predict],
                    "ProbHome": [conf_scores[1]],
                    "ProbAway": [conf_scores[0]],
                    "ODDS1": [odds[0]],
                    "ODDS2": [odds[1]],
                }
            )
        pred_matchs = pd.concat([df, pred_match], ignore_index=True)
        pred_matchs.to_csv(file_path, index=False, encoding="utf-8")

    prediction.probabilityChoice = result
    prediction.probability = np.max(conf_scores)

    return prediction
