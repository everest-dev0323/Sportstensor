from abc import ABC, abstractmethod
from common.data import MatchPrediction, ProbabilityChoice
import pandas as pd
import os
import numpy as np
import bittensor as bt
import random
from datetime import datetime, timezone
import pytz


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

def make_match_prediction(prediction: MatchPrediction, hotkey=None, ss58_address=None):
    # Lazy import to avoid circular dependency
    from st.odds_predictor import Predictor

    predictor = Predictor()
    bt.logging.warning(prediction)
    predictor = Predictor()

    predict, conf_scores, odds, result, league = predictor.predict_winner(prediction)

    home_team = prediction.homeTeamName
    away_team = prediction.awayTeamName

    time_difference = 0
    current_datetime_utc = datetime.now(pytz.UTC)
    try:
        match_datetime_utc = prediction.matchDate.replace(tzinfo=pytz.UTC)
        # Get the current time in UTC
        # Calculate the difference in minutes
        time_difference = (
            match_datetime_utc - current_datetime_utc
        ).total_seconds() / 60
        print(f"The difference is {time_difference} minutes.")
    except Exception as e:
        time_difference = 0
        bt.logging.warning(
            "To calculate the difference in minutes between a match date and the current time",
            e,
        )

    closing_odds = np.min(odds)
    idx = np.argmin(odds)
    if idx == 0:
        result = ProbabilityChoice.HOMETEAM
    else:
        result = ProbabilityChoice.AWAYTEAM

    min_prob = 1 / closing_odds
    min_prob = min_prob if min_prob > 0.95 else 0.95
    max_prob = 1
    prob = random.uniform(min_prob, max_prob)
    
    prediction.probabilityChoice = result
    prediction.probability = prob
    try:
        os.makedirs("./st/logging_odds", exist_ok=True)
        file_path = f"./st/logging_odds/{league}_{ss58_address}.csv"
        # Check if file exists and is non-empty
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            df = pd.read_csv(file_path, encoding="utf-8")
        else:
            df = pd.DataFrame()

        match = pd.DataFrame(
            {
                "Date": [match_datetime_utc.strftime("%Y-%m-%d %H:%M:%S")],
                "HomeTeam": [home_team],
                "AwayTeam": [away_team],
                "Prediction": [result.value],
                "Probability": [prob],
                "ProbHome": [conf_scores[0]],
                "ProbAway": [conf_scores[1]],
                "ODDS1": [odds[0]],
                "ODDS2": [odds[1]],
                "UpdatedTime": [current_datetime_utc.strftime("%Y-%m-%d %H:%M:%S")],
                "Hotkey": [hotkey],
                "DiffTime": [time_difference],
            }
        )
        matches = pd.concat([df, match], ignore_index=True)
        matches.to_csv(file_path, index=False, encoding="utf-8")
    except Exception as e:
        bt.logging.warning("To save the prediction logging for each validator.", e)
    return prediction
