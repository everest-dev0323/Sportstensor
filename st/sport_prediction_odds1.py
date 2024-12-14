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
    from st.odds_predictor1 import Predictor

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

    is_correct = 0 if abs(odds[0] - odds[1]) > 1.5 else 1

    closing_odds = np.min(odds) if is_correct else np.max(odds)
    idx = np.argmin(odds)
    if is_correct == 1 and idx == 0:
        result = ProbabilityChoice.HOMETEAM
    elif is_correct == 1 and idx == 1:
        result = ProbabilityChoice.AWAYTEAM
    elif is_correct == 0 and idx == 0:
        result = ProbabilityChoice.AWAYTEAM
    else:
        result = ProbabilityChoice.HOMETEAM

    min_prob = 1 / closing_odds
    min_prob = min_prob if min_prob > 0.5 else 0.51
    max_prob = 0.99
    prob = random.uniform(min_prob, max_prob)
    
    prediction.probabilityChoice = result
    prediction.probability = prob
    try:
        os.makedirs("./logging_odds", exist_ok=True)
        file_path = f"./logging_odds/{league}_{ss58_address}.csv"
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


prediction = MatchPrediction(
    predictionId=None,
    minerId=None,
    hotkey=None,
    matchId="b45314557329c2ec942453128ab100ec",
    matchDate="2024-12-06 01:00:00",
    sport=4,
    league="NBA",
    isScored=False,
    scoredDate=None,
    homeTeamName="Memphis Grizzlies",
    awayTeamName="Sacramento Kings",
    probabilityChoice=None,
    probability=None,
)
hotkey = "5FFApaS75bv5pJHfAp2FVLBj9ZaXuFDjEypsaBNc1wCfe52v"
result = make_match_prediction(prediction, hotkey)
bt.logging.success(result)
