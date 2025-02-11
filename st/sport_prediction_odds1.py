from abc import ABC, abstractmethod
from common.data import MatchPrediction, ProbabilityChoice
import pandas as pd
import os
import json
import numpy as np
import bittensor as bt
import random
from datetime import datetime, timezone
import pytz

HOTKEY_PATH = "./hotkeys.json"
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

def get_hotkeys():
    hotkeys = None
    with open(HOTKEY_PATH, "r", encoding="utf-8") as file:
        hotkeys = json.load(file)
    return hotkeys
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

    closing_odds = np.min(odds)
    diff_odds = abs(odds[0] - odds[1])
    idx = np.argmin(odds)
    if idx == 0:
        result = ProbabilityChoice.HOMETEAM
    else:
        result = ProbabilityChoice.AWAYTEAM
    hotkeys = get_hotkeys()
    print(hotkeys, ss58_address)
    min_prob = 0.95
    max_prob = 1
    if hotkeys is None or not hotkeys or not ss58_address in hotkeys:
        if diff_odds >= 1:
            min_prob = 1 / closing_odds
            min_prob = min_prob if min_prob > 0.95 else 0.95
            max_prob = 1
        else:
            min_prob = 0.8
            max_prob = 0.85
            if idx == 0:
                result = ProbabilityChoice.AWAYTEAM
            else:
                result = ProbabilityChoice.HOMETEAM
    else:
        min_prob = 1 / closing_odds
        min_prob = min_prob if min_prob > 0.95 else 0.95
        max_prob = 1
    prob = random.uniform(min_prob, max_prob)
    
    prediction.probabilityChoice = result
    prediction.probability = prob
    
    return prediction


prediction = MatchPrediction(
    predictionId=None,
    minerId=None,
    hotkey=None,
    matchId="b45314557329c2ec942453128ab100ec",
    matchDate="2025-02-11 01:00:00",
    sport=4,
    league="NBA",
    isScored=False,
    scoredDate=None,
    homeTeamName="Phoenix Suns",
    awayTeamName="Memphis Grizzlies",
    probabilityChoice=None,
    probability=None,
)
hotkey = "5C867hePvMYAkGXuScReRtNDG1yZNip9zAEZ5eAivaLiQPCF1"
result = make_match_prediction(prediction, hotkey, hotkey)
bt.logging.success(result)
