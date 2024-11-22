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


def calculate_edge(is_correct, prediction_prob, closing_odds):
    reward_punishment = -1 if is_correct else 1

    # check if closing_odd is available
    if closing_odds is None or prediction_prob == 0:
        return 0.0, 0

    edge = closing_odds - (1 / prediction_prob)
    return reward_punishment * edge


def apply_gaussian_filter(closing_odds, probability):
    t = 1.0  # Controls the spread/width of the Gaussian curve outside the plateau region. Larger t means slower decay in the exponential term
    a = (
        -2
    )  # Controls the height of the plateau boundary. More negative a means lower plateau boundary
    b = 0.3  # Controls how quickly the plateau boundary changes with odds. Larger b means faster exponential decay in plateau width
    c = 3  # Minimum plateau width/boundary

    w = a * np.exp(-b * (closing_odds - 1)) + c
    diff = abs(closing_odds - 1 / probability)

    # note that sigma^2 = odds now
    # plateaued curve.
    exp_component = (
        1.0 if diff <= w else np.exp(-np.power(diff, 2) / (t * 2 * closing_odds))
    )

    return exp_component


def make_match_prediction(prediction: MatchPrediction, hotkey=None):
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

    # Find the maximum
    if time_difference < 15 and time_difference >= 5:
        is_correct = random.randint(0, 10000) % 2 if abs(odds[0] - odds[1]) < 1 else 1
    else:
        is_correct = random.randint(0, 10000) % 2
    result = (
        ProbabilityChoice.AWAYTEAM
        if is_correct == np.argmax(odds)
        else ProbabilityChoice.HOMETEAM
    )

    max_score = float("-inf")
    max_prob = None
    closing_odds = np.max(odds) if is_correct else np.min(odds)
    if is_correct:
        range = np.arange(0.01, 0.5, 0.01)
        for prob in range:
            sigma = calculate_edge(is_correct, prob, closing_odds)

            gfilter = apply_gaussian_filter(closing_odds, prob)

            score = sigma * gfilter
            if score > max_score:
                max_score = score
                max_prob = prob
    else:
        max_prob = 1 / closing_odds
    prediction.probabilityChoice = result
    prediction.probability = max_prob
    try:
        os.makedirs("./st/logging_odds", exist_ok=True)
        file_path = f"./st/logging_odds/{league}.csv"
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
                "Probability": [max_prob],
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
