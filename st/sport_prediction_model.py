from abc import ABC, abstractmethod
from typing import List
from common.data import MatchPrediction, Sport, League, get_league_from_string, ProbabilityChoice
import numpy as np


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
    team, conf_scores, odds, result = predictor.predict_winner(prediction)

    prediction.probabilityChoice = result
    prediction.probability = np.max(conf_scores)
    
    return prediction
