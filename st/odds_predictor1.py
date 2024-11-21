import pandas as pd
import numpy as np
import pytz
import json
import joblib
from datetime import datetime
from fuzzywuzzy import process
from sklearn.impute import SimpleImputer
import os
from common.data import MatchPrediction

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
from common.data import ProbabilityChoice

# Set TensorFlow logging to only show errors
tf.get_logger().setLevel("ERROR")

LEAGUES_PATH = "./leagues.json"
UPCOMMING_MATCHES_PATH = "./data/next_matches"

mismatch_teams = {
    "Washington Redskins": "Washington Commanders",
    "St. Louis Rams": "Los Angeles Rams",
    "San Diego Chargers": "Los Angeles Chargers",
    "Oakland Raiders": "Las Vegas Raiders",
    "San Diego": "Los Angeles Chargers",
}


class Predictor:
    def __init__(self):
        # Load league information
        with open(LEAGUES_PATH, "r", encoding="utf-8") as file:
            self.leagues = json.load(file)

    def get_match_odds(self, home_team, away_team, league):
        df = pd.read_csv(f"{UPCOMMING_MATCHES_PATH}/{league}.csv", encoding="utf-8")
        cur_match = df[
            (df["HomeTeam"] == home_team)
            & (df["AwayTeam"] == away_team)
            & (df["ODDS1"] != "-")
            & (df["ODDS2"] != "-")
        ]
        if cur_match.empty:
            return None
        cur_match = cur_match.iloc[0]
        return cur_match[["ODDS1", "ODDS2"]].values.flatten().tolist()

    def get_best_match(self, team_name, encoded_teams):
        match, score = process.extractOne(team_name, encoded_teams)
        if score >= 80:
            return match
        else:
            return None

    def predict_winner(self, prediction: MatchPrediction):
        match_date = prediction.matchDate
        home_team = prediction.homeTeamName
        away_team = prediction.awayTeamName
        competition = prediction.league if prediction.league else None
        sport = prediction.sport if prediction.sport else None
        # Convert match_date to datetime if it's a string
        if isinstance(match_date, str):
            try:
                match_date = datetime.strptime(match_date, "%Y-%m-%d %H:%M:%S").replace(
                    tzinfo=pytz.UTC
                )
            except ValueError:
                try:
                    match_date = datetime.strptime(match_date, "%Y-%m-%d").replace(
                        tzinfo=pytz.UTC
                    )
                except ValueError:
                    raise ValueError(
                        "Invalid date format. Use 'YYYY-MM-DD HH:MM:SS' or 'YYYY-MM-DD'"
                    )

        # Ensure match_date is timezone-aware
        if match_date.tzinfo is None:
            match_date = match_date.replace(tzinfo=pytz.UTC)

        competition = next(
            (league for league in self.leagues if league["league_name"] == competition),
            None,
        )
        if competition is None:
            if sport == 1:
                le = joblib.load("./models/england-premier-league/label_encoder.joblib")
                encoded_teams = set(le.classes_)
                home_match = self.get_best_match(home_team, encoded_teams)
                away_match = self.get_best_match(away_team, encoded_teams)
                if (home_match is None) or (away_match is None):
                    competition = next(
                        (
                            league
                            for league in self.leagues
                            if league["league_name"] == "English Premier League"
                        ),
                        None,
                    )
                else:
                    competition = next(
                        (
                            league
                            for league in self.leagues
                            if league["league_name"] == "American Major League Soccer"
                        ),
                        None,
                    )
            else:
                competition = next(
                    (
                        league
                        for league in self.leagues
                        if league["sport_type"] == sport
                    ),
                    None,
                )

        conf_scores = [0.5, 0.5]
        odds = [2, 2]
        if competition is None:
            return (
                home_team,
                conf_scores,
                odds,
                ProbabilityChoice.HOMETEAM,
                "mismatch",
            )
        model_path = f"./models/{competition['folder_name']}"
        le = joblib.load(f"{model_path}/label_encoder.joblib")
        encoded_teams = set(le.classes_)
        home_match = self.get_best_match(home_team, encoded_teams)
        away_match = self.get_best_match(away_team, encoded_teams)

        if (home_team in mismatch_teams.keys()) and (
            competition["league_name"] == "NFL"
        ):
            home_match = mismatch_teams[home_team]

        if (away_team in mismatch_teams.keys()) and (
            competition["league_name"] == "NFL"
        ):
            away_match = mismatch_teams[away_team]

        if (home_match is None) or (away_match is None):
            return (
                home_team,
                conf_scores,
                odds,
                ProbabilityChoice.HOMETEAM,
                "mismatch",
            )

        league = competition["folder_name"]
        odds = self.get_match_odds(home_match, away_match, league)
        if odds is None:
            return (
                home_team,
                conf_scores,
                [2, 2],
                ProbabilityChoice.HOMETEAM,
                "mismatch",
            )
        conf_scores = [1 / float(value) for value in odds]
        y_pred = np.argmax(conf_scores)

        if y_pred == 0:
            return (
                home_team,
                conf_scores,
                np.array(odds),
                ProbabilityChoice.AWAYTEAM,
                league,
            )
        else:
            return (
                away_team,
                conf_scores,
                np.array(odds),
                ProbabilityChoice.HOMETEAM,
                league,
            )
