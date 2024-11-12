import pandas as pd
import numpy as np
import pytz
import json
import joblib
from datetime import datetime
from fuzzywuzzy import process
from sklearn.impute import SimpleImputer
import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
import warnings

warnings.filterwarnings("ignore")
from common.data import ProbabilityChoice

# Set TensorFlow logging to only show errors
tf.get_logger().setLevel("ERROR")

LEAGUES_PATH = "./st/leagues.json"
UPCOMMING_MATCHES_PATH = "./st/data/next_matches.csv"

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

    def get_match_features(
        self, home_team, away_team, match_date, avg_team_df, can_draw
    ):
        def HTdaysBetweenDates(row):
            if not (pd.isnull(row["HTLastMatchDate"])):
                currDate = pd.to_datetime(row["Date"]).tz_localize(None)
                prevDate = pd.to_datetime(row["HTLastMatchDate"]).tz_localize(None)
                ndays = (currDate - prevDate).days
                if ndays < 20:
                    return ndays
                else:
                    return 0
            else:
                return 0

        def ATdaysBetweenDates(row):
            if not (pd.isnull(row["ATLastMatchDate"])):
                currDate = pd.to_datetime(row["Date"]).tz_localize(None)
                prevDate = pd.to_datetime(row["ATLastMatchDate"]).tz_localize(None)
                ndays = (currDate - prevDate).days
                if ndays < 20:
                    return ndays
                else:
                    return 0
            else:
                return 0

        df = pd.DataFrame(
            {"HomeTeam": [home_team], "AwayTeam": [away_team], "Date": [match_date]}
        )
        if can_draw:
            df = df.merge(
                avg_team_df.rename(
                    columns={
                        "Team": "HomeTeam",
                        "Elo_Score": "HTeamEloScore",
                        "TW_rate": "HTW_rate",
                        "TD_rate": "HTD_rate",
                        "TW_rate_7": "7_HTW_rate",
                        "TD_rate_7": "7_HTD_rate",
                        "TL_rate_7": "7_HTL_rate",
                        "TW_rate_12": "12_HTW_rate",
                        "TD_rate_12": "12_HTD_rate",
                        "TL_rate_12": "12_HTL_rate",
                        "HTHW_rate_5": "5_HTHW_rate",
                        "LastMatchDate": "HTLastMatchDate",
                    }
                ),
                how="left",
                left_on=["HomeTeam"],
                right_on=["HomeTeam"],
            )

            df = df.merge(
                avg_team_df.rename(
                    columns={
                        "Team": "AwayTeam",
                        "Elo_Score": "ATeamEloScore",
                        "TW_rate": "ATW_rate",
                        "TD_rate": "ATD_rate",
                        "TW_rate_7": "7_ATW_rate",
                        "TD_rate_7": "7_ATD_rate",
                        "TL_rate_7": "7_ATL_rate",
                        "TW_rate_12": "12_ATW_rate",
                        "TD_rate_12": "12_ATD_rate",
                        "TL_rate_12": "12_ATL_rate",
                        "ATAW_rate_5": "5_ATAW_rate",
                        "LastMatchDate": "ATLastMatchDate",
                    }
                ),
                how="left",
                left_on=["AwayTeam"],
                right_on=["AwayTeam"],
            )
        else:
            df = df.merge(
                avg_team_df.rename(
                    columns={
                        "Team": "HomeTeam",
                        "Elo_Score": "HTeamEloScore",
                        "TW_rate": "HTW_rate",
                        "TW_rate_7": "7_HTW_rate",
                        "TL_rate_7": "7_HTL_rate",
                        "TW_rate_12": "12_HTW_rate",
                        "TL_rate_12": "12_HTL_rate",
                        "HTHW_rate_5": "5_HTHW_rate",
                        "LastMatchDate": "HTLastMatchDate",
                    }
                ),
                how="left",
                left_on=["HomeTeam"],
                right_on=["HomeTeam"],
            )

            df = df.merge(
                avg_team_df.rename(
                    columns={
                        "Team": "AwayTeam",
                        "Elo_Score": "ATeamEloScore",
                        "TW_rate": "ATW_rate",
                        "TW_rate_7": "7_ATW_rate",
                        "TL_rate_7": "7_ATL_rate",
                        "TW_rate_12": "12_ATW_rate",
                        "TL_rate_12": "12_ATL_rate",
                        "ATAW_rate_5": "5_ATAW_rate",
                        "LastMatchDate": "ATLastMatchDate",
                    }
                ),
                how="left",
                left_on=["AwayTeam"],
                right_on=["AwayTeam"],
            )

        df["HTdaysSinceLastMatch"] = df.apply(HTdaysBetweenDates, axis=1)
        df["ATdaysSinceLastMatch"] = df.apply(ATdaysBetweenDates, axis=1)

        return df

    def get_match_odds(self, home_team, away_team, league, can_draw):
        df = pd.read_csv(UPCOMMING_MATCHES_PATH, encoding="utf-8")
        cur_match = df[
            (df["League"] == league)
            & (df["HomeTeam"] == home_team)
            & (df["AwayTeam"] == away_team)
        ]
        if cur_match.empty:
            return None
        if can_draw:
            return cur_match[["ODDS1", "ODDSX", "ODDS2"]].values.flatten().tolist()
        else:
            return cur_match[["ODDS1", "ODDS2"]].values.flatten().tolist()

    def get_best_match(self, team_name, encoded_teams):
        match, score = process.extractOne(team_name, encoded_teams)
        print(match, score)
        if score >= 80:
            return match
        else:
            return None

    def predict_winner(self, prediction):
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
                le = joblib.load(
                    "./st/models/england-premier-league/label_encoder.joblib"
                )
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

        can_draw = competition["can_draw"]
        conf_scores = [0, 0, 0] if can_draw else [0, 0]
        odds = [0, 0, 0] if can_draw else [0, 0]
        if competition is None:
            return home_team, conf_scores, odds, ProbabilityChoice.HOMETEAM
        model_path = f"./st/models/{competition['folder_name']}"
        data_path = f"./st/match_infos/{competition['folder_name']}"
        fbmodel = joblib.load(f"{model_path}/ml_model.joblib")
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
            return home_team, conf_scores, odds, ProbabilityChoice.HOMETEAM

        prev_matches = pd.read_csv(f"{data_path}/preprocess_data.csv", encoding="utf-8")
        avg_teams = pd.read_csv(f"{data_path}/avg_teams_info.csv", encoding="utf-8")

        cur_match = self.get_match_features(
            home_match, away_match, match_date, avg_teams, can_draw
        )

        league = competition["folder_name"]
        odds = self.get_match_odds(home_match, away_match, league, can_draw)
        if odds is None:
            return home_team, conf_scores, odds, ProbabilityChoice.HOMETEAM
        if can_draw:
            cur_match = cur_match[features1]
            cur_match.loc[:, ["ODDS1", "ODDSX", "ODDS2"]] = odds
        else:
            cur_match = cur_match[features2]
            cur_match.loc[:, ["ODDS1", "ODDS2"]] = odds

        result = pd.concat([cur_match], ignore_index=True)
        result["HomeTeam"] = result["HomeTeam"].apply(
            lambda x: le.transform([x])[0] if x in encoded_teams else None
        )
        result["AwayTeam"] = result["AwayTeam"].apply(
            lambda x: le.transform([x])[0] if x in encoded_teams else None
        )

        imputer = SimpleImputer()
        X = imputer.fit_transform(result)

        y_pred = fbmodel.predict(X)
        y_pred_proba = fbmodel.predict_proba(X)
        if can_draw:
            if y_pred == 2:
                return (
                    home_team,
                    y_pred_proba[0],
                    np.array(odds),
                    ProbabilityChoice.HOMETEAM,
                )
            elif y_pred == 1:
                return (
                    away_team,
                    y_pred_proba[0],
                    np.array(odds),
                    ProbabilityChoice.AWAYTEAM,
                )
            else:
                return "DRAW", y_pred_proba[0], np.array(odds), ProbabilityChoice.DRAW
        else:
            if y_pred == 1:
                return (
                    home_team,
                    y_pred_proba[0],
                    np.array(odds),
                    ProbabilityChoice.HOMETEAM,
                )
            else:
                return (
                    away_team,
                    y_pred_proba[0],
                    np.array(odds),
                    ProbabilityChoice.AWAYTEAM,
                )


# predictor = FootballPredictor()

# competition = "Bundesliga"
# home_teams = ["Tion"]
# away_teams = ["Freiburg"]
# date = "2024-11-08 09:35:30"

# for home_team, away_team in zip(home_teams, away_teams):
#     result, conf_socres, odds = predictor.predict_winner(home_team, away_team, date, competition)
#     prediction = f">>>>>>  Home Team: {home_team}, Away Team: {away_team}, Date: {date}, Competition: {competition}, Result: {result}  <<<<<<<"
#     print(prediction, conf_socres, odds)

#     os.makedirs("./logging", exist_ok=True)
#     file_path = './logging/prediction.csv'
#     pred_match = pd.DataFrame()
#     # Check if file exists and is non-empty
#     if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
#         df = pd.read_csv(file_path, encoding='utf-8')
#         pred_match = df[(df['Date'] == date) & (df['HomeTeam'] == home_team) & (df['AwayTeam'] == away_team) & (df['League'] == competition)]
#     else:
#         df = pd.DataFrame()

#     if pred_match.empty:
#         pred_match = pd.DataFrame({
#             'Date': [date],
#             'HomeTeam': [home_team],
#             'AwayTeam': [away_team],
#             'League': [competition],
#             'Prediction': [result],
#             'ProbHome': [conf_socres[0]],
#             'ProbDraw': [conf_socres[2]],
#             'ProbAway': [conf_socres[1]],
#             'ODDS1': [odds[0]],
#             'ODDSX': [odds[1]],
#             'ODDS2': [odds[2]]
#         })
#         pred_matchs = pd.concat([df, pred_match], ignore_index=True)
#         pred_matchs.to_csv('./logging/prediction.csv', index=False, encoding='utf-8')
