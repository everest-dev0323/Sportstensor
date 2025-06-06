import datetime as dt
import os
from common.data import Sport, League, MatchPrediction
from st.sport_prediction_lstm import make_match_prediction


# from sportstensor.predictions import make_match_prediction


def mls():
    matchDate = "2024-08-20"
    match_prediction = MatchPrediction(
        matchId=1234,
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.SOCCER,
        league=League.MLS,
        homeTeamName="Inter Miami",
        awayTeamName="Miami Fusion",
    )

    match_prediction = make_match_prediction(match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )

    return match_prediction


def mlb():
    matchDate = "2024-08-25"
    match_prediction = MatchPrediction(
        matchId=1234,
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.BASEBALL,
        league=League.MLB,
        homeTeamName="Los Angeles Dodgers",
        awayTeamName="Oakland Athletics",
    )

    match_prediction = make_match_prediction(match_prediction)

    print("match_prediction", match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )


def epl():
    matchDate = "2024-09-20"
    match_prediction = MatchPrediction(
        matchId=1234,
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.SOCCER,
        league=League.EPL,
        homeTeamName="Arsenal",
        awayTeamName="Chelsea",
    )

    match_prediction = make_match_prediction(match_prediction)

    print("match_prediction", match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )


def nfl():
    matchDate = "2024-09-20"
    match_prediction = MatchPrediction(
        matchId=1234,
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.FOOTBALL,
        league=League.NFL,
        homeTeamName="Cincinnati Bengals",
        awayTeamName="Tampa Bay Buccaneers",
    )

    match_prediction = make_match_prediction(match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )

    return match_prediction


def nba():
    matchDate = "2024-09-20"
    match_prediction = MatchPrediction(
        matchId=1234,
        matchDate=dt.datetime.strptime(matchDate, "%Y-%m-%d"),
        sport=Sport.BASKETBALL,
        league=League.NBA,
        homeTeamName="Los Angeles Lakers",
        awayTeamName="Boston Celtics",
    )

    match_prediction = make_match_prediction(match_prediction)

    print(
        f"Match Prediction for {match_prediction.awayTeamName} at {match_prediction.homeTeamName} on {matchDate}: \
    Prediction: {match_prediction.probabilityChoice} ({match_prediction.get_predicted_team()}) with probability {match_prediction.probability}"
    )

    return match_prediction


if __name__ == "__main__":
    # mls()
    # mlb()
    # epl()
    # nfl()
    nba()
