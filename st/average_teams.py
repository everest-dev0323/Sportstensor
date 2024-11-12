#import needed Python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os

# graphics parameters of the notebook
# display graphs inline

# Make graphs prettier
pd.set_option('display.max_columns', 15)
pd.set_option('display.width', 400)
pd.set_option('plotting.matplotlib.register_converters', True)

# Make the fonts bigger
plt.rc('figure', figsize=(14, 7))
plt.rc('font', family='normal', weight='bold', size=15)

ROOT_DIR = './data'
LEAGUES_PATH = './leagues.json'

def get_historical_data(league):
    league_path = os.path.join(ROOT_DIR, league['folder_name'])
        
    # Initialize an empty DataFrame for the league
    leagues_df = pd.DataFrame()
    
    # Check if league path exists and process files
    if os.path.isdir(league_path):
        for filename in os.listdir(league_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(league_path, filename)
                df = pd.read_csv(file_path)
                leagues_df = pd.concat([leagues_df, df], ignore_index=True)
    leagues_df = leagues_df.sort_values(by='Date').reset_index(drop=True)
    return leagues_df

# Load league information
with open(LEAGUES_PATH, 'r') as file:
    leagues = json.load(file)

# Process each league
for league in leagues:
    if league['is_active'] == False:
        continue
    raw_data = get_historical_data(league)

    #Select useful features for datavisualization and analysis purposes
    E0_data = raw_data[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG","FTR","Season"]].copy()

    #create matchID column
    E0_data['matchID'] = E0_data.index

    def resultConverter(A):
        if A == 'H':
            return 'W'
        elif A =='A':
            return 'L'
        else:
            return 'D'

    def resultInverser(A):
        if A == 'W':
            return 'L'
        elif A == 'L':
            return 'W'
        else:
            return 'D'

    def ordinalResultConverter(A):
        if A == 'W':
            return 1
        elif A == 'L':
            return 0
        else:
            return 0.5
        
    #make dummies variables for FTR (result of match), HW = Home Win, AW = Away Win, D = draw
    E0_data['HW'] = E0_data.FTR.map(lambda x : 1 if x == 'H' else 0)
    E0_data['AW'] = E0_data.FTR.map(lambda x : 1 if x == 'A' else 0)
    if league['can_draw']:
        E0_data['D']= E0_data.FTR.map(lambda x : 1 if x == 'D' else 0)

    #make 2 different variable for the result of a match : 1 for the home team point of view, the other for the away team pt of view
    E0_data['HR'] = E0_data.FTR.map(lambda x : resultConverter(x))
    E0_data['AR'] = E0_data.HR.map(lambda x : resultInverser(x))

    #make ordinal variable for the home team point of view result (1 = win, 0.5 = Draw, 0 = loss)
    E0_data['ordinalHR'] = E0_data.HR.map(lambda x : ordinalResultConverter(x))

    #create teams list
    teams = pd.concat([E0_data['HomeTeam'], E0_data['AwayTeam']], ignore_index=True).unique()

    #create seasons list
    seasons = np.sort(E0_data['Season'].unique())
    #current season
    cur_season = seasons[-1]

    #create dictionary containing teams list by season
    teams_by_season = {season : E0_data[E0_data['Season'] == season]['HomeTeam'].unique() for season in seasons}

    avg_teams = pd.DataFrame({
        'Team': teams, 
        'Elo_Score': [0]*len(teams)
    })

    #Elo ranking method parameters
    k = 20.0
    d = 400.0
    c = 10.0

    # Initialize Elo scores with a DataFrame
    elo_df = pd.DataFrame(columns=['Team', 'Date', 'Elo_Score', 'gamma'])
    elo_df.set_index(['Team', 'Date'], inplace=True)

    # Initialize last season teams' average
    last_teams_ELO_av = 0.0
    prev_season_teams = teams.copy()
    prev_ELO_score_d = dict()

    for team in teams:
        prev_ELO_score_d[team] = 0.0

    # Process each season
    for season in E0_data['Season'].unique():
        season_data = E0_data[E0_data['Season'] == season]
        season_teams = season_data['HomeTeam'].unique()
        last_season_date = season_data['Date'].max()

        for Steam in season_teams:
            if not (Steam in prev_season_teams):
                prev_ELO_score_d[Steam] = last_teams_ELO_av

        # Process each match in the season
        def update_elo(row):
            Hteam, Ateam = row['HomeTeam'], row['AwayTeam']
            date = row['Date']

            # Get previous Elo scores
            l0H = prev_ELO_score_d.get(Hteam, 0.0)
            l0A = prev_ELO_score_d.get(Ateam, 0.0)

            # Calculate gamma and alpha
            gammaH = 1.0 / (1.0 + c**((l0A - l0H) / d))
            gammaA = 1.0 - gammaH
            alphaH = row['ordinalHR']
            alphaA = 1 - alphaH

            # Update Elo scores
            new_HT_Elo = l0H + k * (alphaH - gammaH)
            new_AT_Elo = l0A + k * (alphaA - gammaA)

            # Update elo_df immediately for the next match
            elo_df.loc[(Hteam, date), 'Elo_Score'] = l0H
            elo_df.loc[(Ateam, date), 'Elo_Score'] = l0A
            elo_df.loc[(Hteam, date), 'gamma'] = gammaH
            elo_df.loc[(Ateam, date), 'gamma'] = gammaA

            prev_ELO_score_d[Hteam] = new_HT_Elo
            prev_ELO_score_d[Ateam] = new_AT_Elo

            # Return gamma values for tracking purposes
            return pd.Series({
                'HTeamEloScore': new_HT_Elo,
                'ATeamEloScore': new_AT_Elo,
                'gammaH': gammaH,
                'gammaA': gammaA
            })
        # Apply the update to all matches in the season
        elo_updates = season_data.apply(update_elo, axis=1, result_type='expand')

        # Calculate last teams' average Elo at season end using only Team index
        season_elos = [prev_ELO_score_d[team] for team in season_teams]
        season_elos = np.sort(season_elos)  # Sort for consistent exclusion

        # Compute average Elo score for the last season's teams
        if len(season_elos) >= 17:
            last_teams_ELO_av = np.mean(np.sort(season_elos)[:-17])  # Exclude the bottom 17 Elo scores
        else:
            last_teams_ELO_av = np.mean(np.sort(season_elos))  # Exclude the bottom 17 Elo scores
        prev_season_teams = season_teams

    # Prepare elo_df for merging by resetting the index
    elo_df_reset = elo_df.reset_index()

    elo_df_reset = elo_df_reset.sort_values(by='Date')

    E0_HT_grpby = E0_data.groupby('HomeTeam')[['Season']]
    E0_AT_grpby = E0_data.groupby('AwayTeam')[['Season']]

    def average_elo(group, team_name):
        elo_score = prev_ELO_score_d.get(team_name, 0.0)
        latest_match_date = np.asarray(group['Date'])[-1]
        return pd.Series({
            'Elo_Score': elo_score,
            'LastMatchDate': latest_match_date
        })

    grp_by_HT_and_season = E0_data.groupby(['HomeTeam', 'Season'])
    grp_by_AT_and_season = E0_data.groupby(['AwayTeam', 'Season'])

    avg_teams[['Team', 'Elo_Score', 'LastMatchDate']] = elo_df_reset.groupby('Team').apply(lambda group: average_elo(group, group.name), include_groups=False).reset_index()

    def getTeamDay(row):
        team = row['Team']
        if team in E0_HT_grpby.groups:
            df1 = E0_HT_grpby.get_group(team)
            df1 = df1[df1['Season'] == cur_season]
        else:
            df1 = pd.DataFrame(columns=['Season'])  # Empty DataFrame if team has no HomeTeam history
        
        # Get data for the team as AwayTeam
        if team in E0_AT_grpby.groups:
            df2 = E0_AT_grpby.get_group(team)
            df2 = df2[df2['Season'] == cur_season]
        else:
            df2 = pd.DataFrame(columns=['Season'])  # Empty DataFrame if team has no AwayTeam history
        day = len(df1) + len(df2)
        TW_rate, TD_rate, TW_rate_7, TD_rate_7, TW_rate_12, TL_rate_7, TD_rate_12, TL_rate_12, HTHW_rate_5, ATAW_rate_5 = (0,) * 10
        last_TR_isW_1, last_TR_isL_1, last_TR_isW_2, last_TR_isL_2, last_TR_isW_3, last_TR_isL_3, last_HTHR_isW_1, last_HTHR_isL_1, last_HTHR_isW_2, last_HTHR_isL_2, last_ATAR_isW_1, last_ATAR_isL_1, last_ATAR_isW_2, last_ATAR_isL_2 = (0,) * 14
        if team in teams_by_season[cur_season]:
            #retrieve season specific results serie (1 win serie, 1 draw serie the loss  will be computed thanks to
            #the 2 others)
            if (team, cur_season) in grp_by_HT_and_season.groups:
                teamHomeResultsW_s = grp_by_HT_and_season.get_group((team,cur_season))['HW']
                teamHomeResults_s = grp_by_HT_and_season.get_group((team,cur_season))['HR']
                if league['can_draw']:
                    teamHomeResultsD_s = grp_by_HT_and_season.get_group((team,cur_season))['D']
            else:
                teamHomeResultsW_s = pd.Series(dtype='int')
                teamHomeResults_s = pd.Series(dtype='int')
                if league['can_draw']:
                    teamHomeResultsD_s = pd.Series(dtype='int')
            if (team, cur_season) in grp_by_AT_and_season.groups:
                teamAwayResultsW_s = grp_by_AT_and_season.get_group((team,cur_season))['AW']
                teamAwayResults_s = grp_by_AT_and_season.get_group((team,cur_season))['AR']
                if league['can_draw']:
                    teamAwayResultsD_s = grp_by_AT_and_season.get_group((team,cur_season))['D']
            else:
                teamAwayResultsW_s = pd.Series(dtype='int')
                teamAwayResults_s = pd.Series(dtype='int')
                if league['can_draw']:
                    teamAwayResultsD_s = pd.Series(dtype='int')
            teamResultsW_s = pd.concat([teamHomeResultsW_s, teamAwayResultsW_s]).sort_index()
            teamResults_s = pd.concat([teamHomeResults_s, teamAwayResults_s]).sort_index()
            if league['can_draw']:
                teamResultsD_s = pd.concat([teamHomeResultsD_s, teamAwayResultsD_s]).sort_index()
            #(0) compute HW rates, HL rates, AW rates, LW rates since begining of season
            TW_rate = teamResultsW_s.sum() / day
            if league['can_draw']:
                TD_rate = teamResultsD_s.sum() / day
            #k_last_HTR and k_last_ATR are just shifted versions of the results series
            last_TR_isW_1 = 1 if len(teamResults_s) >= 1 and np.asarray(teamResults_s)[-1] == 'W' else 0
            last_TR_isL_1 = 1 if len(teamResults_s) >= 1 and np.asarray(teamResults_s)[-1] == 'L' else 0
            last_TR_isW_2 = 1 if len(teamResults_s) >= 2 and np.asarray(teamResults_s)[-2] == 'W' else 0
            last_TR_isL_2 = 1 if len(teamResults_s) >= 2 and np.asarray(teamResults_s)[-2] == 'L' else 0
            last_TR_isW_3 = 1 if len(teamResults_s) >= 3 and np.asarray(teamResults_s)[-3] == 'W' else 0
            last_TR_isL_3 = 1 if len(teamResults_s) >= 3 and np.asarray(teamResults_s)[-3] == 'L' else 0
            last_HTHR_isW_1 = 1 if len(teamHomeResults_s) >= 1 and np.asarray(teamHomeResults_s)[-1] == 'W' else 0
            last_HTHR_isL_1 = 1 if len(teamHomeResults_s) >= 1 and np.asarray(teamHomeResults_s)[-1] == 'L' else 0
            last_HTHR_isW_2 = 1 if len(teamHomeResults_s) >= 2 and np.asarray(teamHomeResults_s)[-2] == 'W' else 0
            last_HTHR_isL_2 = 1 if len(teamHomeResults_s) >= 2 and np.asarray(teamHomeResults_s)[-2] == 'L' else 0
            last_ATAR_isW_1 = 1 if len(teamAwayResults_s) >= 1 and np.asarray(teamAwayResults_s)[-1] == 'W' else 0
            last_ATAR_isL_1 = 1 if len(teamAwayResults_s) >= 1 and np.asarray(teamAwayResults_s)[-1] == 'L' else 0
            last_ATAR_isW_2 = 1 if len(teamAwayResults_s) >= 2 and np.asarray(teamAwayResults_s)[-2] == 'W' else 0
            last_ATAR_isL_2 = 1 if len(teamAwayResults_s) >= 2 and np.asarray(teamAwayResults_s)[-2] == 'L' else 0
            
            #(i) compute 7_HTW_rate, 12_HTW_rate, 7_HTD_rate, 12_HTD_rate, 7_ATW_rate, 12_ATW_rate, 7_ATD_rate, 12_ATD_rate --> 8 features
            if len(teamResultsW_s) > 7:
                TW_rate_7 = np.asarray(teamResultsW_s)[-7:].mean()
            else:
                TW_rate_7 = np.asarray(teamResultsW_s).mean() if not teamResultsW_s.empty else 0
            if league['can_draw']:
                if len(teamResultsD_s) > 7:
                    TD_rate_7 = np.asarray(teamResultsD_s)[-7:].mean()
                else:
                    TD_rate_7 = np.asarray(teamResultsD_s).mean() if not teamResultsD_s.empty else 0
            TL_rate_7 = 1 - TW_rate_7 - TD_rate_7
            if len(teamResultsW_s) > 12:
                TW_rate_12 = np.asarray(teamResultsW_s)[-12:].mean()
            else:
                TW_rate_12 = np.asarray(teamResultsW_s).mean() if not teamResultsW_s.empty else 0
            if league['can_draw']:
                if len(teamResultsD_s) > 12:
                    TD_rate_12 = np.asarray(teamResultsD_s)[-12:].mean()
                else:
                    TD_rate_12 = np.asarray(teamResultsD_s).mean() if not teamResultsD_s.empty else 0
            TL_rate_12 = 1 - TW_rate_12 - TD_rate_12

            #(ii) compute 5_HTHW_rate and 5_ATAW_rate
            if len(teamHomeResultsW_s) > 5:
                HTHW_rate_5 = np.asarray(teamHomeResultsW_s)[-5:].mean()
            else:
                HTHW_rate_5 = np.asarray(teamHomeResultsW_s).mean() if not teamHomeResultsW_s.empty else 0
            if len(teamAwayResultsW_s) > 5:
                ATAW_rate_5 = np.asarray(teamAwayResultsW_s)[-5:].mean()
            else:
                ATAW_rate_5 = np.asarray(teamAwayResultsW_s).mean() if not teamAwayResultsW_s.empty else 0
        if league['can_draw']:
            return TW_rate, TD_rate, TW_rate_7, TD_rate_7, TW_rate_12, TL_rate_7, TD_rate_12, TL_rate_12, HTHW_rate_5, ATAW_rate_5, last_TR_isW_1, last_TR_isL_1, last_TR_isW_2, last_TR_isL_2, last_TR_isW_3, last_TR_isL_3, last_HTHR_isW_1, last_HTHR_isL_1, last_HTHR_isW_2, last_HTHR_isL_2, last_ATAR_isW_1, last_ATAR_isL_1, last_ATAR_isW_2, last_ATAR_isL_2
        else:
            return TW_rate, TW_rate_7, TW_rate_12, TL_rate_7, TL_rate_12, HTHW_rate_5, ATAW_rate_5, last_TR_isW_1, last_TR_isL_1, last_TR_isW_2, last_TR_isL_2, last_TR_isW_3, last_TR_isL_3, last_HTHR_isW_1, last_HTHR_isL_1, last_HTHR_isW_2, last_HTHR_isL_2, last_ATAR_isW_1, last_ATAR_isL_1, last_ATAR_isW_2, last_ATAR_isL_2
    if league['can_draw']:
        avg_teams[
                ['TW_rate', 'TD_rate', 'TW_rate_7', 'TD_rate_7', 'TW_rate_12', 'TL_rate_7', 'TD_rate_12', 'TL_rate_12', 'HTHW_rate_5', 'ATAW_rate_5', 
                'last_TR_isW_1', 'last_TR_isL_1', 'last_TR_isW_2', 'last_TR_isL_2', 'last_TR_isW_3', 'last_TR_isL_3', 
                'last_HTHR_isW_1', 'last_HTHR_isL_1', 'last_HTHR_isW_2', 'last_HTHR_isL_2', 'last_ATAR_isW_1', 'last_ATAR_isL_1', 'last_ATAR_isW_2', 'last_ATAR_isL_2']
                ] = avg_teams.apply(getTeamDay, axis=1, result_type='expand')
    else:
        avg_teams[
                ['TW_rate', 'TW_rate_7', 'TW_rate_12', 'TL_rate_7', 'TL_rate_12', 'HTHW_rate_5', 'ATAW_rate_5', 
                'last_TR_isW_1', 'last_TR_isL_1', 'last_TR_isW_2', 'last_TR_isL_2', 'last_TR_isW_3', 'last_TR_isL_3', 
                'last_HTHR_isW_1', 'last_HTHR_isL_1', 'last_HTHR_isW_2', 'last_HTHR_isL_2', 'last_ATAR_isW_1', 'last_ATAR_isL_1', 'last_ATAR_isW_2', 'last_ATAR_isL_2']
                ] = avg_teams.apply(getTeamDay, axis=1, result_type='expand')
        
    avg_teams = avg_teams.dropna(subset=['Team'])
    avg_teams.to_csv(f"./match_infos/{league['folder_name']}/avg_teams_info.csv", index=False, encoding='utf-8')
    print(f"{league['league_name']} is completed!")