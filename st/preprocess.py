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
plt.rc('font', family='sans-serif', weight='bold', size=15)

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
    if league['can_draw']:
        E0_data = raw_data[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG","FTR","ODDS1","ODDSX","ODDS2","Season"]].copy()
    else:
        E0_data = raw_data[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG","FTR","ODDS1","ODDS2","Season"]].copy()
    #create teams list
    teams = pd.concat([E0_data['HomeTeam'], E0_data['AwayTeam']], ignore_index=True).unique()
    #create seasons list
    seasons = np.sort(E0_data['Season'].unique())
    #create dictionary containing teams list by season
    teams_by_season = {season : E0_data[E0_data['Season'] == season]['HomeTeam'].unique() for season in seasons}
    E0_HT_grpby = E0_data.groupby('HomeTeam')[['Date', 'Season']]
    E0_AT_grpby = E0_data.groupby('AwayTeam')[['Date', 'Season']]

    def getTeamDday(row, position):
        x = row['HomeTeam'] if position == 'Home' else row['AwayTeam']
        y = row['Date']
        season = row['Season']
        if x in E0_HT_grpby.groups:
            df1 = E0_HT_grpby.get_group(x)
            df1 = df1[(df1['Date'] < y) & (df1['Season'] == season)]
        else:
            df1 = pd.DataFrame(columns=['Date'])  # Empty DataFrame if team has no HomeTeam history
        
        # Get data for the team as AwayTeam
        if x in E0_AT_grpby.groups:
            df2 = E0_AT_grpby.get_group(x)
            df2 = df2[(df2['Date'] < y) & (df2['Season'] == season)]
        else:
            df2 = pd.DataFrame(columns=['Date'])  # Empty DataFrame if team has no AwayTeam history
        day = len(df1) + len(df2)
        return day

    E0_data['HomeTeamDay'] = E0_data.apply(getTeamDday, position='Home', axis=1)
    E0_data['AwayTeamDay'] = E0_data.apply(getTeamDday, position='Away', axis=1)

    E0_data['ones'] = 1
    for season in seasons:
        for team in teams_by_season[season]:
            sH = E0_data[(E0_data['HomeTeam'] == team) & (E0_data['Season'] == season)]['ones']
            E0_data.loc[sH.index, 'HomeTeamHomeDay'] = sH.cumsum()
            
            sA = E0_data[(E0_data['AwayTeam'] == team) & (E0_data['Season'] == season)]['ones']
            E0_data.loc[sA.index, 'AwayTeamAwayDay'] = sA.cumsum()

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

    grp_by_HT = E0_data.groupby('HomeTeam')
    grp_by_AT = E0_data.groupby('AwayTeam')

    grp_by_HT_and_season = E0_data.groupby(['HomeTeam', 'Season'])
    grp_by_AT_and_season = E0_data.groupby(['AwayTeam', 'Season'])

    #past performance features engineering
    for team in teams:
        #we retrieve results series of the team
        if team in grp_by_HT.groups:
            teamHomeResults_s = grp_by_HT.get_group(team)['HR']
        else:
            teamHomeResults_s = pd.Series(dtype='int')  # Empty series if no HomeTeam results
        if team in grp_by_AT.groups:
            teamAwayResults_s = grp_by_AT.get_group(team)['AR']
        else:
            teamAwayResults_s = pd.Series(dtype='int')  # Empty series if no AwayTeam results
        #combine these 2 series and sort the obtained serie
        teamResults_s = pd.concat([teamHomeResults_s, teamAwayResults_s]).sort_index()

        #(i) compute k_last_HR and k_last_AR --> 6 features
        lag1TeamResults_d = teamResults_s.shift(1).to_dict()
        lag2TeamResults_d = teamResults_s.shift(2).to_dict()
        lag3TeamResults_d = teamResults_s.shift(3).to_dict()
        
        #k_last_HTR and k_last_ATR are just shifted versions of the results series
        E0_data.loc[teamHomeResults_s.index,'1_last_HTR'] = E0_data.loc[teamHomeResults_s.index,:].index.map(lambda x : lag1TeamResults_d[x])
        E0_data.loc[teamHomeResults_s.index,'2_last_HTR'] = E0_data.loc[teamHomeResults_s.index,:].index.map(lambda x : lag2TeamResults_d[x])
        E0_data.loc[teamHomeResults_s.index,'3_last_HTR'] = E0_data.loc[teamHomeResults_s.index,:].index.map(lambda x : lag3TeamResults_d[x])
        E0_data.loc[teamAwayResults_s.index,'1_last_ATR'] = E0_data.loc[teamAwayResults_s.index,:].index.map(lambda x : lag1TeamResults_d[x])
        E0_data.loc[teamAwayResults_s.index,'2_last_ATR'] = E0_data.loc[teamAwayResults_s.index,:].index.map(lambda x : lag2TeamResults_d[x])
        E0_data.loc[teamAwayResults_s.index,'3_last_ATR'] = E0_data.loc[teamAwayResults_s.index,:].index.map(lambda x : lag3TeamResults_d[x])
        
        #(ii) Compute k_last_HTRH and k_last ATAR --> 4 features
        #we need here to diferentiate home results and past results. Python dictionaries allows the program to access to
        #needed data faster than with a Pandas serie
        lag1TeamHomeResults_d = teamHomeResults_s.shift(1).to_dict()
        lag2TeamHomeResults_d = teamHomeResults_s.shift(2).to_dict()
        lag1TeamAwayResults_d = teamAwayResults_s.shift(1).to_dict()
        lag2TeamAwayResults_d = teamAwayResults_s.shift(2).to_dict()
        
        E0_data.loc[teamHomeResults_s.index,'1_last_HTHR'] = E0_data.loc[teamHomeResults_s.index,:].index.map(lambda x : lag1TeamHomeResults_d[x])
        E0_data.loc[teamHomeResults_s.index,'2_last_HTHR'] = E0_data.loc[teamHomeResults_s.index,:].index.map(lambda x : lag2TeamHomeResults_d[x])
        E0_data.loc[teamAwayResults_s.index,'1_last_ATAR'] = E0_data.loc[teamAwayResults_s.index,:].index.map(lambda x : lag1TeamAwayResults_d[x])
        E0_data.loc[teamAwayResults_s.index,'2_last_ATAR'] = E0_data.loc[teamAwayResults_s.index,:].index.map(lambda x : lag2TeamAwayResults_d[x])
        
        #(iii) rates based features : we need to get only season specific results series (to avoid taking previous season results into season rates)
        for season in seasons:
            
            if team in teams_by_season[season]:
                #retrieve season specific results serie (1 win serie, 1 draw serie the loss  will be computed thanks to
                #the 2 others)
                if (team, season) in grp_by_HT_and_season.groups:
                    teamHomeResultsW_s = grp_by_HT_and_season.get_group((team,season))['HW']
                    if league['can_draw']:
                        teamHomeResultsD_s = grp_by_HT_and_season.get_group((team,season))['D']
                else:
                    teamHomeResultsW_s = pd.Series(dtype='int')
                    if league['can_draw']:
                        teamHomeResultsD_s = pd.Series(dtype='int')
                if (team, season) in grp_by_AT_and_season.groups:
                    teamAwayResultsW_s = grp_by_AT_and_season.get_group((team,season))['AW']
                    if league['can_draw']:
                        teamAwayResultsD_s = grp_by_AT_and_season.get_group((team,season))['D']
                else:
                    teamAwayResultsW_s = pd.Series(dtype='int')
                    if league['can_draw']:
                        teamAwayResultsD_s = pd.Series(dtype='int')
                teamResultsW_s = pd.concat([teamHomeResultsW_s, teamAwayResultsW_s]).sort_index()
                if league['can_draw']:
                    teamResultsD_s = pd.concat([teamHomeResultsD_s, teamAwayResultsD_s]).sort_index()
            
                #(0) compute HW rates, HL rates, AW rates, LW rates since begining of season
                teamResultsWCumul_d = teamResultsW_s.shift(1).cumsum().to_dict()
                if league['can_draw']:
                    teamResultsDCumul_d = teamResultsD_s.shift(1).cumsum().to_dict()

                #(i) compute 7_HTW_rate, 12_HTW_rate, 7_HTD_rate, 12_HTD_rate, 7_ATW_rate, 12_ATW_rate, 7_ATD_rate, 12_ATD_rate --> 8 features
                win7TeamResultsW_d = teamResultsW_s.shift(1).rolling(window = 7, min_periods = 5).mean().to_dict()
                win12TeamResultsW_d = teamResultsW_s.shift(1).rolling(window = 12, min_periods = 8).mean().to_dict()
                if league['can_draw']:
                    win7TeamResultsD_d = teamResultsD_s.shift(1).rolling( window = 7, min_periods = 5).mean().to_dict()
                    win12TeamResultsD_d = teamResultsD_s.shift(1).rolling( window = 12, min_periods = 8).mean().to_dict()
            
                E0_data.loc[teamHomeResultsW_s.index,'HTW_rate'] = E0_data.loc[teamHomeResultsW_s.index,:].index.map(lambda x : teamResultsWCumul_d[x])
                E0_data.loc[teamAwayResultsW_s.index,'ATW_rate'] = E0_data.loc[teamAwayResultsW_s.index,:].index.map(lambda x : teamResultsWCumul_d[x])
                if league['can_draw']:
                    E0_data.loc[teamHomeResultsW_s.index,'HTD_rate'] = E0_data.loc[teamHomeResultsW_s.index,:].index.map(lambda x : teamResultsDCumul_d[x])
                    E0_data.loc[teamAwayResultsW_s.index,'ATD_rate'] = E0_data.loc[teamAwayResultsW_s.index,:].index.map(lambda x : teamResultsDCumul_d[x])
            
                E0_data.loc[teamHomeResultsW_s.index,'7_HTW_rate'] = E0_data.loc[teamHomeResultsW_s.index,:].index.map(lambda x : win7TeamResultsW_d[x])
                E0_data.loc[teamHomeResultsW_s.index,'12_HTW_rate'] = E0_data.loc[teamHomeResultsW_s.index,:].index.map(lambda x : win12TeamResultsW_d[x])
                E0_data.loc[teamAwayResultsW_s.index,'7_ATW_rate'] = E0_data.loc[teamAwayResultsW_s.index,:].index.map(lambda x : win7TeamResultsW_d[x])
                E0_data.loc[teamAwayResultsW_s.index,'12_ATW_rate'] = E0_data.loc[teamAwayResultsW_s.index,:].index.map(lambda x : win12TeamResultsW_d[x])
                if league['can_draw']:
                    E0_data.loc[teamHomeResultsD_s.index,'7_HTD_rate'] = E0_data.loc[teamHomeResultsD_s.index,:].index.map(lambda x : win7TeamResultsD_d[x])
                    E0_data.loc[teamAwayResultsD_s.index,'7_ATD_rate'] = E0_data.loc[teamAwayResultsD_s.index,:].index.map(lambda x : win7TeamResultsD_d[x])
                    E0_data.loc[teamHomeResultsD_s.index,'12_HTD_rate'] = E0_data.loc[teamHomeResultsD_s.index,:].index.map(lambda x : win12TeamResultsD_d[x])
                    E0_data.loc[teamAwayResultsD_s.index,'12_ATD_rate'] = E0_data.loc[teamAwayResultsD_s.index,:].index.map(lambda x : win12TeamResultsD_d[x])

                #(ii) compute 5_HTHW_rate and 5_ATAW_rate
                win5TeamResultsHomeW_d = teamHomeResultsW_s.shift(1).rolling( window = 5, min_periods = 3).mean().to_dict()
                win5TeamResultsAwayW_d = teamAwayResultsW_s.shift(1).rolling( window = 5, min_periods = 3).mean().to_dict()
                E0_data.loc[teamHomeResultsW_s.index,'5_HTHW_rate'] = E0_data.loc[teamHomeResultsW_s.index,:].index.map(lambda x : win5TeamResultsHomeW_d[x])
                E0_data.loc[teamAwayResultsW_s.index,'5_ATAW_rate'] = E0_data.loc[teamAwayResultsW_s.index,:].index.map(lambda x : win5TeamResultsAwayW_d[x])
                
                #(iii) compute HTHW_rate, ATAW_rate, HTHD_rate, ATAD_rate
                teamHomeResultsCumulW_d = teamHomeResultsW_s.shift(1).cumsum().to_dict()
                teamAwayResultsCumulW_d = teamAwayResultsW_s.shift(1).cumsum().to_dict()
                if league['can_draw']:
                    teamHomeResultsCumulD_d = teamHomeResultsD_s.shift(1).cumsum().to_dict()
                    teamAwayResultsCumulD_d = teamAwayResultsD_s.shift(1).cumsum().to_dict()
                E0_data.loc[teamHomeResultsW_s.index,'HTHW_rate'] = E0_data.loc[teamHomeResultsW_s.index,:].index.map(lambda x : teamHomeResultsCumulW_d[x])
                E0_data.loc[teamAwayResultsW_s.index,'ATAW_rate'] = E0_data.loc[teamAwayResultsW_s.index,:].index.map(lambda x : teamAwayResultsCumulW_d[x])
                if league['can_draw']:
                    E0_data.loc[teamHomeResultsW_s.index,'HTHD_rate'] = E0_data.loc[teamHomeResultsW_s.index,:].index.map(lambda x : teamHomeResultsCumulD_d[x])
                    E0_data.loc[teamAwayResultsW_s.index,'ATAD_rate'] = E0_data.loc[teamAwayResultsW_s.index,:].index.map(lambda x : teamAwayResultsCumulD_d[x])
        
    #compute missing features k_XTL_rate thanks to the k_XTW_rate and k_XTD_rate features
    if league['can_draw']:
        E0_data.loc[:,'7_HTL_rate'] = 1 - (E0_data['7_HTW_rate'] + E0_data['7_HTD_rate'])
        E0_data.loc[:,'12_HTL_rate'] = 1 - (E0_data['7_HTW_rate'] + E0_data['7_HTD_rate'])
        E0_data.loc[:,'7_ATL_rate'] = 1 - (E0_data['7_ATW_rate'] + E0_data['7_ATD_rate'])
        E0_data.loc[:,'12_ATL_rate'] = 1 - (E0_data['7_ATW_rate'] + E0_data['7_ATD_rate'])
    else:
        E0_data.loc[:,'7_HTL_rate'] = 1 - E0_data['7_HTW_rate']
        E0_data.loc[:,'12_HTL_rate'] = 1 - E0_data['7_HTW_rate']
        E0_data.loc[:,'7_ATL_rate'] = 1 - E0_data['7_ATW_rate']
        E0_data.loc[:,'12_ATL_rate'] = 1 - E0_data['7_ATW_rate']

    #compute missing HTL_rate, ATL_rate with features with the wins and draws features
    E0_data.loc[:,'HTW_rate'] = np.where(E0_data['HomeTeamDay'] != 0, E0_data['HTW_rate'] / E0_data['HomeTeamDay'], 0)
    E0_data.loc[:,'ATW_rate'] = np.where(E0_data['AwayTeamDay'] != 0, E0_data['ATW_rate'] / E0_data['AwayTeamDay'], 0)
    if league['can_draw']:
        E0_data.loc[:,'HTD_rate'] = np.where(E0_data['HomeTeamDay'] != 0, E0_data['HTD_rate'] / E0_data['HomeTeamDay'], 0)
        E0_data.loc[:,'ATD_rate'] = np.where(E0_data['AwayTeamDay'] != 0, E0_data['ATD_rate'] / E0_data['AwayTeamDay'], 0)
        E0_data.loc[:,'HTL_rate'] = 1 - (E0_data['HTW_rate'] + E0_data['HTD_rate'])
        E0_data.loc[:,'ATL_rate'] = 1 - (E0_data['ATW_rate'] + E0_data['ATD_rate'])
    else:
        E0_data.loc[:,'HTL_rate'] = 1 - E0_data['HTW_rate']
        E0_data.loc[:,'ATL_rate'] = 1 - E0_data['ATW_rate']

    #we finish to compute HTHW_rate, ..., ATAD_rate features and compute corresponding loss features
    E0_data.loc[:,'HTHW_rate'] = np.where(E0_data['HomeTeamHomeDay'] != 0, E0_data['HTHW_rate']/E0_data['HomeTeamHomeDay'], 0)
    E0_data.loc[:,'ATAW_rate'] = np.where(E0_data['AwayTeamAwayDay'] != 0, E0_data['ATAW_rate']/E0_data['AwayTeamAwayDay'], 0)
    if league['can_draw']:
        E0_data.loc[:,'HTHD_rate'] = np.where(E0_data['HomeTeamHomeDay'] != 0, E0_data['HTHD_rate']/E0_data['HomeTeamHomeDay'], 0)
        E0_data.loc[:,'ATAD_rate'] = np.where(E0_data['AwayTeamAwayDay'] != 0, E0_data['ATAD_rate']/E0_data['AwayTeamAwayDay'], 0)
        E0_data.loc[:,'HTHL_rate'] = 1 - (E0_data['HTHW_rate'] + E0_data['HTHD_rate'])
        E0_data.loc[:,'ATAL_rate'] = 1 - (E0_data['ATAW_rate'] + E0_data['ATAD_rate'])
    else:
        E0_data.loc[:,'HTHL_rate'] = 1 - E0_data['HTHW_rate']
        E0_data.loc[:,'ATAL_rate'] = 1 - E0_data['ATAW_rate']

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
            elo_df.loc[(Hteam, date), 'Elo_Score'] = prev_ELO_score_d.get(Hteam, 0.0)
            elo_df.loc[(Ateam, date), 'Elo_Score'] = prev_ELO_score_d.get(Ateam, 0.0)

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

    # Merge Home Team Elo scores into E0_data
    E0_data = E0_data.merge(
        elo_df_reset.rename(columns={'Team': 'HomeTeam', 'Elo_Score': 'HTeamEloScore', 'gamma': 'gammaH'}),
        how='left',
        left_on=['HomeTeam', 'Date'],
        right_on=['HomeTeam', 'Date']
    )

    # Merge Away Team Elo scores into E0_data
    E0_data = E0_data.merge(
        elo_df_reset.rename(columns={'Team': 'AwayTeam', 'Elo_Score': 'ATeamEloScore', 'gamma': 'gammaA'}),
        how='left',
        left_on=['AwayTeam', 'Date'],
        right_on=['AwayTeam', 'Date']
    )

    for team in teams:
        homeMatchDates_s = E0_data[E0_data['HomeTeam'] == team]['Date']
        awayMatchDates_s = E0_data[E0_data['AwayTeam'] == team]['Date']
        matchDates_s = pd.concat([homeMatchDates_s, awayMatchDates_s]).sort_index()
        lastMatchDates_s = matchDates_s.shift(1)
        matchDates = matchDates_s.values
            
        E0_data.loc[E0_data['HomeTeam'] == team, 'HTLastMatchDate'] = E0_data.loc[E0_data['HomeTeam'] == team].index.map(lambda x : lastMatchDates_s[x])
        E0_data.loc[E0_data['AwayTeam'] == team, 'ATLastMatchDate'] = E0_data.loc[E0_data['AwayTeam'] == team].index.map(lambda x : lastMatchDates_s[x])
        
    def HTdaysBetweenDates(row):
        if not (pd.isnull(row['HTLastMatchDate'])):
            currDate = pd.to_datetime(row['Date'])
            prevDate = pd.to_datetime(row['HTLastMatchDate'])
            ndays = (currDate - prevDate).days 
            if ndays < 20:
                return ndays
            else: 
                return 0
        else:
            return 0 
        
    def ATdaysBetweenDates(row):
        if not (pd.isnull(row['ATLastMatchDate'])):
            currDate = pd.to_datetime(row['Date'])
            prevDate = pd.to_datetime(row['ATLastMatchDate'])
            ndays = (currDate - prevDate).days
            if ndays < 20:
                    return ndays
            else: 
                return 0
        else:
            return 0
        
    E0_data.loc[:, 'HTdaysSinceLastMatch'] = E0_data.apply(HTdaysBetweenDates, axis=1)
    E0_data.loc[:, 'ATdaysSinceLastMatch'] = E0_data.apply(ATdaysBetweenDates, axis=1)
    E0_data.loc[:,'DaysSinceLastMatchRate'] = np.where(E0_data['ATdaysSinceLastMatch'] != 0,E0_data['HTdaysSinceLastMatch']/E0_data['ATdaysSinceLastMatch'], 0)

    E0_data['1_last_HTR_isW'] = E0_data['1_last_HTR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['1_last_HTR_isL'] = E0_data['1_last_HTR'].map(lambda x : 1 if x == 'L' else 0) 
    E0_data['2_last_HTR_isW'] = E0_data['2_last_HTR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['2_last_HTR_isL'] = E0_data['2_last_HTR'].map(lambda x : 1 if x == 'L' else 0) 
    E0_data['3_last_HTR_isW'] = E0_data['3_last_HTR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['3_last_HTR_isL'] = E0_data['3_last_HTR'].map(lambda x : 1 if x == 'L' else 0) 

    E0_data['1_last_ATR_isW'] = E0_data['1_last_ATR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['1_last_ATR_isL'] = E0_data['1_last_ATR'].map(lambda x : 1 if x == 'L' else 0) 
    E0_data['2_last_ATR_isW'] = E0_data['2_last_ATR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['2_last_ATR_isL'] = E0_data['2_last_ATR'].map(lambda x : 1 if x == 'L' else 0) 
    E0_data['3_last_ATR_isW'] = E0_data['3_last_ATR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['3_last_ATR_isL'] = E0_data['3_last_ATR'].map(lambda x : 1 if x == 'L' else 0) 

    E0_data['1_last_HTHR_isW'] = E0_data['1_last_HTHR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['1_last_HTHR_isL'] = E0_data['1_last_HTHR'].map(lambda x : 1 if x == 'L' else 0) 
    E0_data['2_last_HTHR_isW'] = E0_data['2_last_HTHR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['2_last_HTHR_isL'] = E0_data['2_last_HTHR'].map(lambda x : 1 if x == 'L' else 0)

    E0_data['1_last_ATAR_isW'] = E0_data['1_last_ATAR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['1_last_ATAR_isL'] = E0_data['1_last_ATAR'].map(lambda x : 1 if x == 'L' else 0) 
    E0_data['2_last_ATAR_isW'] = E0_data['2_last_ATAR'].map(lambda x : 1 if x == 'W' else 0)
    E0_data['2_last_ATAR_isL'] = E0_data['2_last_ATAR'].map(lambda x : 1 if x == 'L' else 0)

    model_dir = f'./match_infos/{league["folder_name"]}'
    # Create the model directory if it does not exist
    os.makedirs(model_dir, exist_ok=True)
    
    E0_data.to_csv(f'{model_dir}/preprocess_data.csv', index=False, encoding='utf-8')

    #Home wins, Away wins and draws rates variations over seasons
    HW_rates = []
    AW_rates = []
    D_rates = []

    for season in seasons:
        season_data = E0_data[E0_data['Season'] == season]
        total_matches_nb = len(season_data.index)
        HW_rate = float(len(season_data[season_data['FTR'] == 'H'].index))/float(total_matches_nb)
        AW_rate = float(len(season_data[season_data['FTR'] == 'A'].index))/float(total_matches_nb)
        HW_rates.append(HW_rate)
        AW_rates.append(AW_rate)
        if league['can_draw']:
            D_rate = float(len(season_data[season_data['FTR'] == 'D'].index))/float(total_matches_nb)
            D_rates.append(D_rate)

    plt.figure()
    plt.plot(seasons, HW_rates, label="Home wins rate")
    plt.plot(seasons, AW_rates, label="Away wins rate")
    if league['can_draw']:
        plt.plot(seasons, D_rates, label="Draw rates")
    plt.legend()
    plt.xticks([int(season) for season in seasons], seasons)
    plt.xlabel("Seasons")
    plt.ylabel("Home wins, draws and losses rates")
    plt.title("Home wins, draws and losses rates evolution over seasons")
    plt.show()