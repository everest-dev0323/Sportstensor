from sklearn.preprocessing import LabelEncoder
import pandas as pd
import joblib
import json
import os

# Constants
ROOT_DIR = './data'
LEAGUES_PATH = './leagues.json'

def encode_team_names():
    # Load league information
    with open(LEAGUES_PATH, 'r') as file:
        leagues = json.load(file)

    # Process each league
    for league in leagues:
        league_path = os.path.join(ROOT_DIR, league['folder_name'])
        model_dir = f'./models/{league["folder_name"]}'
        
        # Create the model directory if it does not exist
        os.makedirs(model_dir, exist_ok=True)
        
        # Initialize an empty DataFrame for the league
        leagues_df = pd.DataFrame()
        if league['is_active'] == False:
            continue
        
        # Check if league path exists and process files
        if os.path.isdir(league_path):
            for filename in os.listdir(league_path):
                if filename.endswith('.csv'):
                    file_path = os.path.join(league_path, filename)
                    df = pd.read_csv(file_path, encoding='utf-8')
                    leagues_df = pd.concat([leagues_df, df], ignore_index=True)

            # Encode team names
            if not leagues_df.empty:
                le = LabelEncoder()  # Reset LabelEncoder
                teams = pd.concat([leagues_df['HomeTeam'], leagues_df['AwayTeam']]).unique()
                le.fit(teams)
                print(league["league_name"],":", len(teams), 'teams')
                # Save LabelEncoder model
                joblib.dump(le, f'{model_dir}/label_encoder.joblib')
                
    return True

encode_team_names()