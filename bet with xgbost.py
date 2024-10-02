import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import xgboost as xgb

# Load Play-by-Play Data for historical seasons
play_by_play_2019 = pd.read_csv('play_by_play_2019.csv')
play_by_play_2020 = pd.read_csv('play_by_play_2020.csv')
play_by_play_2021 = pd.read_csv('play_by_play_2021.csv')
play_by_play_2023 = pd.read_csv('play_by_play_2023.csv')
play_by_play_2024 = pd.read_csv('play_by_play_2024.csv')

# Combine all play-by-play data into one DataFrame
all_play_by_play_data = pd.concat([play_by_play_2019, play_by_play_2020, play_by_play_2021, play_by_play_2023, play_by_play_2024])

# Load Advanced Stats for the 2024 season
advstats_rush = pd.read_csv('advstats_season_rush.csv')
advstats_rec = pd.read_csv('advstats_season_rec.csv')
advstats_pass = pd.read_csv('advstats_season_pass.csv')
advstats_def = pd.read_csv('advstats_season_def.csv')

# Filter advanced stats for the 2024 season only
advstats_rush_2024 = advstats_rush[advstats_rush['season'] == 2024]
advstats_rec_2024 = advstats_rec[advstats_rec['season'] == 2024]
advstats_pass_2024 = advstats_pass[advstats_pass['season'] == 2024]
advstats_def_2024 = advstats_def[advstats_def['season'] == 2024]

# Function to extract team advanced stats for the 2024 season, with per-game averages and scaling applied
def extract_team_advanced_stats(team, advstats_rush, advstats_rec, advstats_pass, advstats_def, scaling_factor=0.05):
    rush_stats = advstats_rush[advstats_rush['tm'] == team]
    rec_stats = advstats_rec[advstats_rec['tm'] == team]
    pass_stats = advstats_pass[advstats_pass['team'] == team]
    def_stats = advstats_def[advstats_def['tm'] == team]

    total_rushing_yards = rush_stats['yds'].sum() * scaling_factor
    total_rushing_tds = rush_stats['td'].sum() * scaling_factor
    total_yac_rush = rush_stats['yac'].sum() * scaling_factor
    total_broken_tackles_rush = rush_stats['brk_tkl'].sum() * scaling_factor

    total_receiving_yards = rec_stats['yds'].sum() * scaling_factor
    total_receiving_tds = rec_stats['td'].sum() * scaling_factor
    total_yac_rec = rec_stats['yac'].sum() * scaling_factor
    total_broken_tackles_rec = rec_stats['brk_tkl'].sum() * scaling_factor

    total_pass_attempts = pass_stats['pass_attempts'].sum() * scaling_factor

    total_sacks = def_stats['sk'].sum() * scaling_factor
    total_pressures = def_stats['prss'].sum() * scaling_factor
    total_hurries = def_stats['hrry'].sum() * scaling_factor
    total_interceptions = def_stats['int'].sum() * scaling_factor
    total_blitzes = def_stats['bltz'].sum() * scaling_factor

    games_played = rush_stats['g'].nunique()

    if games_played > 0:
        avg_rushing_yards_per_game = total_rushing_yards / games_played
        avg_rushing_tds_per_game = total_rushing_tds / games_played
        avg_yac_rush_per_game = total_yac_rush / games_played
        avg_broken_tackles_rush_per_game = total_broken_tackles_rush / games_played

        avg_receiving_yards_per_game = total_receiving_yards / games_played
        avg_receiving_tds_per_game = total_receiving_tds / games_played
        avg_yac_rec_per_game = total_yac_rec / games_played
        avg_broken_tackles_rec_per_game = total_broken_tackles_rec / games_played

        avg_pass_attempts_per_game = total_pass_attempts / games_played

        avg_sacks_per_game = total_sacks / games_played
        avg_pressures_per_game = total_pressures / games_played
        avg_hurries_per_game = total_hurries / games_played
        avg_interceptions_per_game = total_interceptions / games_played
        avg_blitzes_per_game = total_blitzes / games_played
    else:
        avg_rushing_yards_per_game = avg_rushing_tds_per_game = avg_yac_rush_per_game = avg_broken_tackles_rush_per_game = 0
        avg_receiving_yards_per_game = avg_receiving_tds_per_game = avg_yac_rec_per_game = avg_broken_tackles_rec_per_game = 0
        avg_pass_attempts_per_game = 0
        avg_sacks_per_game = avg_pressures_per_game = avg_hurries_per_game = avg_interceptions_per_game = avg_blitzes_per_game = 0

    advanced_stats = {
        'rushing_yards': avg_rushing_yards_per_game,
        'rushing_tds': avg_rushing_tds_per_game,
        'yac_rush': avg_yac_rush_per_game,
        'broken_tackles_rush': avg_broken_tackles_rush_per_game,
        'receiving_yards': avg_receiving_yards_per_game,
        'receiving_tds': avg_receiving_tds_per_game,
        'yac_rec': avg_yac_rec_per_game,
        'broken_tackles_rec': avg_broken_tackles_rec_per_game,
        'pass_attempts': avg_pass_attempts_per_game,
        'sacks': avg_sacks_per_game,
        'pressures': avg_pressures_per_game,
        'hurries': avg_hurries_per_game,
        'interceptions': avg_interceptions_per_game,
        'blitzes': avg_blitzes_per_game
    }

    return advanced_stats

# Function to extract team metrics with features
def extract_team_metrics_with_features(play_by_play_data, team1, team2):
    team1_as_home = play_by_play_data[(play_by_play_data['home_team'] == team1) & (play_by_play_data['away_team'] == team2)]
    team1_as_away = play_by_play_data[(play_by_play_data['away_team'] == team1) & (play_by_play_data['home_team'] == team2)]
    past_matchups = pd.concat([team1_as_home, team1_as_away])

    if past_matchups.empty:
        print('no past')
        team1_feature_avgs = get_team_avg_play_by_play_2024(team1, play_by_play_2024)
        team2_feature_avgs = get_team_avg_play_by_play_2024(team2, play_by_play_2024)

        avg_team1_points = play_by_play_2024[play_by_play_2024['home_team'] == team1]['home_score'].mean() + \
                           play_by_play_2024[play_by_play_2024['away_team'] == team1]['away_score'].mean()
        avg_team2_points = play_by_play_2024[play_by_play_2024['home_team'] == team2]['home_score'].mean() + \
                           play_by_play_2024[play_by_play_2024['away_team'] == team2]['away_score'].mean()

        return avg_team1_points, avg_team2_points, team1_feature_avgs, team2_feature_avgs, None

    final_plays = past_matchups.loc[past_matchups.groupby('game_id')['play_id'].idxmax()]

    feature_columns = ['yards_gained', 'air_yards', 'yards_after_catch', 'run_location', 'run_gap', 
                       'kick_distance', 'field_goal_result', 'two_point_conv_result', 
                       'total_home_epa', 'total_away_epa', 'air_epa', 'yac_epa', 'wp']
    
    past_matchups[feature_columns] = past_matchups[feature_columns].apply(pd.to_numeric, errors='coerce')

    team1_feature_avgs = past_matchups[past_matchups['posteam'] == team1][feature_columns].mean()
    team2_feature_avgs = past_matchups[past_matchups['posteam'] == team2][feature_columns].mean()

    historical_matchups = final_plays[['game_id', 'game_date', 'posteam', 'defteam', 'posteam_score_post', 'defteam_score_post']].drop_duplicates()
    filename = f"{team1}_vs_{team2}_historical_matchups.csv"
    historical_matchups.to_csv(filename, index=False)

    team1_points = final_plays.loc[final_plays['posteam'] == team1,    'posteam_score_post'].sum() + final_plays.loc[final_plays['defteam'] == team1, 'defteam_score_post'].sum()
    team2_points = final_plays.loc[final_plays['posteam'] == team2, 'posteam_score_post'].sum() + final_plays.loc[final_plays['defteam'] == team2, 'defteam_score_post'].sum()

    games_played = final_plays['game_id'].nunique()
    avg_team1_points = team1_points / games_played if games_played > 0 else 0
    avg_team2_points = team2_points / games_played if games_played > 0 else 0

    return avg_team1_points, avg_team2_points, team1_feature_avgs, team2_feature_avgs, past_matchups

# Function to get team averages for the 2024 season play-by-play data
def get_team_avg_play_by_play_2024(team, play_by_play_2024):
    feature_columns = ['yards_gained', 'air_yards', 'yards_after_catch', 'run_location', 'run_gap', 
                       'kick_distance', 'field_goal_result', 'two_point_conv_result', 
                       'total_home_epa', 'total_away_epa', 'air_epa', 'yac_epa', 'wp']

    team_as_home = play_by_play_2024[play_by_play_2024['home_team'] == team]
    team_as_away = play_by_play_2024[play_by_play_2024['away_team'] == team]
    team_play_by_play = pd.concat([team_as_home, team_as_away])

    team_feature_avgs = team_play_by_play[feature_columns].mean() if not team_play_by_play.empty else pd.Series([0] * len(feature_columns), index=feature_columns)
    
    return team_feature_avgs

# Simulate game outcomes with advanced stats
def simulate_game_outcomes_with_advanced_stats(team1, team2, team1_avg_points, team2_avg_points, 
                                               team1_feature_avgs, team2_feature_avgs, team1_advanced_stats, 
                                               team2_advanced_stats, home_field_team, past_matchups, num_simulations=1000):
    base_score_factor = 0.2
    variance_factor = 10.0  
    home_field_advantage = 1  
    historical_weight = 0.71
    feature_weight = 0.08 
    offensive_stat_weight = 0.1
    defensive_stat_weight = 2.4  

    if past_matchups is None or len(past_matchups['game_id'].unique()) < 2:
        variance_factor = 12.0  
        historical_weight = 0.5  
        feature_weight = 0.2  
        offensive_stat_weight = 0.2  
        defensive_stat_weight = 1.8  

    team1_scores = []
    team2_scores = []

    for _ in range(num_simulations):
        team1_score_factors = (team1_feature_avgs.mean() * feature_weight)
        team2_score_factors = (team2_feature_avgs.mean() * feature_weight)

        random_factor_team1 = np.random.randn() * variance_factor
        random_factor_team2 = np.random.randn() * variance_factor

        team1_offense_boost = (team1_advanced_stats['rushing_yards'] * offensive_stat_weight + 
                              team1_advanced_stats['receiving_yards'] * offensive_stat_weight + 
                              team1_advanced_stats['pass_attempts'] * 0.01)
        team2_offense_boost = (team2_advanced_stats['rushing_yards'] * offensive_stat_weight + 
                              team2_advanced_stats['receiving_yards'] * offensive_stat_weight + 
                              team2_advanced_stats['pass_attempts'] * 0.01)

        team1_defense_impact = (team2_advanced_stats['sacks'] * defensive_stat_weight + 
                                team2_advanced_stats['pressures'] * defensive_stat_weight + 
                                team2_advanced_stats['hurries'] * defensive_stat_weight +
                                team2_advanced_stats['interceptions'] * defensive_stat_weight + 
                                team2_advanced_stats['blitzes'] * defensive_stat_weight) * -0.1
        team2_defense_impact = (team1_advanced_stats['sacks'] * defensive_stat_weight + 
                                team1_advanced_stats['pressures'] * defensive_stat_weight + 
                                team1_advanced_stats['hurries'] * defensive_stat_weight +
                                team1_advanced_stats['interceptions'] * defensive_stat_weight + 
                                team1_advanced_stats['blitzes'] * defensive_stat_weight) * -0.1

        historical_weight_adjusted = historical_weight + np.random.uniform(-0.1, 0.1)

        if home_field_team == team1:
            team1_score = max(0, base_score_factor + team1_score_factors + (team1_avg_points * historical_weight_adjusted) + 
                              team1_offense_boost + team1_defense_impact + random_factor_team1 + home_field_advantage)
            team2_score = max(0, base_score_factor + team2_score_factors + (team2_avg_points * historical_weight_adjusted) + 
                              team2_offense_boost + team2_defense_impact + random_factor_team2)
        else:
            team1_score = max(0, base_score_factor + team1_score_factors + (team1_avg_points * historical_weight_adjusted) + 
                              team1_offense_boost + team1_defense_impact + random_factor_team1)
            team2_score = max(0, base_score_factor + team2_score_factors + (team2_avg_points * historical_weight_adjusted) + 
                              team2_offense_boost + team2_defense_impact + random_factor_team2 + home_field_advantage)

        team1_scores.append(team1_score)
        team2_scores.append(team2_score)

    avg_team1_score = round(np.mean(team1_scores))
    avg_team2_score = round(np.mean(team2_scores))
    
    return avg_team1_score, avg_team2_score, team1_scores, team2_scores

# Train separate XGBoost models for each team
def train_separate_xgboost_models(team1_feature_avgs, team2_feature_avgs, team1_scores, team2_scores):
    features_team1 = team1_feature_avgs.values.reshape(1, -1)
    features_team2 = team2_feature_avgs.values.reshape(1, -1)
    
    target_team1 = pd.Series([team1_scores[0]])
    target_team2 = pd.Series([team2_scores[0]])

    model_team1 = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)
    model_team2 = xgb.XGBRegressor(n_estimators=100, max_depth=3, learning_rate=0.1)

    model_team1.fit(features_team1, target_team1)
    model_team2.fit(features_team2, target_team2)

    prediction_team1 = model_team1.predict(features_team1)
    prediction_team2 = model_team2.predict(features_team2)

    return prediction_team1, prediction_team2

# Calculate the Mean Squared Error (MSE)
def calculate_mse(avg_actual_team1, avg_actual_team2, avg_predicted_team1, avg_predicted_team2):
    mse_team1 = mean_squared_error([avg_actual_team1], [avg_predicted_team1])
    mse_team2 = mean_squared_error([avg_actual_team2], [avg_predicted_team2])
    total_mse = (mse_team1 + mse_team2) / 2
    return total_mse

# Prompt the user for input
team1 = input("Enter the first team (e.g., Cowboys): ")
team2 = input("Enter the second team (e.g., Giants): ")
home_field_team = input(f"Which team has home-field advantage, {team1} or {team2}? ")

# Get average points from historical data or fall back to current stats, along with feature averages
avg_team1_points, avg_team2_points, team1_feature_avgs, team2_feature_avgs, past_matchups = extract_team_metrics_with_features(
    all_play_by_play_data, team1, team2
)

# Get advanced stats for both teams (only for the 2024 season and game averages)
team1_advanced_stats = extract_team_advanced_stats(team1, advstats_rush_2024, advstats_rec_2024, advstats_pass_2024, advstats_def_2024, scaling_factor=0.1)
team2_advanced_stats = extract_team_advanced_stats(team2, advstats_rush_2024, advstats_rec_2024, advstats_pass_2024, advstats_def_2024, scaling_factor=0.1)

# Simulate outcomes based on historical data, performance features, and advanced stats
avg_team1_score, avg_team2_score, team1_scores, team2_scores = simulate_game_outcomes_with_advanced_stats(
    team1, team2, avg_team1_points, avg_team2_points, 
    team1_feature_avgs, team2_feature_avgs, team1_advanced_stats, team2_advanced_stats, home_field_team, past_matchups
)

# Train and predict scores using separate XGBoost models for each team
xgboost_prediction_team1, xgboost_prediction_team2 = train_separate_xgboost_models(
    team1_feature_avgs, team2_feature_avgs, team1_scores, team2_scores
)

# Calculate the Mean Squared Error (MSE) between the actual average scores and predicted average scores
mse = calculate_mse(avg_team1_score, avg_team2_score, xgboost_prediction_team1[0], xgboost_prediction_team2[0])

# Display the results
print(f"\nPredicted Average Score for {team1}: {avg_team1_score}")
print(f"Predicted Average Score for {team2}: {avg_team2_score}")
print(f"XGBoost Predicted Score for {team1}: {xgboost_prediction_team1[0]}")
print(f"XGBoost Predicted Score for {team2}: {xgboost_prediction_team2[0]}")
print(f"Mean Squared Error (MSE) of the model: {mse}")

# Save the individual simulation scores to a CSV file for further analysis
simulation_results = pd.DataFrame({
    'Team 1': [team1] * len(team1_scores),
    'Team 2': [team2] * len(team2_scores),
    'Team 1 Simulated Score': team1_scores,
    'Team 2 Simulated Score': team2_scores,
    'XGBoost Predicted Score Team 1': [xgboost_prediction_team1[0]] * len(team1_scores),
    'XGBoost Predicted Score Team 2': [xgboost_prediction_team2[0]] * len(team2_scores)
})

simulation_filename = f"{team1}_vs_{team2}_simulation_results_with_xgboost.csv"
simulation_results.to_csv(simulation_filename, index=False)
print(f"Simulation results saved to {simulation_filename}")

