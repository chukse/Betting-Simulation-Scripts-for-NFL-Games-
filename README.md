# Betting-Simulation-Scripts-for-NFL-Games-

# Betting Analysis with XGBoost

This repository contains Python scripts designed to analyze NFL data and predict betting outcomes using machine learning models, specifically XGBoost. The scripts process advanced NFL statistics and help generate insights for sports betting, focusing on game predictions and betting decisions.

## Table of Contents

- [Files](#files)
- [Datasets](#datasets)
- [Requirements](#requirements)
- [Usage](#usage)
- [Model Overview](#model-overview)
- [Future Enhancements](#future-enhancements)
- [License](#license)
- [Contact](#contact)

## Files

- **`bet_with_xgboost.py`**: This is the main script for training an XGBoost model using NFL team statistics. It builds a predictive model for betting analysis by processing advanced statistics.
  
- **`bet_without_roster_ratings.py`**: A variation of the main script that excludes player roster ratings from the dataset. This script focuses only on team-based statistics to make predictions.

- **`xgboost_script_update.py`**: An updated version of the betting script, with optimized code and additional features for improved performance and accuracy. It introduces new data preprocessing and model evaluation techniques.

## Datasets

The following CSV files provide the advanced statistics for each NFL season. They are used by the scripts to train and evaluate the betting models:

- **`advstats_season_def.csv`**: Contains defensive statistics for the teams during the season.
  
- **`advstats_season_pass.csv`**: Holds passing statistics for the season.

- **`advstats_season_rec.csv`**: Includes receiving statistics for the season.

- **`advstats_season_rush.csv`**: Rushing statistics for the season are stored here.

## Requirements

To run these scripts, you will need to install the following Python libraries. You can install them by running the following command:

```bash
pip install xgboost pandas numpy scikit-learn
