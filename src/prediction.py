"""
NFL Game Prediction Interface
Makes predictions for upcoming NFL games using trained models
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from feature_engineering import NFLFeatureEngineer


class NFLGamePredictor:
    """Predicts outcomes of NFL games"""

    def __init__(self, model_path='models/best_model.joblib', games_path='data/raw_games.csv'):
        """
        Initialize predictor

        Args:
            model_path: Path to trained model file
            games_path: Path to historical games data
        """
        self.model_path = model_path
        self.games_path = games_path
        self.model = None
        self.engineer = None
        self.games_df = None

        self.load_model()
        self.load_data()

    def load_model(self):
        """Load trained model from disk"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model not found at {self.model_path}. Please train a model first.")

        print(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        print(f"Model loaded: {self.model.name}")

    def load_data(self):
        """Load historical game data for feature engineering"""
        if not os.path.exists(self.games_path):
            raise FileNotFoundError(f"Game data not found at {self.games_path}. Please run data collection first.")

        print(f"Loading historical data from {self.games_path}...")
        self.games_df = pd.read_csv(self.games_path)
        self.engineer = NFLFeatureEngineer(self.games_df)
        self.engineer.calculate_elo_ratings()
        print(f"Loaded {len(self.games_df)} historical games")

    def predict_game(self, home_team, away_team, game_date=None, week=None, season=None):
        """
        Predict the outcome of a single game

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Date of the game (optional, defaults to today)
            week: Week number (optional)
            season: Season year (optional, defaults to current year)

        Returns:
            Dictionary with prediction results
        """
        # Default values
        if game_date is None:
            game_date = datetime.now()
        elif isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)

        if season is None:
            season = game_date.year

        if week is None:
            week = 1

        # Create a dummy game row
        game_row = pd.Series({
            'home_team': home_team,
            'away_team': away_team,
            'gameday': game_date,
            'season': season,
            'week': week,
            'game_type': 'REG',
            'div_game': 0,
            'home_score': np.nan,
            'away_score': np.nan,
            'result': np.nan,
            'game_id': 'PREDICTION'
        })

        # Generate features
        features = self.engineer.create_features_for_game(game_row)

        # Create DataFrame with features
        features_df = pd.DataFrame([features])

        # Extract feature columns (exclude metadata and target)
        X = features_df[self.model.feature_columns].fillna(0)

        # Make prediction
        prediction = self.model.predict(X)[0]
        probabilities = self.model.predict_proba(X)[0]

        home_win_prob = probabilities[1]
        away_win_prob = probabilities[0]

        # Create result dictionary
        result = {
            'home_team': home_team,
            'away_team': away_team,
            'predicted_winner': home_team if prediction == 1 else away_team,
            'home_win_probability': home_win_prob,
            'away_win_probability': away_win_prob,
            'confidence': max(home_win_prob, away_win_prob),
            'features': {
                'home_elo': features['home_elo'],
                'away_elo': features['away_elo'],
                'elo_diff': features['elo_diff'],
                'home_recent_wins': features['home_recent_wins'],
                'away_recent_wins': features['away_recent_wins'],
                'home_season_win_pct': features['home_season_win_pct'],
                'away_season_win_pct': features['away_season_win_pct']
            }
        }

        return result

    def predict_multiple_games(self, games_list):
        """
        Predict outcomes for multiple games

        Args:
            games_list: List of dictionaries with 'home_team', 'away_team', and optional 'game_date', 'week', 'season'

        Returns:
            List of prediction result dictionaries
        """
        results = []

        for game in games_list:
            result = self.predict_game(
                home_team=game['home_team'],
                away_team=game['away_team'],
                game_date=game.get('game_date'),
                week=game.get('week'),
                season=game.get('season')
            )
            results.append(result)

        return results

    def print_prediction(self, result):
        """
        Pretty print a prediction result

        Args:
            result: Prediction result dictionary
        """
        print("\n" + "=" * 70)
        print(f"  {result['away_team']} @ {result['home_team']}")
        print("=" * 70)
        print(f"\nPredicted Winner: {result['predicted_winner']}")
        print(f"Confidence: {result['confidence']:.1%}")
        print(f"\nWin Probabilities:")
        print(f"  {result['home_team']}: {result['home_win_probability']:.1%}")
        print(f"  {result['away_team']}: {result['away_win_probability']:.1%}")
        print(f"\nKey Features:")
        print(f"  Elo Ratings: {result['home_team']} ({result['features']['home_elo']:.0f}) vs {result['away_team']} ({result['features']['away_elo']:.0f})")
        print(f"  Recent Form (last 5): {result['home_team']} ({result['features']['home_recent_wins']:.0f}-{5-result['features']['home_recent_wins']:.0f}) vs {result['away_team']} ({result['features']['away_recent_wins']:.0f}-{5-result['features']['away_recent_wins']:.0f})")
        print(f"  Season Win %: {result['home_team']} ({result['features']['home_season_win_pct']:.1%}) vs {result['away_team']} ({result['features']['away_season_win_pct']:.1%})")
        print("=" * 70)


def main():
    """Main execution function with CLI"""
    parser = argparse.ArgumentParser(description='Predict NFL game outcomes')
    parser.add_argument('--home', type=str, help='Home team name')
    parser.add_argument('--away', type=str, help='Away team name')
    parser.add_argument('--date', type=str, help='Game date (YYYY-MM-DD)', default=None)
    parser.add_argument('--week', type=int, help='Week number', default=None)
    parser.add_argument('--season', type=int, help='Season year', default=None)
    parser.add_argument('--model', type=str, help='Path to model file', default='models/best_model.joblib')

    args = parser.parse_args()

    # Initialize predictor
    predictor = NFLGamePredictor(model_path=args.model)

    if args.home and args.away:
        # Single game prediction
        result = predictor.predict_game(
            home_team=args.home,
            away_team=args.away,
            game_date=args.date,
            week=args.week,
            season=args.season
        )
        predictor.print_prediction(result)
    else:
        # Example predictions
        print("\nNo game specified. Running example predictions...\n")

        example_games = [
            {'home_team': 'Kansas City Chiefs', 'away_team': 'Buffalo Bills'},
            {'home_team': 'San Francisco 49ers', 'away_team': 'Dallas Cowboys'},
            {'home_team': 'Philadelphia Eagles', 'away_team': 'New York Giants'}
        ]

        results = predictor.predict_multiple_games(example_games)

        for result in results:
            predictor.print_prediction(result)


if __name__ == "__main__":
    main()
