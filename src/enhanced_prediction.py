"""
Enhanced NFL Game Prediction Interface
Includes team name validation, better error handling, and improved predictions
"""

import pandas as pd
import numpy as np
import joblib
import argparse
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from enhanced_feature_engineering import EnhancedNFLFeatureEngineer
from team_names import normalize_team_name, validate_team_name, get_all_team_names
from advanced_features import AdvancedNFLFeatures


class EnhancedNFLGamePredictor:
    """Enhanced predictor with team validation and robust features"""

    def __init__(self, model_path='models/best_enhanced_model.joblib', games_path='data/raw_games.csv'):
        """
        Initialize enhanced predictor

        Args:
            model_path: Path to trained model file
            games_path: Path to historical games data
        """
        self.model_path = model_path
        self.games_path = games_path
        self.model = None
        self.engineer = None
        self.games_df = None

        # Try enhanced model first, fallback to basic
        if not os.path.exists(self.model_path):
            print(f"Enhanced model not found at {self.model_path}")
            fallback_path = 'models/best_model.joblib'
            if os.path.exists(fallback_path):
                print(f"Using fallback model: {fallback_path}")
                self.model_path = fallback_path
            else:
                raise FileNotFoundError("No trained model found. Please train a model first.")

        self.load_model()
        self.load_data()

    def load_model(self):
        """Load trained model from disk"""
        print(f"Loading model from {self.model_path}...")
        self.model = joblib.load(self.model_path)
        print(f"Model loaded: {self.model.name}")

    def load_data(self):
        """Load historical game data"""
        if not os.path.exists(self.games_path):
            raise FileNotFoundError(f"Game data not found at {self.games_path}")

        print(f"Loading historical data...")
        self.games_df = pd.read_csv(self.games_path)
        self.engineer = EnhancedNFLFeatureEngineer(self.games_df)

        # Pre-calculate ratings (this takes time but is necessary)
        print("Initializing feature engineering...")
        self.engineer.base_engineer.calculate_elo_ratings()
        self.engineer.advanced_engineer = AdvancedNFLFeatures(
            self.games_df,
            self.engineer.base_engineer.elo_ratings
        )
        self.engineer.mov_elo_ratings = self.engineer.advanced_engineer.calculate_margin_of_victory_elo()

        print(f"Ready to make predictions!")

    def validate_teams(self, home_team, away_team):
        """
        Validate and normalize team names

        Args:
            home_team: Home team name (can be abbreviation)
            away_team: Away team name (can be abbreviation)

        Returns:
            Tuple of (normalized_home, normalized_away, error_message)
        """
        # Validate home team
        is_valid_home, norm_home, suggestions_home = validate_team_name(home_team)
        if not is_valid_home:
            # Try normalization
            norm_home = normalize_team_name(home_team)
            if norm_home:
                is_valid_home = True

        # Validate away team
        is_valid_away, norm_away, suggestions_away = validate_team_name(away_team)
        if not is_valid_away:
            norm_away = normalize_team_name(away_team)
            if norm_away:
                is_valid_away = True

        # Check for errors
        errors = []
        if not is_valid_home:
            errors.append(f"Invalid home team: '{home_team}'. Suggestions: {suggestions_home}")
        if not is_valid_away:
            errors.append(f"Invalid away team: '{away_team}'. Suggestions: {suggestions_away}")

        if errors:
            return None, None, "\n".join(errors)

        return norm_home, norm_away, None

    def predict_game(self, home_team, away_team, game_date=None, week=None, season=None):
        """
        Predict the outcome of a single game

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Date of the game
            week: Week number
            season: Season year

        Returns:
            Dictionary with prediction results or error
        """
        # Validate team names
        norm_home, norm_away, error = self.validate_teams(home_team, away_team)
        if error:
            return {
                'error': error,
                'suggestions': 'Valid team names: ' + ', '.join(get_all_team_names())
            }

        # Default values
        if game_date is None:
            game_date = datetime.now()
        elif isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)

        if season is None:
            season = game_date.year

        if week is None:
            week = 1

        # Create dummy game row
        game_row = pd.Series({
            'home_team': norm_home,
            'away_team': norm_away,
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

        try:
            # Generate basic features
            basic_features = self.engineer.base_engineer.create_features_for_game(game_row)

            # Generate advanced features (need to check if advanced_engineer exists)
            advanced_features = {}
            if self.engineer.advanced_engineer is not None:
                advanced_features = self.engineer.advanced_engineer.create_advanced_features(
                    game_row,
                    mov_elo_ratings=self.engineer.mov_elo_ratings
                )

            # Try to get betting features if available
            betting_features = {}
            if hasattr(self.engineer, 'betting_lines') and self.engineer.betting_lines is not None:
                try:
                    betting_features = self.engineer.betting_lines.get_betting_features(
                        norm_home,
                        norm_away,
                        game_date,
                        season
                    )
                except Exception as e:
                    print(f"Warning: Could not fetch betting features: {e}")

            # Ensure betting features have default values if missing
            default_betting_features = {
                'betting_spread': 0.0,
                'betting_total': 47.5,  # NFL average
                'betting_spread_abs': 0.0,
                'betting_is_favorite': 0,
                'betting_favorite_margin': 0.0
            }
            for key, default_value in default_betting_features.items():
                if key not in betting_features:
                    betting_features[key] = default_value

            # Combine features
            all_features = {**basic_features, **advanced_features, **betting_features}
            features_df = pd.DataFrame([all_features])

            # Extract feature columns (handle None or invalid feature_columns)
            if self.model.feature_columns is None or not self.model.feature_columns:
                # Model doesn't have feature_columns - use all available features
                exclude_cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'gameday', 'home_won', 'score_diff']
                valid_feature_columns = [col for col in features_df.columns if col not in exclude_cols]
            else:
                # Filter out None values from feature_columns
                valid_feature_columns = [col for col in self.model.feature_columns if col is not None and col in features_df.columns]

                # If feature_columns has None or missing columns, fallback to all features
                if len(valid_feature_columns) != len(self.model.feature_columns):
                    print(f"Warning: Model expects {len(self.model.feature_columns)} features, but only {len(valid_feature_columns)} are valid")
                    # Use all available features that aren't metadata
                    exclude_cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'gameday', 'home_won', 'score_diff']
                    valid_feature_columns = [col for col in features_df.columns if col not in exclude_cols]

            X = features_df[valid_feature_columns].fillna(0)

            # Make prediction
            prediction = self.model.predict(X)[0]
            probabilities = self.model.predict_proba(X)[0]

            home_win_prob = probabilities[1]
            away_win_prob = probabilities[0]

            # Extract key features for display
            result = {
                'home_team': norm_home,
                'away_team': norm_away,
                'predicted_winner': norm_home if prediction == 1 else norm_away,
                'home_win_probability': home_win_prob,
                'away_win_probability': away_win_prob,
                'confidence': max(home_win_prob, away_win_prob),
                'spread_implied': (home_win_prob - 0.5) * 28,  # Rough conversion to point spread
                'features': {
                    'home_elo': all_features.get('home_elo', 0),
                    'away_elo': all_features.get('away_elo', 0),
                    'home_mov_elo': all_features.get('home_mov_elo', 0),
                    'away_mov_elo': all_features.get('away_mov_elo', 0),
                    'elo_diff': all_features.get('elo_diff', 0),
                    'home_recent_wins': all_features.get('home_recent_wins', 0),
                    'away_recent_wins': all_features.get('away_recent_wins', 0),
                    'home_weighted_win_rate': all_features.get('home_weighted_win_rate', 0.5),
                    'away_weighted_win_rate': all_features.get('away_weighted_win_rate', 0.5),
                    'home_streak': all_features.get('home_streak', 0),
                    'away_streak': all_features.get('away_streak', 0),
                }
            }

            return result

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return {
                'error': f"Prediction error: {str(e) if str(e) else type(e).__name__}",
                'error_details': error_details,
                'home_team': norm_home,
                'away_team': norm_away
            }

    def predict_multiple_games(self, games_list):
        """
        Predict outcomes for multiple games

        Args:
            games_list: List of game dictionaries

        Returns:
            List of prediction results
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
        """Pretty print a prediction result"""
        if 'error' in result:
            print("\n" + "=" * 70)
            print("ERROR")
            print("=" * 70)
            print(f"\n{result['error']}")
            if 'suggestions' in result:
                print(f"\n{result['suggestions']}")
            if 'error_details' in result:
                print(f"\nFull traceback:\n{result['error_details']}")
            print("=" * 70)
            return

        print("\n" + "=" * 70)
        print(f"  {result['away_team']} @ {result['home_team']}")
        print("=" * 70)
        print(f"\n🏆 Predicted Winner: {result['predicted_winner']}")
        print(f"📊 Confidence: {result['confidence']:.1%}")

        if abs(result['spread_implied']) > 0.5:
            fav = result['home_team'] if result['spread_implied'] > 0 else result['away_team']
            spread = abs(result['spread_implied'])
            print(f"📈 Implied Spread: {fav} by {spread:.1f} points")

        print(f"\nWin Probabilities:")
        print(f"  🏠 {result['home_team']}: {result['home_win_probability']:.1%}")
        print(f"  ✈️  {result['away_team']}: {result['away_win_probability']:.1%}")

        print(f"\n📊 Key Metrics:")
        features = result['features']

        if 'home_mov_elo' in features and features['home_mov_elo'] > 0:
            print(f"  MOV Elo: {result['home_team']} ({features['home_mov_elo']:.0f}) vs {result['away_team']} ({features['away_mov_elo']:.0f})")
        else:
            print(f"  Elo: {result['home_team']} ({features['home_elo']:.0f}) vs {result['away_team']} ({features['away_elo']:.0f})")

        if 'home_weighted_win_rate' in features:
            print(f"  Weighted Recent Form: {result['home_team']} ({features['home_weighted_win_rate']:.1%}) vs {result['away_team']} ({features['away_weighted_win_rate']:.1%})")

        if 'home_streak' in features:
            home_streak = int(features['home_streak'])
            away_streak = int(features['away_streak'])
            if home_streak != 0:
                streak_type = "W" if home_streak > 0 else "L"
                print(f"  {result['home_team']} on {abs(home_streak)}-game {streak_type} streak")
            if away_streak != 0:
                streak_type = "W" if away_streak > 0 else "L"
                print(f"  {result['away_team']} on {abs(away_streak)}-game {streak_type} streak")

        print("=" * 70)


def main():
    """Main execution with CLI"""
    parser = argparse.ArgumentParser(
        description='Enhanced NFL game prediction with team name validation',
        epilog='Example: python src/enhanced_prediction.py --home "KC" --away "Bills"'
    )
    parser.add_argument('--home', type=str, help='Home team name or abbreviation')
    parser.add_argument('--away', type=str, help='Away team name or abbreviation')
    parser.add_argument('--date', type=str, help='Game date (YYYY-MM-DD)', default=None)
    parser.add_argument('--week', type=int, help='Week number', default=None)
    parser.add_argument('--season', type=int, help='Season year', default=None)
    parser.add_argument('--list-teams', action='store_true', help='List all valid team names')

    args = parser.parse_args()

    # List teams if requested
    if args.list_teams:
        teams = get_all_team_names()
        print("\nValid NFL Team Names:")
        print("=" * 70)
        for i, team in enumerate(teams, 1):
            print(f"  {i:2d}. {team}")
        print("=" * 70)
        print(f"\nTotal: {len(teams)} teams")
        return

    # Initialize predictor
    try:
        predictor = EnhancedNFLGamePredictor()
    except Exception as e:
        print(f"Error initializing predictor: {e}")
        return

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
        print("TIP: Use team abbreviations like 'KC', 'BUF', 'SF', etc.")
        print("     Or use --list-teams to see all valid team names")

        example_games = [
            {'home_team': 'KC', 'away_team': 'BUF'},
            {'home_team': 'SF', 'away_team': 'DAL'},
            {'home_team': 'PHI', 'away_team': 'NYG'}
        ]

        results = predictor.predict_multiple_games(example_games)

        for result in results:
            predictor.print_prediction(result)


if __name__ == "__main__":
    # Fix import
    from advanced_features import AdvancedNFLFeatures
    main()
