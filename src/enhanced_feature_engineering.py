"""
Enhanced Feature Engineering Pipeline
Combines basic and advanced features for maximum prediction accuracy
"""

import pandas as pd
import numpy as np
from feature_engineering import NFLFeatureEngineer
from advanced_features import AdvancedNFLFeatures
import warnings
warnings.filterwarnings('ignore')


class EnhancedNFLFeatureEngineer:
    """
    Complete feature engineering pipeline with all features
    """

    def __init__(self, games_df):
        """
        Initialize enhanced feature engineer

        Args:
            games_df: DataFrame with historical game data
        """
        self.games_df = games_df.copy()
        self.games_df['gameday'] = pd.to_datetime(self.games_df['gameday'])
        self.games_df = self.games_df.sort_values('gameday').reset_index(drop=True)

        # Initialize base engineer
        self.base_engineer = NFLFeatureEngineer(games_df)

        # Will be initialized after Elo calculation
        self.advanced_engineer = None
        self.mov_elo_ratings = None

    def create_complete_feature_dataset(self):
        """
        Create comprehensive feature dataset with all features

        Returns:
            DataFrame with all features
        """
        print("=" * 70)
        print("ENHANCED FEATURE ENGINEERING")
        print("=" * 70)

        # Step 1: Calculate base Elo ratings
        print("\nStep 1: Calculating base Elo ratings...")
        elo_ratings = self.base_engineer.calculate_elo_ratings()

        # Step 2: Initialize advanced engineer with Elo ratings
        print("\nStep 2: Initializing advanced feature engineering...")
        self.advanced_engineer = AdvancedNFLFeatures(self.games_df, elo_ratings)

        # Step 3: Calculate MOV-adjusted Elo ratings
        print("\nStep 3: Calculating margin-of-victory Elo ratings...")
        self.mov_elo_ratings = self.advanced_engineer.calculate_margin_of_victory_elo()

        # Step 4: Generate all features for each game
        print("\nStep 4: Generating comprehensive features...")
        features_list = []

        total_games = len(self.games_df)
        for idx, row in self.games_df.iterrows():
            if idx % 500 == 0:
                print(f"  Processing game {idx}/{total_games} ({100*idx/total_games:.1f}%)")

            # Get basic features
            basic_features = self.base_engineer.create_features_for_game(row)

            # Get advanced features
            advanced_features = self.advanced_engineer.create_advanced_features(
                row,
                mov_elo_ratings=self.mov_elo_ratings
            )

            # Combine all features
            all_features = {**basic_features, **advanced_features}

            features_list.append(all_features)

        features_df = pd.DataFrame(features_list)

        # Remove games without targets (future games)
        features_df = features_df[features_df['home_won'].notna()]

        # Count features
        exclude_cols = ['game_id', 'season', 'week', 'home_team', 'away_team', 'gameday', 'home_won', 'score_diff']
        feature_cols = [col for col in features_df.columns if col not in exclude_cols]

        print(f"\nFeature engineering complete!")
        print(f"  Total games: {len(features_df)}")
        print(f"  Total features: {len(feature_cols)}")
        print(f"  Basic features: 29")
        print(f"  Advanced features: {len(feature_cols) - 29}")
        print("=" * 70)

        return features_df


def main():
    """Main execution function"""
    import sys

    # Load raw game data
    try:
        games_df = pd.read_csv('data/raw_games.csv')
    except FileNotFoundError:
        print("Error: data/raw_games.csv not found.")
        print("Please run: python src/data_collection.py")
        sys.exit(1)

    # Create enhanced engineer
    engineer = EnhancedNFLFeatureEngineer(games_df)

    # Generate comprehensive features
    features_df = engineer.create_complete_feature_dataset()

    # Save features
    features_df.to_csv('data/enhanced_features.csv', index=False)

    print("\n" + "=" * 70)
    print("Enhanced features saved to: data/enhanced_features.csv")
    print("=" * 70)


if __name__ == "__main__":
    main()
