"""
NFL Feature Engineering Module
Creates features from raw game data for ML model training
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class NFLFeatureEngineer:
    """Generates features for NFL game prediction"""

    def __init__(self, games_df):
        """
        Initialize feature engineer

        Args:
            games_df: DataFrame with historical game data
        """
        self.games_df = games_df.copy()
        self.games_df['gameday'] = pd.to_datetime(self.games_df['gameday'])
        self.games_df = self.games_df.sort_values('gameday').reset_index(drop=True)

    def calculate_elo_ratings(self, k_factor=20, home_advantage=65, initial_elo=1500):
        """
        Calculate Elo ratings for all teams over time

        Args:
            k_factor: How much ratings change per game
            home_advantage: Points added to home team rating
            initial_elo: Starting rating for all teams

        Returns:
            Dictionary mapping (team, date) to Elo rating
        """
        print("Calculating Elo ratings...")

        elo_ratings = {}
        team_elos = {}

        for idx, row in self.games_df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            game_date = row['gameday']

            # Initialize teams if not seen before
            if home_team not in team_elos:
                team_elos[home_team] = initial_elo
            if away_team not in team_elos:
                team_elos[away_team] = initial_elo

            # Get current Elos
            home_elo = team_elos[home_team]
            away_elo = team_elos[away_team]

            # Store Elo ratings before the game
            elo_ratings[(home_team, game_date, 'home')] = home_elo
            elo_ratings[(away_team, game_date, 'away')] = away_elo

            # Calculate expected scores
            home_expected = 1 / (1 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400))
            away_expected = 1 - home_expected

            # Actual result (1 for win, 0 for loss)
            if pd.notna(row['result']):
                home_score = row['home_score']
                away_score = row['away_score']

                if home_score > away_score:
                    home_actual = 1
                    away_actual = 0
                elif away_score > home_score:
                    home_actual = 0
                    away_actual = 1
                else:  # Tie (rare but possible)
                    home_actual = 0.5
                    away_actual = 0.5

                # Update Elo ratings
                team_elos[home_team] += k_factor * (home_actual - home_expected)
                team_elos[away_team] += k_factor * (away_actual - away_expected)

        self.elo_ratings = elo_ratings
        self.final_elos = team_elos

        return elo_ratings

    def get_recent_form(self, team, game_date, n_games=5):
        """
        Get recent performance metrics for a team

        Args:
            team: Team name
            game_date: Current game date
            n_games: Number of recent games to consider

        Returns:
            Dictionary of recent form metrics
        """
        # Get recent games before this date
        team_games = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (self.games_df['gameday'] < game_date)
        ].tail(n_games)

        if len(team_games) == 0:
            return {
                'recent_wins': 0,
                'recent_losses': 0,
                'recent_points_for': 0,
                'recent_points_against': 0,
                'recent_point_diff': 0,
                'games_played': 0
            }

        wins = 0
        losses = 0
        points_for = 0
        points_against = 0

        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                pf = game['home_score']
                pa = game['away_score']
            else:
                pf = game['away_score']
                pa = game['home_score']

            points_for += pf
            points_against += pa

            if pf > pa:
                wins += 1
            elif pf < pa:
                losses += 1

        return {
            'recent_wins': wins,
            'recent_losses': losses,
            'recent_points_for': points_for / len(team_games),
            'recent_points_against': points_against / len(team_games),
            'recent_point_diff': (points_for - points_against) / len(team_games),
            'games_played': len(team_games)
        }

    def get_season_stats(self, team, season, week):
        """
        Get season-to-date statistics for a team

        Args:
            team: Team name
            season: Season year
            week: Current week

        Returns:
            Dictionary of season statistics
        """
        # Get all games in this season before this week
        season_games = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (self.games_df['season'] == season) &
            (self.games_df['week'] < week)
        ]

        if len(season_games) == 0:
            return {
                'season_wins': 0,
                'season_losses': 0,
                'season_win_pct': 0.5,
                'season_points_for_avg': 0,
                'season_points_against_avg': 0,
                'season_point_diff_avg': 0
            }

        wins = 0
        losses = 0
        points_for = 0
        points_against = 0

        for _, game in season_games.iterrows():
            if game['home_team'] == team:
                pf = game['home_score']
                pa = game['away_score']
            else:
                pf = game['away_score']
                pa = game['home_score']

            points_for += pf
            points_against += pa

            if pf > pa:
                wins += 1
            elif pf < pa:
                losses += 1

        total_games = len(season_games)

        return {
            'season_wins': wins,
            'season_losses': losses,
            'season_win_pct': wins / total_games if total_games > 0 else 0.5,
            'season_points_for_avg': points_for / total_games if total_games > 0 else 0,
            'season_points_against_avg': points_against / total_games if total_games > 0 else 0,
            'season_point_diff_avg': (points_for - points_against) / total_games if total_games > 0 else 0
        }

    def get_head_to_head(self, home_team, away_team, game_date, n_games=5):
        """
        Get head-to-head history between two teams

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Current game date
            n_games: Number of recent H2H games to consider

        Returns:
            Dictionary of H2H metrics
        """
        h2h_games = self.games_df[
            (((self.games_df['home_team'] == home_team) & (self.games_df['away_team'] == away_team)) |
             ((self.games_df['home_team'] == away_team) & (self.games_df['away_team'] == home_team))) &
            (self.games_df['gameday'] < game_date)
        ].tail(n_games)

        if len(h2h_games) == 0:
            return {
                'h2h_home_wins': 0,
                'h2h_away_wins': 0,
                'h2h_games': 0
            }

        home_wins = 0
        away_wins = 0

        for _, game in h2h_games.iterrows():
            if game['home_team'] == home_team:
                if game['home_score'] > game['away_score']:
                    home_wins += 1
                elif game['away_score'] > game['home_score']:
                    away_wins += 1
            else:  # Teams were reversed
                if game['home_score'] > game['away_score']:
                    away_wins += 1
                elif game['away_score'] > game['home_score']:
                    home_wins += 1

        return {
            'h2h_home_wins': home_wins,
            'h2h_away_wins': away_wins,
            'h2h_games': len(h2h_games)
        }

    def get_rest_days(self, team, game_date):
        """
        Calculate days of rest since last game

        Args:
            team: Team name
            game_date: Current game date

        Returns:
            Number of rest days
        """
        last_game = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (self.games_df['gameday'] < game_date)
        ].tail(1)

        if len(last_game) == 0:
            return 7  # Default to 1 week

        last_game_date = last_game.iloc[0]['gameday']
        return (game_date - last_game_date).days

    def create_features_for_game(self, row):
        """
        Create all features for a single game

        Args:
            row: Game row from DataFrame

        Returns:
            Dictionary of features
        """
        home_team = row['home_team']
        away_team = row['away_team']
        game_date = row['gameday']
        season = row['season']
        week = row['week']

        features = {}

        # Elo ratings
        if hasattr(self, 'elo_ratings'):
            # Try exact date lookup first (for historical data)
            home_elo_key = (home_team, game_date, 'home')
            away_elo_key = (away_team, game_date, 'away')

            if home_elo_key in self.elo_ratings and away_elo_key in self.elo_ratings:
                # Historical game - use exact Elo from that date
                features['home_elo'] = self.elo_ratings[home_elo_key]
                features['away_elo'] = self.elo_ratings[away_elo_key]
            elif hasattr(self, 'final_elos'):
                # Future prediction - use latest Elo ratings
                # Try both full name and abbreviation
                from team_names import get_team_abbreviation
                home_abbrev = get_team_abbreviation(home_team)
                away_abbrev = get_team_abbreviation(away_team)

                features['home_elo'] = (
                    self.final_elos.get(home_team) or
                    self.final_elos.get(home_abbrev) or
                    1500
                )
                features['away_elo'] = (
                    self.final_elos.get(away_team) or
                    self.final_elos.get(away_abbrev) or
                    1500
                )
            else:
                # Fallback to defaults
                features['home_elo'] = 1500
                features['away_elo'] = 1500

            features['elo_diff'] = features['home_elo'] - features['away_elo']
        else:
            features['home_elo'] = 1500
            features['away_elo'] = 1500
            features['elo_diff'] = 0

        # Recent form (last 5 games)
        home_form = self.get_recent_form(home_team, game_date, n_games=5)
        away_form = self.get_recent_form(away_team, game_date, n_games=5)

        features['home_recent_wins'] = home_form['recent_wins']
        features['home_recent_losses'] = home_form['recent_losses']
        features['home_recent_ppg'] = home_form['recent_points_for']
        features['home_recent_papg'] = home_form['recent_points_against']
        features['home_recent_diff'] = home_form['recent_point_diff']

        features['away_recent_wins'] = away_form['recent_wins']
        features['away_recent_losses'] = away_form['recent_losses']
        features['away_recent_ppg'] = away_form['recent_points_for']
        features['away_recent_papg'] = away_form['recent_points_against']
        features['away_recent_diff'] = away_form['recent_point_diff']

        # Season stats
        home_season = self.get_season_stats(home_team, season, week)
        away_season = self.get_season_stats(away_team, season, week)

        features['home_season_win_pct'] = home_season['season_win_pct']
        features['home_season_ppg'] = home_season['season_points_for_avg']
        features['home_season_papg'] = home_season['season_points_against_avg']
        features['home_season_diff'] = home_season['season_point_diff_avg']

        features['away_season_win_pct'] = away_season['season_win_pct']
        features['away_season_ppg'] = away_season['season_points_for_avg']
        features['away_season_papg'] = away_season['season_points_against_avg']
        features['away_season_diff'] = away_season['season_point_diff_avg']

        # Head to head
        h2h = self.get_head_to_head(home_team, away_team, game_date, n_games=5)
        features['h2h_home_wins'] = h2h['h2h_home_wins']
        features['h2h_away_wins'] = h2h['h2h_away_wins']
        features['h2h_games'] = h2h['h2h_games']

        # Rest days
        features['home_rest_days'] = self.get_rest_days(home_team, game_date)
        features['away_rest_days'] = self.get_rest_days(away_team, game_date)

        # Situational
        features['is_division_game'] = 1 if row.get('div_game', 0) == 1 else 0
        features['week'] = week
        features['is_playoff'] = 1 if row['game_type'] != 'REG' else 0

        # Target variable
        if pd.notna(row['home_score']) and pd.notna(row['away_score']):
            features['home_won'] = 1 if row['home_score'] > row['away_score'] else 0
            features['score_diff'] = row['home_score'] - row['away_score']
        else:
            features['home_won'] = None
            features['score_diff'] = None

        return features

    def create_feature_dataset(self):
        """
        Create full feature dataset for all games

        Returns:
            DataFrame with features for all games
        """
        print("Creating feature dataset...")

        # Calculate Elo ratings first
        self.calculate_elo_ratings()

        features_list = []

        for idx, row in self.games_df.iterrows():
            if idx % 500 == 0:
                print(f"Processing game {idx}/{len(self.games_df)}")

            features = self.create_features_for_game(row)

            # Add game metadata
            features['game_id'] = row['game_id']
            features['season'] = row['season']
            features['week'] = row['week']
            features['home_team'] = row['home_team']
            features['away_team'] = row['away_team']
            features['gameday'] = row['gameday']

            features_list.append(features)

        features_df = pd.DataFrame(features_list)

        # Remove games without targets (future games)
        features_df = features_df[features_df['home_won'].notna()]

        print(f"Created {len(features_df)} training examples")

        return features_df


def main():
    """Main execution function"""
    # Load raw game data
    games_df = pd.read_csv('data/raw_games.csv')

    # Create feature engineer
    engineer = NFLFeatureEngineer(games_df)

    # Generate features
    features_df = engineer.create_feature_dataset()

    # Save features
    features_df.to_csv('data/features.csv', index=False)

    print("\n" + "=" * 50)
    print("Feature engineering complete!")
    print(f"Features saved to: data/features.csv")
    print(f"Total examples: {len(features_df)}")
    print(f"Total features: {len([col for col in features_df.columns if col not in ['game_id', 'season', 'week', 'home_team', 'away_team', 'gameday', 'home_won', 'score_diff']])}")
    print("=" * 50)


if __name__ == "__main__":
    main()
