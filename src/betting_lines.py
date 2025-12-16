"""
NFL Betting Lines Integration
Fetches and processes historical betting lines data
"""

import pandas as pd
import numpy as np
import requests
from io import StringIO
import warnings
warnings.filterwarnings('ignore')


class NFLBettingLines:
    """
    Fetches and processes NFL betting lines data

    Data sources:
    - Primary: spreadspoke_scores.csv (1966-present, betting data from 1979+)
      GitHub: https://github.com/slieb74/NFL-Betting-Data
    - Backup: nflverse data (includes spread_line and total_line)
    """

    def __init__(self):
        self.betting_data = None
        self.source_url = "https://raw.githubusercontent.com/slieb74/NFL-Betting-Data/master/spreadspoke_scores.csv"

    def fetch_betting_data(self):
        """
        Fetch historical betting lines from GitHub

        Returns:
            DataFrame with betting lines data
        """
        print("Fetching historical betting lines data...")
        print(f"Source: {self.source_url}")

        try:
            # Fetch data from GitHub
            response = requests.get(self.source_url, timeout=30)
            response.raise_for_status()

            # Parse CSV
            self.betting_data = pd.read_csv(StringIO(response.text))

            print(f"✓ Loaded {len(self.betting_data)} games with betting data")

            # Display available columns
            print(f"\nAvailable columns: {list(self.betting_data.columns)}")

            # Show date range
            if 'schedule_date' in self.betting_data.columns:
                self.betting_data['schedule_date'] = pd.to_datetime(self.betting_data['schedule_date'])
                print(f"Date range: {self.betting_data['schedule_date'].min()} to {self.betting_data['schedule_date'].max()}")

            return self.betting_data

        except Exception as e:
            print(f"Error fetching betting data: {e}")
            return None

    def process_betting_data(self):
        """
        Process and clean betting data

        Returns:
            Cleaned DataFrame
        """
        if self.betting_data is None:
            self.fetch_betting_data()

        if self.betting_data is None:
            return None

        print("\nProcessing betting data...")

        df = self.betting_data.copy()

        # Standardize team names to match our format
        team_mapping = {
            'Washington Redskins': 'Washington Commanders',
            'Washington Football Team': 'Washington Commanders',
            'Oakland Raiders': 'Las Vegas Raiders',
            'San Diego Chargers': 'Los Angeles Chargers',
            'St. Louis Rams': 'Los Angeles Rams',
        }

        for col in ['team_home', 'team_away', 'team_favorite_id']:
            if col in df.columns:
                df[col] = df[col].replace(team_mapping)

        # Ensure date column exists
        if 'schedule_date' in df.columns:
            df['gameday'] = pd.to_datetime(df['schedule_date'])
        elif 'schedule_week' in df.columns and 'schedule_season' in df.columns:
            # Approximate date from season and week
            df['gameday'] = pd.to_datetime(df['schedule_season'].astype(str) + '-09-01') + pd.to_timedelta(df['schedule_week'] * 7, unit='D')

        # Key betting features
        betting_features = {}

        # Spread (point spread)
        if 'spread_favorite' in df.columns:
            df['spread'] = df['spread_favorite'].fillna(0)
            # Positive = favorite, negative = underdog
            # Adjust so positive = home team favored
            if 'team_favorite_id' in df.columns and 'team_home' in df.columns:
                df['home_spread'] = df.apply(
                    lambda row: row['spread'] if row['team_favorite_id'] == row['team_home'] else -row['spread'],
                    axis=1
                )
            else:
                df['home_spread'] = df['spread']

        # Over/Under (total points)
        if 'over_under_line' in df.columns:
            df['total_line'] = df['over_under_line'].fillna(0)

        # Moneyline (if available)
        if 'ml_favorite' in df.columns:
            df['moneyline_favorite'] = df['ml_favorite'].fillna(0)

        # Weather data (if available)
        weather_cols = ['weather_temperature', 'weather_wind_mph', 'weather_humidity']
        for col in weather_cols:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')

        print(f"✓ Processed {len(df)} games")

        self.betting_data = df
        return df

    def get_betting_features(self, home_team, away_team, game_date, season):
        """
        Get betting line features for a specific game

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Game date
            season: Season year

        Returns:
            Dictionary of betting features
        """
        if self.betting_data is None:
            self.process_betting_data()

        if self.betting_data is None:
            return self._default_betting_features()

        # Try to find the game
        game_date = pd.to_datetime(game_date)

        # Match by teams and date (within 3 days to handle date discrepancies)
        mask = (
            (self.betting_data['team_home'] == home_team) &
            (self.betting_data['team_away'] == away_team) &
            (abs((self.betting_data['gameday'] - game_date).dt.days) <= 3)
        )

        game = self.betting_data[mask]

        if len(game) == 0:
            # Try reverse team order
            mask = (
                (self.betting_data['team_home'] == away_team) &
                (self.betting_data['team_away'] == home_team) &
                (abs((self.betting_data['gameday'] - game_date).dt.days) <= 3)
            )
            game = self.betting_data[mask]

            if len(game) == 0:
                # No betting data found, use defaults
                return self._default_betting_features()

        game = game.iloc[0]

        features = {
            'betting_spread': game.get('home_spread', 0),
            'betting_total': game.get('total_line', 0),
            'betting_spread_abs': abs(game.get('home_spread', 0)),
            'betting_is_favorite': 1 if game.get('home_spread', 0) > 0 else 0,
            'betting_favorite_margin': abs(game.get('home_spread', 0)),
        }

        # Weather features if available
        if 'weather_temperature' in game and pd.notna(game['weather_temperature']):
            features['weather_temperature'] = game['weather_temperature']
        if 'weather_wind_mph' in game and pd.notna(game['weather_wind_mph']):
            features['weather_wind_mph'] = game['weather_wind_mph']
        if 'weather_humidity' in game and pd.notna(game['weather_humidity']):
            features['weather_humidity'] = game['weather_humidity']

        return features

    def _default_betting_features(self):
        """Default betting features when data not available"""
        return {
            'betting_spread': 0,
            'betting_total': 47.5,  # NFL average
            'betting_spread_abs': 0,
            'betting_is_favorite': 0,
            'betting_favorite_margin': 0,
        }

    def add_betting_to_games(self, games_df):
        """
        Add betting line data to games DataFrame

        Args:
            games_df: DataFrame with game data

        Returns:
            DataFrame with betting features added
        """
        if self.betting_data is None:
            self.process_betting_data()

        print("\nAdding betting lines to games data...")

        games_df = games_df.copy()
        games_df['gameday'] = pd.to_datetime(games_df['gameday'])

        # Add betting features
        betting_features_list = []

        for idx, row in games_df.iterrows():
            features = self.get_betting_features(
                row['home_team'],
                row['away_team'],
                row['gameday'],
                row['season']
            )
            betting_features_list.append(features)

            if idx % 500 == 0:
                print(f"  Processed {idx}/{len(games_df)} games")

        # Convert to DataFrame and join
        betting_df = pd.DataFrame(betting_features_list)
        games_with_betting = pd.concat([games_df, betting_df], axis=1)

        print(f"✓ Added betting features to {len(games_with_betting)} games")

        return games_with_betting

    def save_processed_data(self, output_path='data/betting_lines.csv'):
        """Save processed betting data"""
        if self.betting_data is not None:
            self.betting_data.to_csv(output_path, index=False)
            print(f"✓ Saved betting data to {output_path}")


def main():
    """Demo/test function"""
    print("=" * 70)
    print("NFL BETTING LINES DATA INTEGRATION")
    print("=" * 70)

    # Initialize
    betting = NFLBettingLines()

    # Fetch and process data
    data = betting.process_betting_data()

    if data is not None:
        print("\n" + "=" * 70)
        print("SAMPLE DATA")
        print("=" * 70)
        print(data[['schedule_season', 'team_home', 'team_away', 'home_spread', 'total_line']].head(10))

        # Save
        betting.save_processed_data()

        print("\n" + "=" * 70)
        print("BETTING DATA READY!")
        print("=" * 70)
        print("\nKey features available:")
        print("  • betting_spread: Point spread (+ = home favored)")
        print("  • betting_total: Over/under line")
        print("  • betting_is_favorite: 1 if home team favored")
        print("  • betting_favorite_margin: Absolute spread value")
        print("\nData source: https://github.com/slieb74/NFL-Betting-Data")


if __name__ == "__main__":
    main()
