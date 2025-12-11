"""
NFL Data Collection Module
Fetches historical NFL game data and team statistics
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings
warnings.filterwarnings('ignore')

try:
    import nfl_data_py as nfl
except ImportError:
    print("Warning: nfl_data_py not installed. Install with: pip install nfl-data-py")
    nfl = None


class NFLDataCollector:
    """Collects and processes NFL historical data"""

    def __init__(self, start_year=2015, end_year=2024):
        """
        Initialize data collector

        Args:
            start_year: First season to collect data from
            end_year: Last season to collect data from
        """
        self.start_year = start_year
        self.end_year = end_year
        self.data_dir = 'data'
        os.makedirs(self.data_dir, exist_ok=True)

    def fetch_game_data(self):
        """Fetch historical game results"""
        print(f"Fetching game data from {self.start_year} to {self.end_year}...")

        if nfl is None:
            raise ImportError("nfl_data_py is required. Install with: pip install nfl-data-py")

        # Get schedule data with results
        years = list(range(self.start_year, self.end_year + 1))
        schedules = nfl.import_schedules(years)

        # Filter to regular season and playoffs, exclude games not yet played
        schedules = schedules[schedules['game_type'].isin(['REG', 'WC', 'DIV', 'CON', 'SB'])]
        schedules = schedules[schedules['result'].notna()]

        print(f"Collected {len(schedules)} completed games")

        # Save raw data
        schedules.to_csv(f'{self.data_dir}/raw_games.csv', index=False)

        return schedules

    def fetch_team_stats(self):
        """Fetch team statistics by season"""
        print("Fetching team statistics...")

        if nfl is None:
            raise ImportError("nfl_data_py is required")

        years = list(range(self.start_year, self.end_year + 1))

        # Weekly team stats
        weekly_stats = nfl.import_weekly_data(years)

        # Aggregate by team and season
        team_season_stats = weekly_stats.groupby(['season', 'recent_team']).agg({
            'completions': 'sum',
            'attempts': 'sum',
            'passing_yards': 'sum',
            'passing_tds': 'sum',
            'interceptions': 'sum',
            'sacks': 'sum',
            'rushing_yards': 'sum',
            'rushing_tds': 'sum',
            'receptions': 'sum',
            'receiving_yards': 'sum',
            'receiving_tds': 'sum',
            'fantasy_points': 'sum'
        }).reset_index()

        team_season_stats.to_csv(f'{self.data_dir}/team_season_stats.csv', index=False)

        return team_season_stats

    def fetch_pbp_stats(self):
        """Fetch play-by-play aggregated statistics"""
        print("Fetching play-by-play statistics...")

        if nfl is None:
            raise ImportError("nfl_data_py is required")

        years = list(range(self.start_year, self.end_year + 1))

        # This is a large dataset, so we'll aggregate it
        pbp_data = nfl.import_pbp_data(years)

        # Calculate team offensive and defensive stats from PBP
        # Group by game and team
        team_game_stats = pbp_data.groupby(['game_id', 'season', 'week', 'posteam']).agg({
            'yards_gained': 'sum',
            'pass_attempt': 'sum',
            'rush_attempt': 'sum',
            'third_down_converted': 'sum',
            'third_down_failed': 'sum',
            'fourth_down_converted': 'sum',
            'fourth_down_failed': 'sum',
            'interception': 'sum',
            'fumble_lost': 'sum',
            'touchdown': 'sum',
            'field_goal_result': lambda x: (x == 'made').sum()
        }).reset_index()

        team_game_stats.columns = ['game_id', 'season', 'week', 'team',
                                     'total_yards', 'pass_attempts', 'rush_attempts',
                                     'third_down_conversions', 'third_down_fails',
                                     'fourth_down_conversions', 'fourth_down_fails',
                                     'interceptions_thrown', 'fumbles_lost',
                                     'touchdowns', 'field_goals_made']

        team_game_stats.to_csv(f'{self.data_dir}/team_game_stats.csv', index=False)

        return team_game_stats

    def collect_all_data(self):
        """Collect all necessary data"""
        print("Starting NFL data collection...")
        print("=" * 50)

        # Fetch game schedules and results
        games = self.fetch_game_data()

        print("\nData collection complete!")
        print(f"Total games collected: {len(games)}")

        return {
            'games': games
        }


def main():
    """Main execution function"""
    # Collect data from 2015 to 2023 (2024 is current season)
    collector = NFLDataCollector(start_year=2015, end_year=2023)
    data = collector.collect_all_data()

    print("\n" + "=" * 50)
    print("Data collection complete!")
    print(f"Games saved to: data/raw_games.csv")
    print("=" * 50)


if __name__ == "__main__":
    main()
