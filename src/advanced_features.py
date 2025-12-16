"""
Advanced Feature Engineering for NFL Prediction
Implements sophisticated features for improved accuracy
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AdvancedNFLFeatures:
    """Creates advanced features beyond basic statistics"""

    def __init__(self, games_df, elo_ratings):
        """
        Initialize advanced feature engineer

        Args:
            games_df: DataFrame with historical game data
            elo_ratings: Dictionary of Elo ratings from basic feature engineering
        """
        self.games_df = games_df.copy()
        self.games_df['gameday'] = pd.to_datetime(self.games_df['gameday'])
        self.games_df = self.games_df.sort_values('gameday').reset_index(drop=True)
        self.elo_ratings = elo_ratings

    def calculate_margin_of_victory_elo(self, k_factor=20, home_advantage=65, initial_elo=1500):
        """
        Enhanced Elo system that considers margin of victory

        Args:
            k_factor: Base K-factor
            home_advantage: Home field advantage points
            initial_elo: Starting Elo for all teams

        Returns:
            Dictionary of MOV-adjusted Elo ratings
        """
        print("Calculating margin-of-victory Elo ratings...")

        mov_elo_ratings = {}
        team_elos = {}

        for idx, row in self.games_df.iterrows():
            home_team = row['home_team']
            away_team = row['away_team']
            game_date = row['gameday']

            # Initialize teams
            if home_team not in team_elos:
                team_elos[home_team] = initial_elo
            if away_team not in team_elos:
                team_elos[away_team] = initial_elo

            # Get current Elos
            home_elo = team_elos[home_team]
            away_elo = team_elos[away_team]

            # Store before-game Elos
            mov_elo_ratings[(home_team, game_date, 'home')] = home_elo
            mov_elo_ratings[(away_team, game_date, 'away')] = away_elo

            # Calculate expected scores
            home_expected = 1 / (1 + 10 ** ((away_elo - (home_elo + home_advantage)) / 400))

            # Update based on actual result with MOV multiplier
            if pd.notna(row['result']):
                home_score = row['home_score']
                away_score = row['away_score']

                if home_score > away_score:
                    home_actual = 1
                    away_actual = 0
                    mov = home_score - away_score
                elif away_score > home_score:
                    home_actual = 0
                    away_actual = 1
                    mov = away_score - home_score
                else:
                    home_actual = 0.5
                    away_actual = 0.5
                    mov = 0

                # Margin of victory multiplier (logarithmic to prevent blowouts from dominating)
                mov_multiplier = np.log(max(mov, 1) + 1) / np.log(2)

                # Adjust K-factor based on MOV and Elo difference
                elo_diff = abs(home_elo - away_elo)
                autocorrelation = 2.2 / ((elo_diff * 0.001) + 2.2)
                adjusted_k = k_factor * mov_multiplier * autocorrelation

                # Update Elos
                team_elos[home_team] += adjusted_k * (home_actual - home_expected)
                team_elos[away_team] += adjusted_k * (away_actual - (1 - home_expected))

        # Store final MOV Elo ratings for future predictions
        self.final_mov_elos = team_elos.copy()
        return mov_elo_ratings

    def get_weighted_recent_form(self, team, game_date, n_games=10):
        """
        Get exponentially weighted recent performance

        Args:
            team: Team name
            game_date: Current game date
            n_games: Number of recent games to consider

        Returns:
            Dictionary of weighted form metrics
        """
        # Get recent games
        team_games = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (self.games_df['gameday'] < game_date)
        ].tail(n_games)

        if len(team_games) == 0:
            return {
                'weighted_win_rate': 0.5,
                'weighted_ppg': 20,
                'weighted_papg': 20,
                'weighted_point_diff': 0,
                'form_trend': 0
            }

        # Exponential weights (more recent = higher weight)
        weights = np.exp(np.linspace(-1, 0, len(team_games)))
        weights = weights / weights.sum()

        wins = []
        points_for = []
        points_against = []

        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                pf = game['home_score']
                pa = game['away_score']
            else:
                pf = game['away_score']
                pa = game['home_score']

            wins.append(1 if pf > pa else 0)
            points_for.append(pf)
            points_against.append(pa)

        # Calculate trend (comparing first half vs second half)
        mid = len(wins) // 2
        if mid > 0:
            first_half_wr = np.mean(wins[:mid])
            second_half_wr = np.mean(wins[mid:])
            form_trend = second_half_wr - first_half_wr
        else:
            form_trend = 0

        return {
            'weighted_win_rate': np.average(wins, weights=weights),
            'weighted_ppg': np.average(points_for, weights=weights),
            'weighted_papg': np.average(points_against, weights=weights),
            'weighted_point_diff': np.average(np.array(points_for) - np.array(points_against), weights=weights),
            'form_trend': form_trend
        }

    def get_home_away_splits(self, team, game_date, season):
        """
        Calculate separate home and away performance

        Args:
            team: Team name
            game_date: Current game date
            season: Current season

        Returns:
            Dictionary of home/away split metrics
        """
        season_games = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (self.games_df['season'] == season) &
            (self.games_df['gameday'] < game_date)
        ]

        home_games = season_games[season_games['home_team'] == team]
        away_games = season_games[season_games['away_team'] == team]

        # Home stats
        if len(home_games) > 0:
            home_wins = (home_games['home_score'] > home_games['away_score']).sum()
            home_ppg = home_games['home_score'].mean()
            home_papg = home_games['away_score'].mean()
        else:
            home_wins = 0
            home_ppg = 20
            home_papg = 20

        # Away stats
        if len(away_games) > 0:
            away_wins = (away_games['away_score'] > away_games['home_score']).sum()
            away_ppg = away_games['away_score'].mean()
            away_papg = away_games['home_score'].mean()
        else:
            away_wins = 0
            away_ppg = 20
            away_papg = 20

        return {
            'home_record': home_wins / max(len(home_games), 1),
            'away_record': away_wins / max(len(away_games), 1),
            'home_ppg': home_ppg,
            'away_ppg': away_ppg,
            'home_papg': home_papg,
            'away_papg': away_papg,
            'home_away_diff': (home_wins / max(len(home_games), 1)) - (away_wins / max(len(away_games), 1))
        }

    def get_strength_of_schedule(self, team, game_date, season):
        """
        Calculate strength of schedule based on opponents' Elo ratings

        Args:
            team: Team name
            game_date: Current game date
            season: Current season

        Returns:
            Average opponent Elo rating
        """
        team_games = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (self.games_df['season'] == season) &
            (self.games_df['gameday'] < game_date)
        ]

        if len(team_games) == 0:
            return 1500  # Default

        opponent_elos = []
        for _, game in team_games.iterrows():
            if game['home_team'] == team:
                opponent = game['away_team']
                location = 'away'
            else:
                opponent = game['home_team']
                location = 'home'

            # Get opponent's Elo at that time
            opp_elo = self.elo_ratings.get((opponent, game['gameday'], location), 1500)
            opponent_elos.append(opp_elo)

        return np.mean(opponent_elos)

    def get_momentum_features(self, team, game_date):
        """
        Calculate momentum indicators (streaks, recent performance trajectory)

        Args:
            team: Team name
            game_date: Current game date

        Returns:
            Dictionary of momentum features
        """
        recent_games = self.games_df[
            ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
            (self.games_df['gameday'] < game_date)
        ].tail(10)

        if len(recent_games) == 0:
            return {
                'current_streak': 0,
                'streak_type': 0,
                'performance_variance': 0
            }

        # Calculate streak
        results = []
        point_diffs = []

        for _, game in recent_games.iterrows():
            if game['home_team'] == team:
                result = 1 if game['home_score'] > game['away_score'] else -1
                point_diff = game['home_score'] - game['away_score']
            else:
                result = 1 if game['away_score'] > game['home_score'] else -1
                point_diff = game['away_score'] - game['home_score']

            results.append(result)
            point_diffs.append(point_diff)

        # Current streak
        streak = 0
        for r in reversed(results):
            if len(results) == 0 or r == results[-1]:
                streak += 1
            else:
                break

        streak_type = results[-1] if len(results) > 0 else 0

        return {
            'current_streak': streak * streak_type,  # Positive for win streak, negative for loss streak
            'streak_type': streak_type,
            'performance_variance': np.std(point_diffs) if len(point_diffs) > 1 else 10
        }

    def get_rest_advantage(self, home_team, away_team, game_date):
        """
        Calculate relative rest advantage

        Args:
            home_team: Home team name
            away_team: Away team name
            game_date: Current game date

        Returns:
            Difference in rest days (positive = home team more rested)
        """
        def get_rest(team):
            last_game = self.games_df[
                ((self.games_df['home_team'] == team) | (self.games_df['away_team'] == team)) &
                (self.games_df['gameday'] < game_date)
            ].tail(1)

            if len(last_game) == 0:
                return 7

            return (game_date - last_game.iloc[0]['gameday']).days

        home_rest = get_rest(home_team)
        away_rest = get_rest(away_team)

        return home_rest - away_rest

    def create_advanced_features(self, row, mov_elo_ratings=None):
        """
        Create all advanced features for a single game

        Args:
            row: Game row from DataFrame
            mov_elo_ratings: Optional MOV-adjusted Elo ratings

        Returns:
            Dictionary of advanced features
        """
        home_team = row['home_team']
        away_team = row['away_team']
        game_date = row['gameday']
        season = row['season']

        features = {}

        # MOV-adjusted Elo ratings
        if mov_elo_ratings:
            # Try exact date lookup first (for historical data)
            home_mov_key = (home_team, game_date, 'home')
            away_mov_key = (away_team, game_date, 'away')

            if home_mov_key in mov_elo_ratings and away_mov_key in mov_elo_ratings:
                # Historical game - use exact MOV Elo from that date
                features['home_mov_elo'] = mov_elo_ratings[home_mov_key]
                features['away_mov_elo'] = mov_elo_ratings[away_mov_key]
            elif hasattr(self, 'final_mov_elos'):
                # Future prediction - use latest MOV Elo ratings
                # Try both full name and abbreviation
                from team_names import get_team_abbreviation
                home_abbrev = get_team_abbreviation(home_team)
                away_abbrev = get_team_abbreviation(away_team)

                features['home_mov_elo'] = (
                    self.final_mov_elos.get(home_team) or
                    self.final_mov_elos.get(home_abbrev) or
                    1500
                )
                features['away_mov_elo'] = (
                    self.final_mov_elos.get(away_team) or
                    self.final_mov_elos.get(away_abbrev) or
                    1500
                )
            else:
                # Fallback to defaults
                features['home_mov_elo'] = 1500
                features['away_mov_elo'] = 1500

            features['mov_elo_diff'] = features['home_mov_elo'] - features['away_mov_elo']

        # Weighted recent form
        home_weighted = self.get_weighted_recent_form(home_team, game_date)
        away_weighted = self.get_weighted_recent_form(away_team, game_date)

        features['home_weighted_win_rate'] = home_weighted['weighted_win_rate']
        features['home_weighted_ppg'] = home_weighted['weighted_ppg']
        features['home_weighted_papg'] = home_weighted['weighted_papg']
        features['home_weighted_diff'] = home_weighted['weighted_point_diff']
        features['home_form_trend'] = home_weighted['form_trend']

        features['away_weighted_win_rate'] = away_weighted['weighted_win_rate']
        features['away_weighted_ppg'] = away_weighted['weighted_ppg']
        features['away_weighted_papg'] = away_weighted['weighted_papg']
        features['away_weighted_diff'] = away_weighted['weighted_point_diff']
        features['away_form_trend'] = away_weighted['form_trend']

        # Home/Away splits
        home_splits = self.get_home_away_splits(home_team, game_date, season)
        away_splits = self.get_home_away_splits(away_team, game_date, season)

        features['home_team_home_record'] = home_splits['home_record']
        features['home_team_home_ppg'] = home_splits['home_ppg']
        features['away_team_away_record'] = away_splits['away_record']
        features['away_team_away_ppg'] = away_splits['away_ppg']

        # Strength of schedule
        features['home_sos'] = self.get_strength_of_schedule(home_team, game_date, season)
        features['away_sos'] = self.get_strength_of_schedule(away_team, game_date, season)
        features['sos_diff'] = features['home_sos'] - features['away_sos']

        # Momentum
        home_momentum = self.get_momentum_features(home_team, game_date)
        away_momentum = self.get_momentum_features(away_team, game_date)

        features['home_streak'] = home_momentum['current_streak']
        features['away_streak'] = away_momentum['current_streak']
        features['home_consistency'] = 1 / (1 + home_momentum['performance_variance'])
        features['away_consistency'] = 1 / (1 + away_momentum['performance_variance'])

        # Rest advantage
        features['rest_advantage'] = self.get_rest_advantage(home_team, away_team, game_date)

        # Feature interactions
        if mov_elo_ratings:
            features['elo_form_interaction'] = features['mov_elo_diff'] * (
                features['home_weighted_win_rate'] - features['away_weighted_win_rate']
            )
            features['elo_momentum_interaction'] = features['mov_elo_diff'] * (
                features['home_streak'] - features['away_streak']
            )

        return features
