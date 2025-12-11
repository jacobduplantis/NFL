# NFL Prediction Model Features

This document details all features used in the NFL game prediction model.

## Feature Categories

### 1. Elo Rating Features (3 features)

**Elo ratings** are a dynamic rating system that updates after each game based on the outcome and opponent strength.

- `home_elo`: Home team's Elo rating before the game (default: 1500)
- `away_elo`: Away team's Elo rating before the game (default: 1500)
- `elo_diff`: Difference in Elo ratings (home_elo - away_elo)

**How it works:**
- Teams start at 1500 rating
- Winners gain rating points, losers lose points
- Upset victories result in larger rating changes
- Home field advantage worth ~65 Elo points
- Ratings carry over across seasons

**Why it matters:** Elo ratings are highly predictive because they:
- Capture team strength in a single number
- Account for strength of schedule automatically
- React quickly to team performance changes
- Have proven effectiveness in sports prediction

### 2. Recent Form Features (10 features)

Features based on the **last 5 games** for each team:

**Home Team:**
- `home_recent_wins`: Number of wins in last 5 games
- `home_recent_losses`: Number of losses in last 5 games
- `home_recent_ppg`: Average points scored per game
- `home_recent_papg`: Average points allowed per game
- `home_recent_diff`: Average point differential

**Away Team:**
- `away_recent_wins`: Number of wins in last 5 games
- `away_recent_losses`: Number of losses in last 5 games
- `away_recent_ppg`: Average points scored per game
- `away_recent_papg`: Average points allowed per game
- `away_recent_diff`: Average point differential

**Why it matters:**
- Recent performance is more predictive than season-long stats
- Captures momentum and current team form
- Identifies teams improving or declining
- More relevant than games from months ago

### 3. Season Statistics Features (8 features)

Features based on **season-to-date performance**:

**Home Team:**
- `home_season_win_pct`: Win percentage this season
- `home_season_ppg`: Average points scored per game
- `home_season_papg`: Average points allowed per game
- `home_season_diff`: Average point differential

**Away Team:**
- `away_season_win_pct`: Win percentage this season
- `away_season_ppg`: Average points scored per game
- `away_season_papg`: Average points allowed per game
- `away_season_diff`: Average point differential

**Why it matters:**
- Provides context for overall team quality
- Balances recent form with longer-term performance
- More stable than very recent games
- Useful early in season when Elo hasn't stabilized

### 4. Head-to-Head Features (3 features)

Features based on **recent matchups** between the two teams:

- `h2h_home_wins`: Home team wins in last 5 H2H games
- `h2h_away_wins`: Away team wins in last 5 H2H games
- `h2h_games`: Total H2H games available (≤5)

**Why it matters:**
- Some teams match up better against certain opponents
- Historical performance can indicate tactical advantages
- Division rivals play frequently and know each other well
- Captures stylistic matchup effects

### 5. Situational Features (5 features)

Features describing **game context and rest**:

- `home_rest_days`: Days since home team's last game
- `away_rest_days`: Days since away team's last game
- `is_division_game`: 1 if division game, 0 otherwise
- `week`: Week number in the season (1-18 for regular season)
- `is_playoff`: 1 if playoff game, 0 if regular season

**Why it matters:**
- **Rest days**: Teams on short rest (Thursday games) often underperform
- **Division games**: Often closer and more competitive
- **Week number**: Teams may improve or decline as season progresses
- **Playoffs**: Higher stakes, different dynamics

## Total Features

**29 features** in total:
- 3 Elo rating features
- 10 recent form features
- 8 season statistics features
- 3 head-to-head features
- 5 situational features

## Target Variable

- `home_won`: Binary (1 if home team won, 0 if away team won)

## Feature Engineering Process

### 1. Chronological Processing
- Games are processed in chronological order
- Features only use data from **before** the game
- Prevents data leakage (looking into the future)

### 2. Handling Missing Data
- For teams' first games, default values are used
- Recent form uses all available games (up to 5)
- H2H features handle cases where teams haven't played recently

### 3. Feature Scaling
- Features are standardized (zero mean, unit variance) before model training
- Prevents features with larger scales from dominating
- Required for neural networks and distance-based models

## Most Important Features

Based on typical Random Forest feature importance:

1. **elo_diff** - Single most predictive feature (~15-20% importance)
2. **home_elo** - Raw home team strength
3. **away_elo** - Raw away team strength
4. **home_recent_diff** - Recent scoring margin
5. **away_recent_diff** - Recent scoring margin
6. **home_season_win_pct** - Overall record
7. **away_season_win_pct** - Overall record
8. **home_recent_wins** - Short-term form
9. **away_recent_wins** - Short-term form
10. **home_rest_days** - Fatigue factor

The Elo-based features consistently rank as the most important, which validates the Elo rating system's effectiveness for NFL prediction.

## Feature Correlations

### Highly Correlated Features
- `home_elo` and `home_season_win_pct` (~0.7)
- `away_elo` and `away_season_win_pct` (~0.7)
- `recent_wins` and `recent_diff` (~0.9)

These correlations are expected and not problematic for tree-based models. The ensemble approach and regularization help prevent overfitting.

### Intentionally Independent Features
- Elo ratings vs. H2H history
- Recent form vs. season statistics
- Rest days vs. performance metrics

This independence ensures the model captures different aspects of game prediction.

## Missing Features (Future Work)

Features that could improve the model:

1. **Weather**: Temperature, wind, precipitation
2. **Injuries**: Key player availability
3. **Coaching**: Head coach win percentage
4. **Betting Lines**: Market expectations
5. **Home Field Type**: Dome vs. outdoor
6. **Travel Distance**: Cross-country games
7. **Primetime**: Performance in national TV games
8. **Streaks**: Current win/loss streaks

## Feature Engineering Philosophy

The features were designed following these principles:

1. **No Future Information**: Only use data available before the game
2. **Multiple Timescales**: Recent (5 games), season, all-time
3. **Relative and Absolute**: Both team strength and matchup factors
4. **Simple and Interpretable**: Features humans understand
5. **Robust**: Handle edge cases (new teams, start of season)
6. **Proven Effective**: Based on successful sports prediction research

This feature set balances predictive power with simplicity and interpretability.
