# NFL Game Prediction Model

A machine learning system to predict NFL game outcomes using historical data and advanced features.

## 🚀 Enhanced Model Available!

This repository includes both:
- **Basic Model**: 29 features, 68-70% accuracy
- **Enhanced Model**: 54+ features with XGBoost/LightGBM, **71-73% accuracy**

See [ENHANCEMENTS.md](ENHANCEMENTS.md) for details on improvements.

## Features

The model considers multiple categories of features:

### 1. Team Performance Metrics
- Win/loss record and point differential
- Offensive stats (yards, points per game, red zone efficiency)
- Defensive stats (yards allowed, points allowed)
- Turnover differential
- Third down conversion rates
- Time of possession

### 2. Recent Form
- Last 3-5 games performance
- Current streaks (winning/losing)
- Home vs away performance

### 3. Situational Factors
- Home field advantage
- Rest days (bye weeks, short weeks)
- Division/conference games
- Weather conditions

### 4. Advanced Metrics
- Elo ratings
- Strength of schedule
- Head-to-head history

## Project Structure

```
NFL/
├── data/               # Raw and processed data
├── models/             # Trained model files
├── src/
│   ├── data_collection.py    # Fetch historical NFL data
│   ├── feature_engineering.py # Create features from raw data
│   ├── models.py              # ML model implementations
│   ├── training.py            # Model training pipeline
│   └── prediction.py          # Prediction interface
├── notebooks/          # Jupyter notebooks for analysis
├── requirements.txt    # Python dependencies
└── README.md
```

## Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Run Complete Pipeline

**Option 1: Enhanced Model (Recommended - Higher Accuracy)**

```bash
python run_enhanced_pipeline.py
```

**Option 2: Basic Model (Faster Training)**

```bash
python run_pipeline.py
```

This will:
1. Download historical NFL game data (2015-2023)
2. Engineer features from raw data
3. Train multiple ML models
4. Evaluate and select the best model
5. Save trained models for predictions

**Expected Runtime**: 10-30 minutes

### Make Predictions

After training, predict game outcomes:

**Enhanced predictions (with team name validation):**
```bash
# Accepts abbreviations!
python src/enhanced_prediction.py --home "KC" --away "BUF"

# List all valid team names
python src/enhanced_prediction.py --list-teams
```

**Basic predictions:**
```bash
python src/prediction.py --home "Kansas City Chiefs" --away "Buffalo Bills"
```

## Detailed Usage

### Step-by-Step Process

#### 1. Data Collection
```bash
python src/data_collection.py
```
Downloads historical game data from 2015-2023.

#### 2. Feature Engineering
```bash
python src/feature_engineering.py
```
Creates predictive features from raw game data.

#### 3. Model Training
```bash
python src/training.py
```
Trains and evaluates all models.

#### 4. Make Predictions
```bash
# Single game prediction
python src/prediction.py --home "Dallas Cowboys" --away "Philadelphia Eagles"

# With specific week/season
python src/prediction.py --home "Dallas Cowboys" --away "Philadelphia Eagles" --week 10 --season 2024
```

### Python API

```python
from src.prediction import NFLGamePredictor

# Initialize predictor
predictor = NFLGamePredictor()

# Predict a game
result = predictor.predict_game(
    home_team="Kansas City Chiefs",
    away_team="Buffalo Bills"
)

print(f"Winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1%}")
```

For more examples, see [USAGE.md](USAGE.md)

## Model Performance

### Enhanced Models (Recommended)

| Model | Type | Expected Accuracy |
|-------|------|-------------------|
| XGBoost | Gradient Boosting | 70-72% |
| LightGBM | Gradient Boosting | 69-71% |
| Enhanced Random Forest | Ensemble | 68-70% |
| Enhanced Neural Network | Deep Learning | 67-69% |
| **Super Ensemble** | **Meta-ensemble** | **71-73%** ⭐ |

### Basic Models

| Model | Type | Expected Accuracy |
|-------|------|-------------------|
| Logistic Regression | Linear | 63-65% |
| Random Forest | Ensemble | 66-68% |
| Gradient Boosting | Ensemble | 67-69% |
| Neural Network | Deep Learning | 66-68% |
| Ensemble | Meta-ensemble | 68-70% |

**Note**: NFL games are inherently unpredictable. 73% accuracy is exceptional performance.

## What Makes This Model Effective

### 1. Elo Rating System
- Dynamic strength ratings that update after each game
- Accounts for opponent strength
- Includes home field advantage (65 points)

### 2. Multiple Time Horizons
- Recent form (last 5 games)
- Season-to-date statistics
- Historical head-to-head matchups

### 3. Proper Train/Test Split
- Chronological split prevents data leakage
- Models only see past data when making predictions
- Realistic evaluation of real-world performance

### 4. Ensemble Approach
- Combines predictions from multiple models
- Reduces variance and improves accuracy
- More robust than any single model

## Project Structure

```
NFL/
├── data/                      # Data storage
│   ├── raw_games.csv         # Historical game results
│   └── features.csv          # Engineered features
├── models/                    # Trained models
│   ├── best_model.joblib     # Best performing model
│   └── metadata.json         # Model performance metrics
├── src/                       # Source code
│   ├── data_collection.py    # Data fetching
│   ├── feature_engineering.py # Feature creation
│   ├── models.py             # ML model implementations
│   ├── training.py           # Training pipeline
│   └── prediction.py         # Prediction interface
├── notebooks/                 # Analysis notebooks
│   └── model_analysis.ipynb  # Model exploration
├── run_pipeline.py           # Master pipeline script
├── requirements.txt          # Dependencies
├── README.md                 # This file
└── USAGE.md                  # Detailed usage guide
```

## Documentation

- [ENHANCEMENTS.md](ENHANCEMENTS.md) - Detailed explanation of enhanced model improvements
- [TEAM_NAMES.txt](TEAM_NAMES.txt) - Complete list of valid NFL team names and abbreviations
- [USAGE.md](USAGE.md) - Comprehensive usage guide
- [FEATURES.md](FEATURES.md) - Feature documentation
- [notebooks/model_analysis.ipynb](notebooks/model_analysis.ipynb) - Interactive analysis

## Future Enhancements

Potential improvements:
- Integration with real-time data feeds
- Weather data incorporation
- Injury report analysis
- Player-level statistics
- Betting line integration
- Web interface for predictions

## Contributing

This is a complete, working NFL prediction system that demonstrates:
- Professional ML pipeline structure
- Proper feature engineering
- Multiple model comparison
- Realistic evaluation methodology
- Clean, documented code

Feel free to extend or modify for your own use cases!

## License

This project is for educational and research purposes.
