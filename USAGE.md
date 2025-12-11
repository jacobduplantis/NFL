# NFL Prediction Model - Usage Guide

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Complete Pipeline

This will collect data, engineer features, and train models:

```bash
python run_pipeline.py
```

This process will:
- Download historical NFL game data from 2015-2023
- Calculate Elo ratings and engineer features
- Train 5 different ML models (Logistic Regression, Random Forest, Gradient Boosting, Neural Network, Ensemble)
- Evaluate models and select the best one
- Save trained models to the `models/` directory

**Expected Runtime**: 10-30 minutes depending on your system

### 3. Make Predictions

After training, predict game outcomes:

```bash
python src/prediction.py --home "Kansas City Chiefs" --away "Buffalo Bills"
```

## Detailed Usage

### Data Collection

Collect historical NFL data for specific years:

```python
from src.data_collection import NFLDataCollector

# Collect data from 2015 to 2023
collector = NFLDataCollector(start_year=2015, end_year=2023)
data = collector.collect_all_data()
```

### Feature Engineering

Create features from raw game data:

```python
from src.feature_engineering import NFLFeatureEngineer
import pandas as pd

# Load game data
games_df = pd.read_csv('data/raw_games.csv')

# Create engineer
engineer = NFLFeatureEngineer(games_df)

# Generate features
features_df = engineer.create_feature_dataset()
features_df.to_csv('data/features.csv', index=False)
```

### Model Training

Train and evaluate models:

```python
from src.training import NFLModelTrainer

trainer = NFLModelTrainer()
best_model, results = trainer.run_full_pipeline()

# View results
for model_name, result in results.items():
    print(f"{model_name}: {result['test']['accuracy']:.4f}")
```

### Making Predictions

#### Command Line

```bash
# Predict a single game
python src/prediction.py --home "Dallas Cowboys" --away "Philadelphia Eagles"

# Specify week and season
python src/prediction.py --home "Dallas Cowboys" --away "Philadelphia Eagles" --week 10 --season 2024

# Use a different model
python src/prediction.py --home "Dallas Cowboys" --away "Philadelphia Eagles" --model models/gradient_boosting_20240101.joblib
```

#### Python API

```python
from src.prediction import NFLGamePredictor

# Initialize predictor
predictor = NFLGamePredictor()

# Predict a single game
result = predictor.predict_game(
    home_team="Kansas City Chiefs",
    away_team="Buffalo Bills",
    week=10,
    season=2024
)

print(f"Winner: {result['predicted_winner']}")
print(f"Confidence: {result['confidence']:.1%}")
print(f"Home win probability: {result['home_win_probability']:.1%}")

# Predict multiple games
games = [
    {'home_team': 'Kansas City Chiefs', 'away_team': 'Buffalo Bills'},
    {'home_team': 'San Francisco 49ers', 'away_team': 'Dallas Cowboys'},
    {'home_team': 'Philadelphia Eagles', 'away_team': 'New York Giants'}
]

results = predictor.predict_multiple_games(games)

for result in results:
    predictor.print_prediction(result)
```

## Model Features

The model uses the following features to make predictions:

### Team Performance
- **Elo Ratings**: Dynamic strength ratings that update after each game
- **Win/Loss Records**: Season and recent performance
- **Points Per Game**: Offensive and defensive averages
- **Point Differential**: Scoring margin trends

### Recent Form
- Last 5 games performance
- Recent wins/losses
- Recent scoring trends

### Head-to-Head
- Historical matchup results
- Recent game outcomes between teams

### Situational
- Home field advantage (built into Elo)
- Rest days since last game
- Division games
- Playoff games
- Week number

## Model Performance

Expected accuracy ranges by model:

- **Logistic Regression**: ~63-65% (baseline)
- **Random Forest**: ~66-68%
- **Gradient Boosting**: ~67-69%
- **Neural Network**: ~66-68%
- **Ensemble**: ~68-70% (typically best)

Note: NFL games are inherently unpredictable. A 70% accuracy rate is considered excellent in sports betting contexts.

## Understanding Predictions

### Win Probability
- **50-60%**: Slight favorite (close game)
- **60-70%**: Moderate favorite
- **70-80%**: Strong favorite
- **80%+**: Heavy favorite

### Confidence
The model's confidence is the maximum of the two win probabilities. Higher confidence generally indicates:
- Larger Elo rating difference
- Better recent form for one team
- Stronger historical performance

## Tips for Best Results

1. **Use Current Season Data**: The model performs best when it has seen games from the current season
2. **Consider Context**: The model doesn't account for injuries, weather, or other situational factors
3. **Interpret Probabilities**: A 60% win probability means the team should win 6 out of 10 times, not that they'll definitely win
4. **Update Regularly**: Re-train the model periodically with new game results for best accuracy

## Troubleshooting

### "Model not found" Error
Run the training pipeline first:
```bash
python run_pipeline.py
```

### "Game data not found" Error
Run data collection:
```bash
python src/data_collection.py
```

### Team Name Not Found
Make sure you use the full, official team name:
- ✅ "Kansas City Chiefs"
- ✅ "New England Patriots"
- ❌ "KC Chiefs"
- ❌ "Patriots"

### Low Accuracy
- Ensure you have enough historical data (at least 5+ seasons)
- Verify the model has been trained on recent seasons
- Consider that NFL games have inherent randomness

## Advanced Usage

### Custom Model Training

```python
from src.models import RandomForestPredictor
import pandas as pd

# Load features
features_df = pd.read_csv('data/features.csv')

# Create custom model
model = RandomForestPredictor()

# Prepare data
X, y = model.prepare_data(features_df)

# Train
model.train(X, y)

# Evaluate
metrics = model.evaluate(X, y)
print(f"Accuracy: {metrics['accuracy']:.4f}")
```

### Feature Importance

```python
from src.training import NFLModelTrainer

trainer = NFLModelTrainer()
trainer.load_data()
X_train, X_test, y_train, y_test = trainer.create_train_test_split()
trainer.train_all_models(X_train, y_train)

# Get feature importance for Random Forest
rf_model = trainer.models['random_forest']
importance = rf_model.get_feature_importance(top_n=20)
print(importance)
```

## Further Development

Potential improvements:
- Add weather data integration
- Include injury reports
- Incorporate betting lines
- Add player-level statistics
- Implement real-time data updates
- Build a web interface

## Support

For issues or questions:
1. Check the troubleshooting section
2. Review the code comments and docstrings
3. Ensure all dependencies are installed correctly
