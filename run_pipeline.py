#!/usr/bin/env python3
"""
Master Pipeline Script
Runs the complete NFL prediction model pipeline from data collection to training
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import NFLDataCollector
from feature_engineering import NFLFeatureEngineer
from training import NFLModelTrainer


def main():
    """Run the complete pipeline"""
    print("\n" + "=" * 70)
    print("  NFL GAME PREDICTION MODEL - FULL PIPELINE")
    print("=" * 70)
    print("\nThis will:")
    print("  1. Collect historical NFL game data (2015-2023)")
    print("  2. Engineer features from raw data")
    print("  3. Train multiple ML models")
    print("  4. Evaluate and select the best model")
    print("  5. Save trained models for predictions")
    print("\n" + "=" * 70)

    # Step 1: Data Collection
    print("\n\nSTEP 1: DATA COLLECTION")
    print("=" * 70)
    collector = NFLDataCollector(start_year=2015, end_year=2023)
    data = collector.collect_all_data()

    # Step 2: Feature Engineering
    print("\n\nSTEP 2: FEATURE ENGINEERING")
    print("=" * 70)
    games_df = data['games']
    engineer = NFLFeatureEngineer(games_df)
    features_df = engineer.create_feature_dataset()
    features_df.to_csv('data/features.csv', index=False)
    print(f"\nFeatures saved to data/features.csv")
    print(f"Total training examples: {len(features_df)}")

    # Step 3: Model Training
    print("\n\nSTEP 3: MODEL TRAINING & EVALUATION")
    print("=" * 70)
    trainer = NFLModelTrainer()
    best_model, results = trainer.run_full_pipeline()

    # Summary
    print("\n\n" + "=" * 70)
    print("  PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nBest Model: {trainer.best_model_name}")
    print(f"Test Accuracy: {results[trainer.best_model_name]['test']['accuracy']:.4f}")
    print(f"\nModel saved to: models/best_model.joblib")
    print("\nYou can now make predictions using:")
    print("  python src/prediction.py --home 'Kansas City Chiefs' --away 'Buffalo Bills'")
    print("\nOr use the prediction module in your own code.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
