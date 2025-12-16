#!/usr/bin/env python3
"""
Enhanced Master Pipeline Script
Runs the complete enhanced NFL prediction pipeline with all advanced features
"""

import sys
import os
import argparse

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from data_collection import NFLDataCollector
from enhanced_feature_engineering import EnhancedNFLFeatureEngineer
from enhanced_training import EnhancedNFLModelTrainer


def main():
    """Run the complete enhanced pipeline"""
    parser = argparse.ArgumentParser(description='Run enhanced NFL prediction pipeline')
    parser.add_argument('--yes', '-y', action='store_true', help='Skip confirmation prompt')
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print("  ENHANCED NFL GAME PREDICTION MODEL - FULL PIPELINE")
    print("=" * 70)
    print("\nThis enhanced version includes:")
    print("  ✓ Margin-of-victory adjusted Elo ratings")
    print("  ✓ Exponentially weighted recent form")
    print("  ✓ Home/away performance splits")
    print("  ✓ Strength of schedule calculations")
    print("  ✓ Momentum and streak indicators")
    print("  ✓ Rest advantage metrics")
    print("  ✓ Feature interactions")
    print("  ✓ XGBoost and LightGBM models")
    print("  ✓ Calibrated probability predictions")
    print("  ✓ Super ensemble with weighted voting")
    print("\nExpected improvement: 2-4% higher accuracy than basic model")
    print("\n" + "=" * 70)

    if not args.yes:
        response = input("\nContinue with enhanced pipeline? (y/n): ")
        if response.lower() != 'y':
            print("Pipeline cancelled.")
            return

    # Step 1: Data Collection
    print("\n\nSTEP 1: DATA COLLECTION")
    print("=" * 70)

    if not os.path.exists('data/raw_games.csv'):
        print("Collecting historical NFL data...")
        collector = NFLDataCollector(start_year=2015, end_year=2023)
        data = collector.collect_all_data()
    else:
        print("Using existing data/raw_games.csv")
        import pandas as pd
        data = {'games': pd.read_csv('data/raw_games.csv')}

    # Step 2: Enhanced Feature Engineering
    print("\n\nSTEP 2: ENHANCED FEATURE ENGINEERING")
    print("=" * 70)
    print("This will take longer than basic features due to advanced calculations...")

    games_df = data['games']
    engineer = EnhancedNFLFeatureEngineer(games_df)
    features_df = engineer.create_complete_feature_dataset()
    features_df.to_csv('data/enhanced_features.csv', index=False)

    print(f"\n✓ Enhanced features saved to data/enhanced_features.csv")

    # Step 3: Enhanced Model Training
    print("\n\nSTEP 3: ENHANCED MODEL TRAINING & EVALUATION")
    print("=" * 70)

    trainer = EnhancedNFLModelTrainer()
    best_model, results = trainer.run_full_pipeline()

    # Summary
    print("\n\n" + "=" * 70)
    print("  🏆 ENHANCED PIPELINE COMPLETE!")
    print("=" * 70)

    if best_model and results:
        print(f"\n✓ Best Model: {trainer.best_model_name}")
        print(f"✓ Test Accuracy: {results[trainer.best_model_name]['test']['accuracy']:.4f}")
        print(f"✓ Test Brier Score: {results[trainer.best_model_name]['test']['brier_score']:.4f}")
        print(f"✓ Test Log Loss: {results[trainer.best_model_name]['test']['log_loss']:.4f}")

        print(f"\n✓ Model saved to: models/best_enhanced_model.joblib")
        print("\nYou can now make predictions using:")
        print("  python src/enhanced_prediction.py --home 'KC' --away 'BUF'")
        print("\nOr list all valid team names:")
        print("  python src/enhanced_prediction.py --list-teams")
    else:
        print("\n⚠️  Training encountered issues. Check logs above.")

    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
