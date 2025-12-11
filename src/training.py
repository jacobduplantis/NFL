"""
NFL Model Training Pipeline
Trains and evaluates multiple ML models for game prediction
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, TimeSeriesSplit
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from models import create_all_models


class NFLModelTrainer:
    """Trains and evaluates NFL prediction models"""

    def __init__(self, features_path='data/features.csv'):
        """
        Initialize trainer

        Args:
            features_path: Path to features CSV file
        """
        self.features_path = features_path
        self.features_df = None
        self.models = None
        self.results = {}
        self.best_model = None
        self.models_dir = 'models'
        os.makedirs(self.models_dir, exist_ok=True)

    def load_data(self):
        """Load feature dataset"""
        print("Loading feature data...")
        self.features_df = pd.read_csv(self.features_path)

        print(f"Loaded {len(self.features_df)} games")
        print(f"Date range: {self.features_df['gameday'].min()} to {self.features_df['gameday'].max()}")

        # Check class balance
        home_wins = self.features_df['home_won'].sum()
        total_games = len(self.features_df)
        print(f"Home team wins: {home_wins}/{total_games} ({100*home_wins/total_games:.1f}%)")

    def create_train_test_split(self, test_size=0.2, time_based=True):
        """
        Split data into train and test sets

        Args:
            test_size: Proportion of data for testing
            time_based: If True, use chronological split (more realistic)

        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("\nCreating train/test split...")

        # Get all models to determine feature columns
        temp_model = list(create_all_models().values())[0]
        feature_columns = temp_model.get_feature_columns(self.features_df)

        X = self.features_df[feature_columns].fillna(0)
        y = self.features_df['home_won'].values

        if time_based:
            # Sort by date and split chronologically
            sorted_indices = self.features_df.sort_values('gameday').index
            split_idx = int(len(sorted_indices) * (1 - test_size))

            train_indices = sorted_indices[:split_idx]
            test_indices = sorted_indices[split_idx:]

            X_train = X.loc[train_indices]
            X_test = X.loc[test_indices]
            y_train = y[train_indices]
            y_test = y[test_indices]

            print(f"Using chronological split")
            print(f"Training: {self.features_df.loc[train_indices, 'gameday'].min()} to {self.features_df.loc[train_indices, 'gameday'].max()}")
            print(f"Testing:  {self.features_df.loc[test_indices, 'gameday'].min()} to {self.features_df.loc[test_indices, 'gameday'].max()}")
        else:
            # Random split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
            print(f"Using random split")

        print(f"Training set: {len(X_train)} games")
        print(f"Test set: {len(X_test)} games")

        return X_train, X_test, y_train, y_test

    def train_all_models(self, X_train, y_train):
        """
        Train all models

        Args:
            X_train: Training features
            y_train: Training labels
        """
        print("\n" + "=" * 50)
        print("Training models...")
        print("=" * 50)

        self.models = create_all_models()

        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            model.train(X_train, y_train)

    def evaluate_all_models(self, X_train, X_test, y_train, y_test):
        """
        Evaluate all trained models

        Args:
            X_train: Training features
            X_test: Test features
            y_train: Training labels
            y_test: Test labels
        """
        print("\n" + "=" * 50)
        print("Evaluating models...")
        print("=" * 50)

        for name, model in self.models.items():
            print(f"\n{name}:")

            # Training performance
            train_metrics = model.evaluate(X_train, y_train)
            print(f"  Training Accuracy: {train_metrics['accuracy']:.4f}")
            print(f"  Training Brier Score: {train_metrics['brier_score']:.4f}")

            # Test performance
            test_metrics = model.evaluate(X_test, y_test)
            print(f"  Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"  Test Brier Score: {test_metrics['brier_score']:.4f}")

            self.results[name] = {
                'train': train_metrics,
                'test': test_metrics,
                'model': model
            }

            # Feature importance (if available)
            if hasattr(model, 'get_feature_importance') and model.get_feature_importance() is not None:
                importance_df = model.get_feature_importance(top_n=10)
                print(f"\n  Top 10 Features:")
                for idx, row in importance_df.iterrows():
                    print(f"    {row['feature']}: {row['importance']:.4f}")

    def select_best_model(self):
        """Select the best performing model based on test accuracy"""
        print("\n" + "=" * 50)
        print("Model Comparison")
        print("=" * 50)

        best_accuracy = 0
        best_name = None

        print(f"\n{'Model':<25} {'Train Acc':<12} {'Test Acc':<12} {'Test Brier':<12}")
        print("-" * 65)

        for name, result in self.results.items():
            train_acc = result['train']['accuracy']
            test_acc = result['test']['accuracy']
            test_brier = result['test']['brier_score']

            print(f"{name:<25} {train_acc:<12.4f} {test_acc:<12.4f} {test_brier:<12.4f}")

            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_name = name

        print("-" * 65)
        print(f"\nBest model: {best_name} (Test Accuracy: {best_accuracy:.4f})")

        self.best_model = self.models[best_name]
        self.best_model_name = best_name

        return best_name, best_accuracy

    def save_models(self):
        """Save all trained models to disk"""
        print("\n" + "=" * 50)
        print("Saving models...")
        print("=" * 50)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, model in self.models.items():
            model_path = f"{self.models_dir}/{name}_{timestamp}.joblib"
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")

        # Save best model separately
        best_model_path = f"{self.models_dir}/best_model.joblib"
        joblib.dump(self.best_model, best_model_path)
        print(f"\nSaved best model ({self.best_model_name}) to {best_model_path}")

        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'timestamp': timestamp,
            'results': {name: {
                'train_accuracy': result['train']['accuracy'],
                'test_accuracy': result['test']['accuracy'],
                'test_brier': result['test']['brier_score']
            } for name, result in self.results.items()}
        }

        import json
        with open(f"{self.models_dir}/metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Saved metadata to {self.models_dir}/metadata.json")

    def run_full_pipeline(self):
        """Run complete training pipeline"""
        print("=" * 50)
        print("NFL MODEL TRAINING PIPELINE")
        print("=" * 50)

        # Load data
        self.load_data()

        # Create train/test split
        X_train, X_test, y_train, y_test = self.create_train_test_split(
            test_size=0.2,
            time_based=True
        )

        # Train models
        self.train_all_models(X_train, y_train)

        # Evaluate models
        self.evaluate_all_models(X_train, X_test, y_train, y_test)

        # Select best model
        self.select_best_model()

        # Save models
        self.save_models()

        print("\n" + "=" * 50)
        print("TRAINING COMPLETE!")
        print("=" * 50)

        return self.best_model, self.results


def main():
    """Main execution function"""
    trainer = NFLModelTrainer()
    best_model, results = trainer.run_full_pipeline()

    print("\nYour NFL prediction model is ready!")
    print(f"Best model: {trainer.best_model_name}")
    print(f"Test accuracy: {results[trainer.best_model_name]['test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
