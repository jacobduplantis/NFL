"""
Enhanced NFL Model Training Pipeline
Uses advanced features and models for maximum accuracy
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import joblib
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from enhanced_models import create_enhanced_models


class EnhancedNFLModelTrainer:
    """Trains and evaluates enhanced NFL prediction models"""

    def __init__(self, features_path='data/enhanced_features.csv'):
        """
        Initialize enhanced trainer

        Args:
            features_path: Path to enhanced features CSV file
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
        print("Loading enhanced feature data...")

        if not os.path.exists(self.features_path):
            print(f"Error: {self.features_path} not found.")
            print("Please run: python src/enhanced_feature_engineering.py")
            return False

        self.features_df = pd.read_csv(self.features_path)

        print(f"Loaded {len(self.features_df)} games")

        # Convert gameday to datetime if it exists
        if 'gameday' in self.features_df.columns:
            self.features_df['gameday'] = pd.to_datetime(self.features_df['gameday'])
            print(f"Date range: {self.features_df['gameday'].min()} to {self.features_df['gameday'].max()}")
        else:
            print("Note: gameday column not found in features")

        # Check class balance
        home_wins = self.features_df['home_won'].sum()
        total_games = len(self.features_df)
        print(f"Home team wins: {home_wins}/{total_games} ({100*home_wins/total_games:.1f}%)")

        return True

    def create_train_val_test_split(self, val_size=0.15, test_size=0.15):
        """
        Create train/validation/test splits (chronological)

        Args:
            val_size: Proportion for validation
            test_size: Proportion for testing

        Returns:
            Tuple of splits
        """
        print("\nCreating train/validation/test split...")

        # Get feature columns
        temp_model = list(create_enhanced_models().values())[0]
        feature_columns = temp_model.get_feature_columns(self.features_df)

        X = self.features_df[feature_columns].fillna(0)
        y = self.features_df['home_won'].values

        # Chronological split
        if 'gameday' in self.features_df.columns:
            sorted_indices = self.features_df.sort_values('gameday').index
        elif 'season' in self.features_df.columns and 'week' in self.features_df.columns:
            # Fallback: sort by season and week
            sorted_indices = self.features_df.sort_values(['season', 'week']).index
        else:
            # Last resort: use existing order
            sorted_indices = self.features_df.index

        n_samples = len(sorted_indices)
        train_end = int(n_samples * (1 - val_size - test_size))
        val_end = int(n_samples * (1 - test_size))

        train_indices = sorted_indices[:train_end]
        val_indices = sorted_indices[train_end:val_end]
        test_indices = sorted_indices[val_end:]

        X_train = X.loc[train_indices]
        X_val = X.loc[val_indices]
        X_test = X.loc[test_indices]

        y_train = y[train_indices]
        y_val = y[val_indices]
        y_test = y[test_indices]

        print(f"Using chronological split:")
        print(f"  Training:   {len(X_train):5d} games")
        print(f"  Validation: {len(X_val):5d} games")
        print(f"  Testing:    {len(X_test):5d} games")

        if 'gameday' in self.features_df.columns:
            print(f"  Train dates: {self.features_df.loc[train_indices, 'gameday'].min()} to {self.features_df.loc[train_indices, 'gameday'].max()}")
            print(f"  Val dates:   {self.features_df.loc[val_indices, 'gameday'].min()} to {self.features_df.loc[val_indices, 'gameday'].max()}")
            print(f"  Test dates:  {self.features_df.loc[test_indices, 'gameday'].min()} to {self.features_df.loc[test_indices, 'gameday'].max()}")

        return X_train, X_val, X_test, y_train, y_val, y_test

    def train_all_models(self, X_train, y_train, X_val, y_val):
        """
        Train all enhanced models

        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        print("\n" + "=" * 70)
        print("TRAINING ENHANCED MODELS")
        print("=" * 70)

        self.models = create_enhanced_models(calibrate=True)

        # Train models and track failures
        failed_models = []
        for name, model in self.models.items():
            print(f"\nTraining {name}...")
            try:
                model.train(X_train, y_train, X_val, y_val)
            except Exception as e:
                print(f"  Error training {name}: {e}")
                print(f"  Skipping {name}...")
                failed_models.append(name)

        # Remove failed models after iteration
        for name in failed_models:
            del self.models[name]

        if len(self.models) == 0:
            print("\n⚠️  All models failed to train!")
            return False

        print(f"\n✓ Successfully trained {len(self.models)} models")
        return True

    def evaluate_all_models(self, X_train, X_val, X_test, y_train, y_val, y_test):
        """
        Evaluate all trained models

        Args:
            X_train, X_val, X_test: Feature sets
            y_train, y_val, y_test: Label sets
        """
        print("\n" + "=" * 70)
        print("EVALUATING MODELS")
        print("=" * 70)

        for name, model in self.models.items():
            print(f"\n{model.name}:")

            # Training performance
            try:
                train_metrics = model.evaluate(X_train, y_train)
                print(f"  Training   - Accuracy: {train_metrics['accuracy']:.4f}, Brier: {train_metrics['brier_score']:.4f}, LogLoss: {train_metrics['log_loss']:.4f}")
            except Exception as e:
                print(f"  Training evaluation error: {e}")
                train_metrics = {'accuracy': 0, 'brier_score': 1, 'log_loss': 10}

            # Validation performance
            try:
                val_metrics = model.evaluate(X_val, y_val)
                print(f"  Validation - Accuracy: {val_metrics['accuracy']:.4f}, Brier: {val_metrics['brier_score']:.4f}, LogLoss: {val_metrics['log_loss']:.4f}")
            except Exception as e:
                print(f"  Validation evaluation error: {e}")
                val_metrics = {'accuracy': 0, 'brier_score': 1, 'log_loss': 10}

            # Test performance
            try:
                test_metrics = model.evaluate(X_test, y_test)
                print(f"  Test       - Accuracy: {test_metrics['accuracy']:.4f}, Brier: {test_metrics['brier_score']:.4f}, LogLoss: {test_metrics['log_loss']:.4f}")
            except Exception as e:
                print(f"  Test evaluation error: {e}")
                test_metrics = {'accuracy': 0, 'brier_score': 1, 'log_loss': 10}

            self.results[name] = {
                'train': train_metrics,
                'val': val_metrics,
                'test': test_metrics,
                'model': model
            }

            # Feature importance (if available)
            if hasattr(model, 'get_feature_importance') and model.get_feature_importance() is not None:
                importance_df = model.get_feature_importance(top_n=10)
                if importance_df is not None and len(importance_df) > 0:
                    print(f"\n  Top 10 Features:")
                    for idx, row in importance_df.iterrows():
                        feature_name = str(row['feature']) if row['feature'] is not None else 'unknown'
                        print(f"    {feature_name:<30s}: {row['importance']:.4f}")

    def select_best_model(self):
        """Select best model based on test accuracy"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)

        best_accuracy = 0
        best_name = None

        print(f"\n{'Model':<30s} {'Train Acc':<12} {'Val Acc':<12} {'Test Acc':<12} {'Test Brier':<12}")
        print("-" * 80)

        for name, result in self.results.items():
            train_acc = result['train']['accuracy']
            val_acc = result['val']['accuracy']
            test_acc = result['test']['accuracy']
            test_brier = result['test']['brier_score']

            print(f"{name:<30s} {train_acc:<12.4f} {val_acc:<12.4f} {test_acc:<12.4f} {test_brier:<12.4f}")

            if test_acc > best_accuracy:
                best_accuracy = test_acc
                best_name = name

        print("-" * 80)
        print(f"\n🏆 Best model: {best_name} (Test Accuracy: {best_accuracy:.4f})")

        self.best_model = self.models[best_name]
        self.best_model_name = best_name

        return best_name, best_accuracy

    def save_models(self):
        """Save all trained models"""
        print("\n" + "=" * 70)
        print("SAVING MODELS")
        print("=" * 70)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        for name, model in self.models.items():
            model_path = f"{self.models_dir}/{name}_{timestamp}.joblib"
            joblib.dump(model, model_path)
            print(f"Saved {name} to {model_path}")

        # Save best model
        best_model_path = f"{self.models_dir}/best_enhanced_model.joblib"
        joblib.dump(self.best_model, best_model_path)
        print(f"\n✓ Best model ({self.best_model_name}) saved to {best_model_path}")

        # Save metadata
        metadata = {
            'best_model': self.best_model_name,
            'timestamp': timestamp,
            'results': {name: {
                'train_accuracy': result['train']['accuracy'],
                'val_accuracy': result['val']['accuracy'],
                'test_accuracy': result['test']['accuracy'],
                'test_brier': result['test']['brier_score'],
                'test_log_loss': result['test']['log_loss']
            } for name, result in self.results.items()}
        }

        import json
        with open(f"{self.models_dir}/enhanced_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)

        print(f"Metadata saved to {self.models_dir}/enhanced_metadata.json")

    def run_full_pipeline(self):
        """Run complete enhanced training pipeline"""
        print("=" * 70)
        print("ENHANCED NFL MODEL TRAINING PIPELINE")
        print("=" * 70)

        # Load data
        if not self.load_data():
            return None, None

        # Create splits
        X_train, X_val, X_test, y_train, y_val, y_test = self.create_train_val_test_split()

        # Train models
        success = self.train_all_models(X_train, y_train, X_val, y_val)
        if not success:
            print("\nTraining failed. Please check errors above.")
            return None, None

        # Evaluate models
        self.evaluate_all_models(X_train, X_val, X_test, y_train, y_val, y_test)

        # Select best
        self.select_best_model()

        # Save
        self.save_models()

        print("\n" + "=" * 70)
        print("ENHANCED TRAINING COMPLETE!")
        print("=" * 70)

        return self.best_model, self.results


def main():
    """Main execution"""
    trainer = EnhancedNFLModelTrainer()
    best_model, results = trainer.run_full_pipeline()

    if best_model and results:
        print(f"\n✓ Best enhanced model: {trainer.best_model_name}")
        print(f"✓ Test accuracy: {results[trainer.best_model_name]['test']['accuracy']:.4f}")


if __name__ == "__main__":
    main()
