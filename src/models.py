"""
NFL Prediction Models
Implements multiple machine learning models for predicting game outcomes
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')


class NFLPredictor:
    """Base class for NFL game prediction models"""

    def __init__(self, name="Base Model"):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False

    def get_feature_columns(self, df):
        """
        Get feature columns (exclude metadata and target)

        Args:
            df: DataFrame with features

        Returns:
            List of feature column names
        """
        exclude_cols = [
            'game_id', 'season', 'week', 'home_team', 'away_team',
            'gameday', 'home_won', 'score_diff'
        ]
        return [col for col in df.columns if col not in exclude_cols]

    def prepare_data(self, df):
        """
        Prepare features and target for training

        Args:
            df: DataFrame with features

        Returns:
            Tuple of (X, y) - features and target
        """
        if self.feature_columns is None:
            self.feature_columns = self.get_feature_columns(df)

        X = df[self.feature_columns].fillna(0)
        y = df['home_won'].values

        return X, y

    def train(self, X_train, y_train):
        """
        Train the model

        Args:
            X_train: Training features
            y_train: Training labels
        """
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train model
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True

        print(f"{self.name} training complete")

    def predict(self, X):
        """
        Make predictions

        Args:
            X: Features to predict on

        Returns:
            Binary predictions (0 or 1)
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """
        Get prediction probabilities

        Args:
            X: Features to predict on

        Returns:
            Probability estimates
        """
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance

        Args:
            X_test: Test features
            y_test: Test labels

        Returns:
            Dictionary of evaluation metrics
        """
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]

        accuracy = (predictions == y_test).mean()

        # Calculate Brier score (lower is better)
        brier_score = np.mean((probabilities - y_test) ** 2)

        return {
            'accuracy': accuracy,
            'brier_score': brier_score
        }

    def get_feature_importance(self, top_n=20):
        """
        Get feature importance if available

        Args:
            top_n: Number of top features to return

        Returns:
            DataFrame with feature importance
        """
        if not hasattr(self.model, 'feature_importances_'):
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df


class RandomForestPredictor(NFLPredictor):
    """Random Forest Classifier for NFL predictions"""

    def __init__(self):
        super().__init__(name="Random Forest")
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=4,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1
        )


class GradientBoostingPredictor(NFLPredictor):
    """Gradient Boosting Classifier for NFL predictions"""

    def __init__(self):
        super().__init__(name="Gradient Boosting")
        self.model = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            min_samples_split=10,
            min_samples_leaf=4,
            subsample=0.8,
            random_state=42
        )


class NeuralNetPredictor(NFLPredictor):
    """Neural Network (MLP) for NFL predictions"""

    def __init__(self):
        super().__init__(name="Neural Network")
        self.model = MLPClassifier(
            hidden_layer_sizes=(100, 50, 25),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )


class LogisticRegressionPredictor(NFLPredictor):
    """Logistic Regression for NFL predictions (baseline)"""

    def __init__(self):
        super().__init__(name="Logistic Regression")
        self.model = LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42
        )


class EnsemblePredictor(NFLPredictor):
    """Ensemble of multiple models"""

    def __init__(self, models=None):
        super().__init__(name="Ensemble")

        if models is None:
            self.models = [
                RandomForestPredictor(),
                GradientBoostingPredictor(),
                NeuralNetPredictor()
            ]
        else:
            self.models = models

    def train(self, X_train, y_train):
        """Train all models in the ensemble"""
        print(f"Training {len(self.models)} models in ensemble...")

        for model in self.models:
            model.feature_columns = self.feature_columns
            model.train(X_train, y_train)

        self.is_trained = True
        print("Ensemble training complete")

    def predict_proba(self, X):
        """Average predictions from all models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")

        probabilities = np.array([model.predict_proba(X) for model in self.models])
        avg_probabilities = probabilities.mean(axis=0)

        return avg_probabilities

    def predict(self, X):
        """Make binary predictions based on averaged probabilities"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)

    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]

        accuracy = (predictions == y_test).mean()
        brier_score = np.mean((probabilities - y_test) ** 2)

        return {
            'accuracy': accuracy,
            'brier_score': brier_score
        }


def create_all_models():
    """
    Create instances of all available models

    Returns:
        Dictionary of model name to model instance
    """
    return {
        'logistic_regression': LogisticRegressionPredictor(),
        'random_forest': RandomForestPredictor(),
        'gradient_boosting': GradientBoostingPredictor(),
        'neural_network': NeuralNetPredictor(),
        'ensemble': EnsemblePredictor()
    }
