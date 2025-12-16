"""
Enhanced NFL Prediction Models
Adds XGBoost, LightGBM, and calibrated predictions for improved accuracy
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
import warnings
warnings.filterwarnings('ignore')

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: XGBoost not available. Install with: pip install xgboost")

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    print("Warning: LightGBM not available. Install with: pip install lightgbm")


class BaseNFLPredictor:
    """Enhanced base class with calibration support"""

    def __init__(self, name="Base Model", calibrate=True):
        self.name = name
        self.model = None
        self.scaler = StandardScaler()
        self.feature_columns = None
        self.is_trained = False
        self.calibrate = calibrate
        self.calibrated_model = None

    def get_feature_columns(self, df):
        """Get feature columns (exclude metadata and target)"""
        exclude_cols = [
            'game_id', 'season', 'week', 'home_team', 'away_team',
            'gameday', 'home_won', 'score_diff'
        ]
        return [col for col in df.columns if col not in exclude_cols]

    def prepare_data(self, df):
        """Prepare features and target for training"""
        if self.feature_columns is None:
            self.feature_columns = self.get_feature_columns(df)

        X = df[self.feature_columns].fillna(0)

        # Handle infinite values
        X = X.replace([np.inf, -np.inf], 0)

        y = df['home_won'].values

        return X, y

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train the model with optional calibration"""
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)

        # Train base model
        self.model.fit(X_train_scaled, y_train)

        # Calibrate if requested
        if self.calibrate and X_val is not None and y_val is not None:
            print(f"  Calibrating {self.name}...")
            X_val_scaled = self.scaler.transform(X_val)
            try:
                # Try newer sklearn API (cv='prefit')
                self.calibrated_model = CalibratedClassifierCV(
                    self.model,
                    method='isotonic',
                    cv='prefit'
                )
                self.calibrated_model.fit(X_val_scaled, y_val)
            except (ValueError, TypeError):
                # Fallback for older sklearn versions
                print(f"  Note: Using cv=2 for calibration (sklearn compatibility)")
                # Use small cv for faster calibration
                self.calibrated_model = CalibratedClassifierCV(
                    self.model,
                    method='isotonic',
                    cv=2
                )
                # Combine train and val for calibration
                import numpy as np
                X_combined = np.vstack([X_train_scaled, X_val_scaled])
                y_combined = np.hstack([y_train, y_val])
                self.calibrated_model.fit(X_combined, y_combined)

        self.is_trained = True
        print(f"{self.name} training complete")

    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)

        if self.calibrated_model:
            return self.calibrated_model.predict(X_scaled)
        return self.model.predict(X_scaled)

    def predict_proba(self, X):
        """Get prediction probabilities"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        X_scaled = self.scaler.transform(X)

        if self.calibrated_model:
            return self.calibrated_model.predict_proba(X_scaled)
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]

        accuracy = (predictions == y_test).mean()
        brier_score = np.mean((probabilities - y_test) ** 2)

        # Log loss
        log_loss = -np.mean(
            y_test * np.log(probabilities + 1e-15) +
            (1 - y_test) * np.log(1 - probabilities + 1e-15)
        )

        return {
            'accuracy': accuracy,
            'brier_score': brier_score,
            'log_loss': log_loss
        }

    def get_feature_importance(self, top_n=20):
        """Get feature importance if available"""
        if not hasattr(self.model, 'feature_importances_'):
            return None

        importance_df = pd.DataFrame({
            'feature': self.feature_columns,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False).head(top_n)

        return importance_df


class XGBoostPredictor(BaseNFLPredictor):
    """XGBoost Classifier - typically highest accuracy"""

    def __init__(self, calibrate=True):
        super().__init__(name="XGBoost", calibrate=calibrate)

        if not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost not available")

        self.model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            min_child_weight=3,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            eval_metric='logloss'
        )


class LightGBMPredictor(BaseNFLPredictor):
    """LightGBM Classifier - fast and accurate"""

    def __init__(self, calibrate=True):
        super().__init__(name="LightGBM", calibrate=calibrate)

        if not LIGHTGBM_AVAILABLE:
            raise ImportError("LightGBM not available")

        self.model = lgb.LGBMClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbose=-1
        )


class EnhancedRandomForestPredictor(BaseNFLPredictor):
    """Enhanced Random Forest with better hyperparameters"""

    def __init__(self, calibrate=True):
        super().__init__(name="Enhanced Random Forest", calibrate=calibrate)
        self.model = RandomForestClassifier(
            n_estimators=300,
            max_depth=20,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            oob_score=True,
            random_state=42,
            n_jobs=-1
        )


class EnhancedGradientBoostingPredictor(BaseNFLPredictor):
    """Enhanced Gradient Boosting"""

    def __init__(self, calibrate=True):
        super().__init__(name="Enhanced Gradient Boosting", calibrate=calibrate)
        self.model = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            min_samples_split=5,
            min_samples_leaf=2,
            subsample=0.8,
            max_features='sqrt',
            random_state=42
        )


class EnhancedNeuralNetPredictor(BaseNFLPredictor):
    """Enhanced Neural Network"""

    def __init__(self, calibrate=True):
        super().__init__(name="Enhanced Neural Network", calibrate=calibrate)
        self.model = MLPClassifier(
            hidden_layer_sizes=(128, 64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.0001,
            batch_size=64,
            learning_rate='adaptive',
            learning_rate_init=0.001,
            max_iter=1000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.15,
            n_iter_no_change=20
        )


class SuperEnsemblePredictor(BaseNFLPredictor):
    """
    Super ensemble combining all models
    Uses weighted voting based on validation performance
    """

    def __init__(self, models=None, calibrate=True):
        super().__init__(name="Super Ensemble", calibrate=calibrate)

        if models is None:
            # Create default ensemble with all available models
            self.models = []

            if XGBOOST_AVAILABLE:
                self.models.append(('xgb', XGBoostPredictor(calibrate=False)))

            if LIGHTGBM_AVAILABLE:
                self.models.append(('lgb', LightGBMPredictor(calibrate=False)))

            self.models.extend([
                ('rf', EnhancedRandomForestPredictor(calibrate=False)),
                ('gb', EnhancedGradientBoostingPredictor(calibrate=False)),
                ('nn', EnhancedNeuralNetPredictor(calibrate=False))
            ])
        else:
            self.models = models

        self.model_weights = None

    def train(self, X_train, y_train, X_val=None, y_val=None):
        """Train all models and calculate optimal weights"""
        print(f"Training {len(self.models)} models in super ensemble...")

        # Train each model
        for name, model in self.models:
            model.feature_columns = self.feature_columns
            model.train(X_train, y_train, X_val, y_val)

        # Calculate weights based on validation performance
        if X_val is not None and y_val is not None:
            print("  Calculating optimal ensemble weights...")
            accuracies = []

            for name, model in self.models:
                val_metrics = model.evaluate(X_val, y_val)
                accuracies.append(val_metrics['accuracy'])
                print(f"    {model.name}: {val_metrics['accuracy']:.4f}")

            # Convert to softmax weights (models with higher accuracy get more weight)
            accuracies = np.array(accuracies)
            exp_acc = np.exp((accuracies - accuracies.max()) * 10)
            self.model_weights = exp_acc / exp_acc.sum()

            print(f"  Ensemble weights: {dict(zip([m[0] for m in self.models], self.model_weights))}")
        else:
            # Equal weights
            self.model_weights = np.ones(len(self.models)) / len(self.models)

        self.is_trained = True
        print("Super ensemble training complete")

    def predict_proba(self, X):
        """Weighted average of predictions from all models"""
        if not self.is_trained:
            raise ValueError("Models must be trained before making predictions")

        probabilities = []
        for name, model in self.models:
            probabilities.append(model.predict_proba(X))

        # Weighted average
        probabilities = np.array(probabilities)
        weighted_probs = np.average(probabilities, axis=0, weights=self.model_weights)

        return weighted_probs

    def predict(self, X):
        """Make binary predictions based on weighted probabilities"""
        probabilities = self.predict_proba(X)
        return (probabilities[:, 1] >= 0.5).astype(int)

    def evaluate(self, X_test, y_test):
        """Evaluate ensemble performance"""
        predictions = self.predict(X_test)
        probabilities = self.predict_proba(X_test)[:, 1]

        accuracy = (predictions == y_test).mean()
        brier_score = np.mean((probabilities - y_test) ** 2)
        log_loss = -np.mean(
            y_test * np.log(probabilities + 1e-15) +
            (1 - y_test) * np.log(1 - probabilities + 1e-15)
        )

        return {
            'accuracy': accuracy,
            'brier_score': brier_score,
            'log_loss': log_loss
        }


def create_enhanced_models(calibrate=True):
    """
    Create instances of all enhanced models

    Args:
        calibrate: Whether to calibrate probability predictions

    Returns:
        Dictionary of model name to model instance
    """
    models = {
        'enhanced_random_forest': EnhancedRandomForestPredictor(calibrate=calibrate),
        'enhanced_gradient_boosting': EnhancedGradientBoostingPredictor(calibrate=calibrate),
        'enhanced_neural_network': EnhancedNeuralNetPredictor(calibrate=calibrate),
    }

    if XGBOOST_AVAILABLE:
        models['xgboost'] = XGBoostPredictor(calibrate=calibrate)

    if LIGHTGBM_AVAILABLE:
        models['lightgbm'] = LightGBMPredictor(calibrate=calibrate)

    models['super_ensemble'] = SuperEnsemblePredictor(calibrate=calibrate)

    return models
