"""
Model Failure Prediction System (MFPS)
======================================
An AI-based system to predict whether a trained ML model is stable or failure-prone
before deployment, with Green AI sustainability tracking.

Author: ML Engineering Team
Version: 1.0
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    log_loss, brier_score_loss, confusion_matrix, roc_auc_score,
    classification_report
)
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from scipy.stats import entropy, ks_2samp
from scipy.spatial.distance import jensenshannon
import warnings
import pickle
import json
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """System configuration"""
    RANDOM_SEED = 42
    N_SAMPLES = 1000
    N_FEATURES = 20
    TEST_SIZE = 0.3
    
    # Risk thresholds
    SAFE_THRESHOLD = 0.3
    MONITOR_THRESHOLD = 0.7
    
    # Meta-model parameters
    META_ESTIMATORS = 200
    META_MAX_DEPTH = 10
    META_MIN_SAMPLES_SPLIT = 10
    
    # Green AI metrics
    KWH_PER_RETRAIN = 50
    CO2_PER_KWH = 0.5
    COST_PER_KWH = 0.10

# ============================================================================
# MODEL SIMULATOR - Generate Diverse Model Scenarios
# ============================================================================

class ModelSimulator:
    """
    Simulates different types of models to create training data for meta-model.
    
    Model Types:
    - HEALTHY: Well-balanced, properly regularized
    - OVERFIT: High training accuracy, poor generalization
    - UNDERFIT: Poor performance on both train and test
    - DRIFT: Covariate shift between train and test
    - IMBALANCED: Trained on imbalanced data without handling
    - NOISY: Trained on data with label noise
    """
    
    def __init__(self, n_samples=Config.N_SAMPLES, n_features=Config.N_FEATURES):
        self.n_samples = n_samples
        self.n_features = n_features
    
    def create_healthy_model(self):
        """Create a healthy, well-performing model"""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=15,
            n_redundant=3,
            n_classes=2,
            weights=[0.5, 0.5],
            flip_y=0.05,
            random_state=np.random.randint(0, 10000)
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
        )
        
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=Config.RANDOM_SEED)
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label': 0,  # Not failure-prone
            'type': 'HEALTHY'
        }
    
    def create_overfit_model(self):
        """Create an overfitted model with high train-test gap"""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=5,
            n_redundant=10,
            n_classes=2,
            random_state=np.random.randint(0, 10000)
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
        )
        
        # Use decision tree with no max_depth to encourage overfitting
        model = DecisionTreeClassifier(
            max_depth=None,
            min_samples_split=2,
            random_state=Config.RANDOM_SEED
        )
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label': 1,  # Failure-prone
            'type': 'OVERFIT'
        }
    
    def create_underfit_model(self):
        """Create an underfitted model with poor performance"""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=18,
            n_redundant=0,
            n_classes=2,
            random_state=np.random.randint(0, 10000)
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
        )
        
        # Very simple model that will underfit
        model = LogisticRegression(C=0.001, max_iter=10, random_state=Config.RANDOM_SEED)
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label': 1,  # Failure-prone
            'type': 'UNDERFIT'
        }
    
    def create_drift_model(self):
        """Create a model experiencing covariate shift (data drift)"""
        X_train, y_train = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=15,
            n_classes=2,
            random_state=np.random.randint(0, 10000)
        )
        
        # Create drifted test set by shifting distribution
        X_test, y_test = make_classification(
            n_samples=int(self.n_samples * Config.TEST_SIZE),
            n_features=self.n_features,
            n_informative=15,
            n_classes=2,
            random_state=np.random.randint(0, 10000)
        )
        
        # Add significant drift
        drift_magnitude = np.random.uniform(1.5, 3.0)
        X_test = X_test + np.random.normal(drift_magnitude, 0.5, X_test.shape)
        
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=Config.RANDOM_SEED)
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label': 1,  # Failure-prone
            'type': 'DRIFT'
        }
    
    def create_imbalanced_model(self):
        """Create a model trained on severely imbalanced data"""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=15,
            n_classes=2,
            weights=[0.9, 0.1],  # 90:10 imbalance
            flip_y=0.02,
            random_state=np.random.randint(0, 10000)
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
        )
        
        # Train without handling imbalance
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=Config.RANDOM_SEED)
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label': 1,  # Failure-prone
            'type': 'IMBALANCED'
        }
    
    def create_noisy_model(self):
        """Create a model trained on data with high label noise"""
        X, y = make_classification(
            n_samples=self.n_samples,
            n_features=self.n_features,
            n_informative=15,
            n_classes=2,
            flip_y=0.3,  # 30% label noise
            random_state=np.random.randint(0, 10000)
        )
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=Config.TEST_SIZE, random_state=Config.RANDOM_SEED
        )
        
        model = LogisticRegression(C=1.0, max_iter=1000, random_state=Config.RANDOM_SEED)
        model.fit(X_train, y_train)
        
        return {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'label': 1,  # Failure-prone
            'type': 'NOISY'
        }

# ============================================================================
# META-FEATURE EXTRACTOR
# ============================================================================

class MetaFeatureExtractor:
    """
    Extracts health indicators (meta-features) from trained models.
    
    Feature Categories:
    - Performance Metrics: train-test gap, accuracy metrics
    - Confidence Statistics: mean, variance, entropy
    - Drift Indicators: distribution shifts
    - Stability Metrics: robustness to noise
    - Calibration: prediction reliability
    """
    
    def extract_features(self, model_info):
        """Extract all meta-features from a model"""
        model = model_info['model']
        X_train = model_info.get('X_train')
        X_test = model_info.get('X_test')
        y_train = model_info.get('y_train')
        y_test = model_info.get('y_test')
        
        has_data = all(v is not None for v in [X_train, X_test, y_train, y_test])
        
        features = {}
        
        # ===== A. Performance Metrics =====
        if has_data:
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_acc = accuracy_score(y_train, y_train_pred)
            test_acc = accuracy_score(y_test, y_test_pred)
            
            features['train_test_gap'] = abs(train_acc - test_acc)
            features['test_accuracy'] = test_acc
            features['train_accuracy'] = train_acc
        else:
            # Defaults for data-less analysis
            features['train_test_gap'] = 0.0
            features['test_accuracy'] = 0.5
            features['train_accuracy'] = 0.5
            features['WARNING_DATA_MISSING'] = 1.0
        
        # Precision and recall
        try:
            test_precision = precision_score(y_test, y_test_pred, zero_division=0)
            test_recall = recall_score(y_test, y_test_pred, zero_division=0)
            features['precision_recall_gap'] = abs(test_precision - test_recall)
        except:
            features['precision_recall_gap'] = 0.0
        
        # ===== B. Confidence Statistics =====
        try:
            y_train_proba = model.predict_proba(X_train)[:, 1]
            y_test_proba = model.predict_proba(X_test)[:, 1]
            
            features['mean_confidence'] = np.mean(y_test_proba)
            features['confidence_variance'] = np.var(y_test_proba)
            features['confidence_std'] = np.std(y_test_proba)
            
            # Prediction entropy
            eps = 1e-10
            proba_full = model.predict_proba(X_test)
            entropies = -np.sum(proba_full * np.log(proba_full + eps), axis=1)
            features['prediction_entropy'] = np.mean(entropies)
            
            # Overconfidence rate (predictions > 0.95 or < 0.05)
            features['overconfidence_rate'] = np.mean(
                (y_test_proba > 0.95) | (y_test_proba < 0.05)
            )
        except:
            features['mean_confidence'] = 0.5
            features['confidence_variance'] = 0.0
            features['confidence_std'] = 0.0
            features['prediction_entropy'] = 0.0
            features['overconfidence_rate'] = 0.0
        
        # ===== C. Drift Indicators =====
        try:
            if has_data:
                eps = 1e-10
                # Feature distribution drift
                train_mean = np.mean(X_train, axis=0)
                test_mean = np.mean(X_test, axis=0)
                train_std = np.std(X_train, axis=0) + eps
                test_std = np.std(X_test, axis=0) + eps
                
                # Normalized mean difference
                features['feature_drift_score'] = np.mean(
                    np.abs(train_mean - test_mean) / ((train_std + test_std) / 2)
                )
                
                # Prediction distribution drift
                train_proba_hist = np.histogram(y_train_proba, bins=10, range=(0, 1))[0] + eps
                test_proba_hist = np.histogram(y_test_proba, bins=10, range=(0, 1))[0] + eps
                
                # Normalize
                train_proba_hist = train_proba_hist / train_proba_hist.sum()
                test_proba_hist = test_proba_hist / test_proba_hist.sum()
                
                features['prediction_drift'] = jensenshannon(train_proba_hist, test_proba_hist)
            else:
                features['feature_drift_score'] = 0.0
                features['prediction_drift'] = 0.0
        except:
            features['feature_drift_score'] = 0.0
            features['prediction_drift'] = 0.0
        
        # KS test for covariate shift (on first 3 features)
        ks_scores = []
        try:
            if has_data:
                for i in range(min(3, X_train.shape[1])):
                    try:
                        ks_stat, _ = ks_2samp(X_train[:, i], X_test[:, i])
                        ks_scores.append(ks_stat)
                    except:
                        pass
        except:
            pass
        features['ks_statistic'] = np.mean(ks_scores) if ks_scores else 0.0
        
        # ===== D. Stability Metrics =====
        # Noise robustness
        try:
            if has_data:
                X_test_noisy = X_test + np.random.normal(0, 0.1, X_test.shape)
                y_test_noisy_pred = model.predict(X_test_noisy)
                noisy_acc = accuracy_score(y_test, y_test_noisy_pred)
                features['noise_robustness'] = features['test_accuracy'] - noisy_acc
            else:
                features['noise_robustness'] = 0.0
        except:
            features['noise_robustness'] = 0.0
        
        # Cross-validation stability
        try:
            if has_data:
                cv_scores = cross_val_score(
                    model, X_train, y_train, cv=3, scoring='accuracy'
                )
                features['cv_std'] = np.std(cv_scores)
                features['cv_mean'] = np.mean(cv_scores)
            else:
                features['cv_std'] = 0.0
                features['cv_mean'] = features.get('train_accuracy', 0.5)
        except:
            features['cv_std'] = 0.0
            features['cv_mean'] = features.get('train_accuracy', 0.5)
        
        # ===== E. Calibration Metrics =====
        try:
            if has_data:
                features['brier_score'] = brier_score_loss(y_test, y_test_proba)
                features['log_loss'] = log_loss(y_test, model.predict_proba(X_test))
            else:
                features['brier_score'] = 0.25
                features['log_loss'] = 1.0
        except:
            features['brier_score'] = 0.25
            features['log_loss'] = 1.0
        
        return features

# ============================================================================
# MODEL FAILURE PREDICTION SYSTEM (Main Component)
# ============================================================================

class ModelFailurePredictionSystem:
    """
    Main system for predicting model failures before deployment.
    
    Workflow:
    1. Generate training data (simulate various model types)
    2. Extract meta-features from each model
    3. Train meta-classifier (Random Forest)
    4. Predict failure risk for new models
    5. Provide explanations and recommendations
    """
    
    def __init__(self):
        self.meta_model = None
        self.feature_names = None
        self.training_stats = {}
    
    def generate_training_data(self, n_models_per_type=50):
        """
        Generate synthetic training data for meta-model.
        
        Args:
            n_models_per_type: Number of models to generate per archetype
            
        Returns:
            X_meta: Meta-features DataFrame
            y_meta: Failure labels
            model_types: List of model type names
        """
        simulator = ModelSimulator()
        extractor = MetaFeatureExtractor()
        
        meta_features_list = []
        labels = []
        model_types = []
        
        print(f"{'='*70}")
        print(f"Generating Meta-Training Data")
        print(f"{'='*70}")
        print(f"Models per type: {n_models_per_type}")
        
        model_functions = [
            ('HEALTHY', simulator.create_healthy_model),
            ('OVERFIT', simulator.create_overfit_model),
            ('UNDERFIT', simulator.create_underfit_model),
            ('DRIFT', simulator.create_drift_model),
            ('IMBALANCED', simulator.create_imbalanced_model),
            ('NOISY', simulator.create_noisy_model),
        ]
        
        total_models = len(model_functions) * n_models_per_type
        current = 0
        
        for model_type, model_func in model_functions:
            print(f"\nGenerating {model_type} models...", end=' ')
            for i in range(n_models_per_type):
                try:
                    model_info = model_func()
                    features = extractor.extract_features(model_info)
                    
                    meta_features_list.append(features)
                    labels.append(model_info['label'])
                    model_types.append(model_info['type'])
                    
                    current += 1
                    if (current) % 10 == 0:
                        print(f"{current}/{total_models}", end=' ')
                except Exception as e:
                    print(f"Error: {e}", end=' ')
                    continue
            print("✓")
        
        # Convert to DataFrame
        X_meta = pd.DataFrame(meta_features_list)
        y_meta = np.array(labels)
        
        print(f"\n{'─'*70}")
        print(f"Generated {len(X_meta)} model samples")
        print(f"Features: {len(X_meta.columns)}")
        print(f"Failure rate: {np.mean(y_meta):.2%}")
        print(f"{'='*70}\n")
        
        return X_meta, y_meta, model_types
    
    def train_meta_model(self, X_meta, y_meta):
        """
        Train the meta-classifier to predict model failures.
        
        Args:
            X_meta: Meta-features
            y_meta: Failure labels
            
        Returns:
            Dictionary with training results and metrics
        """
        print(f"{'='*70}")
        print(f"Training Meta-Model")
        print(f"{'='*70}")
        
        self.feature_names = X_meta.columns.tolist()
        
        # Split meta-data
        X_train, X_test, y_train, y_test = train_test_split(
            X_meta, y_meta, test_size=0.2, random_state=Config.RANDOM_SEED, stratify=y_meta
        )
        
        print(f"Training samples: {len(X_train)}")
        print(f"Test samples: {len(X_test)}")
        
        # Train Random Forest meta-classifier
        self.meta_model = RandomForestClassifier(
            n_estimators=Config.META_ESTIMATORS,
            max_depth=Config.META_MAX_DEPTH,
            min_samples_split=Config.META_MIN_SAMPLES_SPLIT,
            min_samples_leaf=5,
            class_weight='balanced',
            random_state=Config.RANDOM_SEED,
            n_jobs=-1
        )
        
        self.meta_model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.meta_model.predict(X_test)
        y_proba = self.meta_model.predict_proba(X_test)[:, 1]
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc_roc = roc_auc_score(y_test, y_proba)
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"\n{'─'*70}")
        print(f"Meta-Model Performance:")
        print(f"{'─'*70}")
        print(f"Accuracy:  {accuracy:.3f}")
        print(f"Precision: {precision:.3f}")
        print(f"Recall:    {recall:.3f}")
        print(f"F1-Score:  {f1:.3f}")
        print(f"AUC-ROC:   {auc_roc:.3f}")
        
        print(f"\nConfusion Matrix:")
        print(f"                 Predicted")
        print(f"              PASS    FAIL")
        print(f"Actual PASS    {cm[0][0]:3d}     {cm[0][1]:3d}")
        print(f"       FAIL    {cm[1][0]:3d}     {cm[1][1]:3d}")
        
        # Feature importance
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': self.meta_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\n{'─'*70}")
        print(f"Top 10 Most Important Features:")
        print(f"{'─'*70}")
        for idx, row in importance_df.head(10).iterrows():
            print(f"{row['feature']:30s} {row['importance']:.4f}")
        
        print(f"{'='*70}\n")
        
        # Store results
        self.training_stats = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc_roc': auc_roc,
            'confusion_matrix': cm.tolist(),
            'feature_importance': importance_df.to_dict('records'),
            'train_date': datetime.now().isoformat()
        }
        
        return self.training_stats
    
    def predict_failure(self, model_info):
        """
        Predict if a model will fail in production.
        
        Args:
            model_info: Dictionary with model and data
            
        Returns:
            Dictionary with prediction results
        """
        if self.meta_model is None:
            raise ValueError("Meta-model not trained. Call train_meta_model first.")
        
        extractor = MetaFeatureExtractor()
        features = extractor.extract_features(model_info)
        
        # Convert to DataFrame
        X_new = pd.DataFrame([features])[self.feature_names]
        
        # Predict
        fail_prob = self.meta_model.predict_proba(X_new)[0][1]
        prediction = self.meta_model.predict(X_new)[0]
        
        # Risk categorization
        if fail_prob < Config.SAFE_THRESHOLD:
            risk_level = "SAFE"
            recommendation = "✅ Auto-Deploy"
            action = "DEPLOY"
        elif fail_prob < Config.MONITOR_THRESHOLD:
            risk_level = "MONITOR"
            recommendation = "⚠️  Deploy with Monitoring"
            action = "DEPLOY_WITH_MONITORING"
        else:
            risk_level = "FAILURE LIKELY"
            recommendation = "❌ Block Deployment"
            action = "BLOCK"
        
        return {
            'failure_probability': fail_prob,
            'prediction': "FAIL" if prediction == 1 else "PASS",
            'risk_level': risk_level,
            'recommendation': recommendation,
            'action': action,
            'features': features,
            'timestamp': datetime.now().isoformat()
        }
    
    def explain_prediction(self, model_info, top_n=5):
        """
        Explain why a model is predicted to fail.
        
        Args:
            model_info: Dictionary with model and data
            top_n: Number of top features to show
            
        Returns:
            List of (feature, value, contribution) tuples
        """
        result = self.predict_failure(model_info)
        features = result['features']
        fail_prob = result['failure_probability']
        
        # Get feature importance from meta-model
        feature_importance = dict(zip(
            self.feature_names,
            self.meta_model.feature_importances_
        ))
        
        # Calculate contributions (value × importance)
        contributions = {
            k: features[k] * feature_importance[k]
            for k in self.feature_names
        }
        
        # Sort by absolute contribution
        sorted_contrib = sorted(
            contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True
        )[:top_n]
        
        # Print explanation
        print(f"\n{'='*70}")
        print(f"FAILURE PREDICTION EXPLANATION")
        print(f"{'='*70}")
        print(f"Failure Probability: {fail_prob:.2%}")
        print(f"Risk Level: {result['risk_level']}")
        print(f"Recommendation: {result['recommendation']}")
        
        print(f"\n{'─'*70}")
        print(f"Top {top_n} Risk Factors:")
        print(f"{'─'*70}")
        print(f"{'Feature':<30} {'Value':>10} {'Weight':>10}")
        print(f"{'─'*70}")
        
        for feat, contrib in sorted_contrib:
            value = features[feat]
            print(f"{feat:<30} {value:>10.4f} {contrib:>10.4f}")
        
        print(f"{'='*70}\n")
        
        return sorted_contrib
    
    def save_model(self, filepath='mfps_model.pkl'):
        """Save trained meta-model to disk"""
        model_data = {
            'meta_model': self.meta_model,
            'feature_names': self.feature_names,
            'training_stats': self.training_stats,
            'config': {
                'safe_threshold': Config.SAFE_THRESHOLD,
                'monitor_threshold': Config.MONITOR_THRESHOLD
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        print(f"Model saved to: {filepath}")
    
    def load_model(self, filepath='mfps_model.pkl'):
        """Load trained meta-model from disk"""
        if filepath.endswith('.joblib'):
            import joblib
            model_data = joblib.load(filepath)
        else:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
        
        self.meta_model = model_data['meta_model']
        self.feature_names = model_data['feature_names']
        self.training_stats = model_data['training_stats']
        
        print(f"Model loaded from: {filepath}")

# ============================================================================
# FAIL-SAFE DEPLOYMENT GATE
# ============================================================================

class FailSafeDeploymentGate:
    """
    Multi-layer defense system for model deployment decisions.
    
    Layers:
    1. Meta-model prediction
    2. Hard threshold checks on critical metrics
    3. Human-in-the-loop for edge cases
    4. Final deployment decision
    """
    
    def __init__(self, mfps):
        self.mfps = mfps
        self.human_review_threshold = 0.5
    
    def evaluate_model(self, model_info):
        """
        Comprehensive model evaluation with multiple safety checks.
        
        Args:
            model_info: Dictionary with model and data
            
        Returns:
            Dictionary with deployment decision and reasoning
        """
        # Layer 1: Meta-model prediction
        result = self.mfps.predict_failure(model_info)
        fail_prob = result['failure_probability']
        features = result['features']
        
        # Layer 2: Hard threshold checks
        hard_failures = []
        warnings = []
        
        # Critical failures
        if features['train_test_gap'] > 0.20:
            hard_failures.append(f"❌ Excessive overfitting (gap: {features['train_test_gap']:.3f})")
        
        if features['feature_drift_score'] > 2.0:
            hard_failures.append(f"❌ Severe data drift (score: {features['feature_drift_score']:.3f})")
        
        if features['test_accuracy'] < 0.60:
            hard_failures.append(f"❌ Unacceptable test accuracy ({features['test_accuracy']:.3f})")
        
        # Warnings
        if features['noise_robustness'] > 0.15:
            warnings.append(f"⚠️  Low noise robustness (drop: {features['noise_robustness']:.3f})")
        
        if features['confidence_variance'] > 0.15:
            warnings.append(f"⚠️  High confidence instability (var: {features['confidence_variance']:.3f})")
        
        # Layer 3: Human-in-the-Loop decision
        requires_human_review = (
            fail_prob > self.human_review_threshold or
            len(hard_failures) > 0 or
            features['test_accuracy'] < 0.70
        )
        
        # Layer 4: Final decision
        if hard_failures:
            decision = 'BLOCK'
            auto_deploy = False
            reason = "Critical failures detected"
        elif fail_prob < Config.SAFE_THRESHOLD:
            decision = 'DEPLOY'
            auto_deploy = True
            reason = "Model passed all safety checks"
        elif fail_prob < Config.MONITOR_THRESHOLD:
            decision = 'DEPLOY_WITH_MONITORING'
            auto_deploy = not requires_human_review
            reason = "Moderate risk - requires monitoring"
        else:
            decision = 'BLOCK'
            auto_deploy = False
            reason = "High failure risk detected"
        
        return {
            'decision': decision,
            'auto_deploy': auto_deploy,
            'requires_human_review': requires_human_review,
            'failure_probability': fail_prob,
            'risk_level': result['risk_level'],
            'hard_failures': hard_failures,
            'warnings': warnings,
            'reason': reason,
            'timestamp': datetime.now().isoformat()
        }
    
    def print_decision(self, evaluation):
        """Pretty print deployment decision"""
        print(f"\n{'='*70}")
        print(f"DEPLOYMENT DECISION")
        print(f"{'='*70}")
        print(f"Decision: {evaluation['decision']}")
        print(f"Failure Probability: {evaluation['failure_probability']:.2%}")
        print(f"Risk Level: {evaluation['risk_level']}")
        print(f"Auto-Deploy: {'Yes' if evaluation['auto_deploy'] else 'No'}")
        print(f"Human Review Required: {'Yes' if evaluation['requires_human_review'] else 'No'}")
        print(f"\nReason: {evaluation['reason']}")
        
        if evaluation['hard_failures']:
            print(f"\n{'─'*70}")
            print(f"Critical Failures:")
            for failure in evaluation['hard_failures']:
                print(f"  {failure}")
        
        if evaluation['warnings']:
            print(f"\n{'─'*70}")
            print(f"Warnings:")
            for warning in evaluation['warnings']:
                print(f"  {warning}")
        
        print(f"{'='*70}\n")

# ============================================================================
# GREEN AI IMPACT CALCULATOR
# ============================================================================

class GreenAIImpactCalculator:
    """
    Calculate environmental and cost impact of the MFPS system.
    
    Metrics:
    - Energy savings (kWh)
    - CO2 emissions avoided (kg)
    - Cost savings (USD)
    - Failed deployments prevented
    """
    
    def calculate_impact(self, total_models, baseline_failure_rate=0.40, system_failure_rate=0.10):
        """
        Calculate Green AI impact metrics.
        
        Args:
            total_models: Total number of models evaluated
            baseline_failure_rate: Failure rate without MFPS
            system_failure_rate: Failure rate with MFPS
            
        Returns:
            Dictionary with impact metrics
        """
        # Models saved from failing
        models_saved = total_models * (baseline_failure_rate - system_failure_rate)
        
        # Energy calculations
        energy_saved_kwh = models_saved * Config.KWH_PER_RETRAIN
        co2_saved_kg = energy_saved_kwh * Config.CO2_PER_KWH
        co2_saved_tons = co2_saved_kg / 1000
        
        # Cost calculations
        energy_cost_saved = energy_saved_kwh * Config.COST_PER_KWH
        
        # Carbon tax (assuming $50/ton)
        carbon_tax_saved = co2_saved_tons * 50
        
        total_cost_saved = energy_cost_saved + carbon_tax_saved
        
        return {
            'total_models': total_models,
            'models_saved': models_saved,
            'energy_saved_kwh': energy_saved_kwh,
            'co2_saved_kg': co2_saved_kg,
            'co2_saved_tons': co2_saved_tons,
            'energy_cost_saved_usd': energy_cost_saved,
            'carbon_tax_saved_usd': carbon_tax_saved,
            'total_cost_saved_usd': total_cost_saved,
            'baseline_failure_rate': baseline_failure_rate,
            'system_failure_rate': system_failure_rate
        }
    
    def print_impact(self, impact):
        """Pretty print Green AI impact"""
        print(f"\n{'='*70}")
        print(f"GREEN AI IMPACT ANALYSIS")
        print(f"{'='*70}")
        print(f"Total Models Evaluated: {impact['total_models']:.0f}")
        print(f"Baseline Failure Rate: {impact['baseline_failure_rate']:.1%}")
        print(f"System Failure Rate: {impact['system_failure_rate']:.1%}")
        
        print(f"\n{'─'*70}")
        print(f"Environmental Impact:")
        print(f"{'─'*70}")
        print(f"🌳 Models Prevented from Failing: {impact['models_saved']:.0f}")
        print(f"⚡ Energy Saved: {impact['energy_saved_kwh']:,.0f} kWh")
        print(f"🌍 CO2 Emissions Avoided: {impact['co2_saved_kg']:,.0f} kg ({impact['co2_saved_tons']:.2f} tons)")
        
        print(f"\n{'─'*70}")
        print(f"Financial Impact:")
        print(f"{'─'*70}")
        print(f"💰 Energy Cost Saved: ${impact['energy_cost_saved_usd']:,.2f}")
        print(f"💰 Carbon Tax Saved: ${impact['carbon_tax_saved_usd']:,.2f}")
        print(f"💰 Total Cost Saved: ${impact['total_cost_saved_usd']:,.2f}")
        
        print(f"\n{'─'*70}")
        print(f"Sustainability Benefits:")
        print(f"{'─'*70}")
        print(f"• Reduced computational waste by {(impact['baseline_failure_rate'] - impact['system_failure_rate'])*100:.0f}%")
        print(f"• Equivalent to removing {impact['co2_saved_kg']/400:.1f} cars from the road for a day*")
        print(f"• Equivalent to planting {impact['co2_saved_kg']/20:.0f} trees**")
        print(f"\n* Average car produces 400g CO2/km")
        print(f"** Average tree absorbs 20kg CO2/year")
        print(f"{'='*70}\n")

# ============================================================================
# MAIN DEMONSTRATION
# ============================================================================

def main():
    """Full system demonstration"""
    
    print(f"\n{'#'*70}")
    print(f"#{'MODEL FAILURE PREDICTION SYSTEM (MFPS)':^68}#")
    print(f"#{'AI-Based Pre-Deployment Model Validation':^68}#")
    print(f"#{'With Green AI Sustainability Tracking':^68}#")
    print(f"{'#'*70}\n")
    
    # ===== PHASE 1: Initialize System =====
    print("PHASE 1: System Initialization")
    print("─" * 70)
    mfps = ModelFailurePredictionSystem()
    
    # ===== PHASE 2: Generate Training Data =====
    print("\nPHASE 2: Meta-Model Training")
    print("─" * 70)
    X_meta, y_meta, model_types = mfps.generate_training_data(n_models_per_type=40)
    
    # ===== PHASE 3: Train Meta-Model =====
    results = mfps.train_meta_model(X_meta, y_meta)
    
    # ===== PHASE 4: Test on New Models =====
    print("\nPHASE 3: Testing on New Models")
    print("─" * 70)
    
    simulator = ModelSimulator()
    gate = FailSafeDeploymentGate(mfps)
    
    test_cases = [
        ("Healthy Model", simulator.create_healthy_model()),
        ("Overfit Model", simulator.create_overfit_model()),
        ("Drift Model", simulator.create_drift_model()),
        ("Imbalanced Model", simulator.create_imbalanced_model()),
    ]
    
    for name, model_info in test_cases:
        print(f"\n{'='*70}")
        print(f"Testing: {name} ({model_info['type']})")
        print(f"{'='*70}")
        
        # Predict failure
        result = mfps.predict_failure(model_info)
        
        print(f"\nPrediction Results:")
        print(f"  Failure Probability: {result['failure_probability']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
        print(f"  Recommendation: {result['recommendation']}")
        
        # Get explanation
        mfps.explain_prediction(model_info, top_n=5)
        
        # Deployment decision
        evaluation = gate.evaluate_model(model_info)
        gate.print_decision(evaluation)
    
    # ===== PHASE 5: Green AI Impact =====
    print("\nPHASE 4: Green AI Impact Assessment")
    print("─" * 70)
    
    green_calc = GreenAIImpactCalculator()
    impact = green_calc.calculate_impact(
        total_models=100,
        baseline_failure_rate=0.40,
        system_failure_rate=0.10
    )
    green_calc.print_impact(impact)
    
    # ===== PHASE 6: Save Model =====
    print("\nPHASE 5: Model Persistence")
    print("─" * 70)
    mfps.save_model('mfps_model.pkl')
    
    # Generate summary report
    summary = {
        'system': 'Model Failure Prediction System',
        'version': '1.0',
        'training_date': results['train_date'],
        'meta_model_performance': {
            'accuracy': results['accuracy'],
            'precision': results['precision'],
            'recall': results['recall'],
            'f1_score': results['f1_score'],
            'auc_roc': results['auc_roc']
        },
        'green_ai_impact': impact,
        'configuration': {
            'safe_threshold': Config.SAFE_THRESHOLD,
            'monitor_threshold': Config.MONITOR_THRESHOLD,
            'models_trained_on': len(X_meta)
        }
    }
    
    with open('mfps_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n✅ System ready for deployment!")
    print(f"📊 Summary report saved to: mfps_summary.json")
    print(f"💾 Model saved to: mfps_model.pkl")
    
    print(f"\n{'#'*70}")
    print(f"#{'DEMONSTRATION COMPLETE':^68}#")
    print(f"{'#'*70}\n")

if __name__ == "__main__":
    main()
