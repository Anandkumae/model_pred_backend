"""
MFPS Usage Examples
===================
Demonstrates how to use the Model Failure Prediction System
for different scenarios.
"""

import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from model_failure_prediction_system import (
    ModelFailurePredictionSystem,
    FailSafeDeploymentGate,
    GreenAIImpactCalculator
)

# ============================================================================
# EXAMPLE 1: Quick Start - Evaluate Your Own Model
# ============================================================================

def example_1_evaluate_your_model():
    """Evaluate a custom trained model"""
    print("\n" + "="*70)
    print("EXAMPLE 1: Evaluate Your Own Model")
    print("="*70)
    
    # Step 1: Train your model (example)
    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_informative=15,
        random_state=42
    )
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    
    print("✓ Model trained")
    
    # Step 2: Load pre-trained MFPS
    mfps = ModelFailurePredictionSystem()
    try:
        mfps.load_model('mfps_model.pkl')
        print("✓ MFPS loaded")
    except:
        print("⚠️  MFPS not found. Training new one...")
        X_meta, y_meta, _ = mfps.generate_training_data(n_models_per_type=30)
        mfps.train_meta_model(X_meta, y_meta)
    
    # Step 3: Prepare model info
    model_info = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Step 4: Get prediction
    result = mfps.predict_failure(model_info)
    
    print(f"\n📊 Results:")
    print(f"  Failure Probability: {result['failure_probability']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Recommendation: {result['recommendation']}")
    
    # Step 5: Get explanation
    mfps.explain_prediction(model_info)

# ============================================================================
# EXAMPLE 2: Batch Model Evaluation
# ============================================================================

def example_2_batch_evaluation():
    """Evaluate multiple models at once"""
    print("\n" + "="*70)
    print("EXAMPLE 2: Batch Model Evaluation")
    print("="*70)
    
    # Initialize MFPS
    mfps = ModelFailurePredictionSystem()
    try:
        mfps.load_model('mfps_model.pkl')
    except:
        X_meta, y_meta, _ = mfps.generate_training_data(n_models_per_type=30)
        mfps.train_meta_model(X_meta, y_meta)
    
    # Create multiple models with different configurations
    models_to_test = []
    
    for i, C_value in enumerate([0.001, 0.1, 1.0, 10.0, 100.0]):
        X, y = make_classification(n_samples=1000, n_features=20, random_state=i)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
        
        model = LogisticRegression(C=C_value, max_iter=1000)
        model.fit(X_train, y_train)
        
        models_to_test.append({
            'name': f'Model_C_{C_value}',
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        })
    
    # Evaluate all models
    results = []
    for model_info in models_to_test:
        result = mfps.predict_failure(model_info)
        results.append({
            'name': model_info['name'],
            'fail_prob': result['failure_probability'],
            'risk_level': result['risk_level'],
            'action': result['action']
        })
    
    # Print summary
    print(f"\n{'Model Name':<20} {'Fail Prob':>12} {'Risk Level':<20} {'Action':<20}")
    print("─" * 70)
    for r in results:
        print(f"{r['name']:<20} {r['fail_prob']:>11.2%} {r['risk_level']:<20} {r['action']:<20}")

# ============================================================================
# EXAMPLE 3: CI/CD Integration
# ============================================================================

def example_3_cicd_integration():
    """Simulate CI/CD pipeline integration"""
    print("\n" + "="*70)
    print("EXAMPLE 3: CI/CD Pipeline Integration")
    print("="*70)
    
    # Initialize systems
    mfps = ModelFailurePredictionSystem()
    try:
        mfps.load_model('mfps_model.pkl')
    except:
        X_meta, y_meta, _ = mfps.generate_training_data(n_models_per_type=30)
        mfps.train_meta_model(X_meta, y_meta)
    
    gate = FailSafeDeploymentGate(mfps)
    
    # Simulate model from CI/CD pipeline
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    model_info = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    # Gate evaluation
    print("\n🔍 Running deployment gate checks...")
    evaluation = gate.evaluate_model(model_info)
    gate.print_decision(evaluation)
    
    # Simulate CI/CD decision
    if evaluation['auto_deploy']:
        print("✅ PIPELINE: Proceeding to deployment stage")
        print("   Next: Deploy to production")
    elif evaluation['requires_human_review']:
        print("⏸️  PIPELINE: Pausing for human review")
        print("   Action: Notify MLOps team")
    else:
        print("❌ PIPELINE: Blocking deployment")
        print("   Action: Send model back to development")

# ============================================================================
# EXAMPLE 4: A/B Testing Model Comparison
# ============================================================================

def example_4_ab_testing():
    """Compare two model candidates"""
    print("\n" + "="*70)
    print("EXAMPLE 4: A/B Testing - Model Comparison")
    print("="*70)
    
    # Initialize MFPS
    mfps = ModelFailurePredictionSystem()
    try:
        mfps.load_model('mfps_model.pkl')
    except:
        X_meta, y_meta, _ = mfps.generate_training_data(n_models_per_type=30)
        mfps.train_meta_model(X_meta, y_meta)
    
    # Prepare data
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    
    # Model A: Logistic Regression
    model_a = LogisticRegression(C=1.0, max_iter=1000)
    model_a.fit(X_train, y_train)
    
    # Model B: Random Forest
    model_b = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
    model_b.fit(X_train, y_train)
    
    models = {
        'Model A (LogReg)': model_a,
        'Model B (RF)': model_b
    }
    
    # Compare
    print("\n📊 Comparing Model Candidates:")
    print("─" * 70)
    
    comparison = []
    for name, model in models.items():
        model_info = {
            'model': model,
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test
        }
        
        result = mfps.predict_failure(model_info)
        comparison.append({
            'name': name,
            'fail_prob': result['failure_probability'],
            'risk_level': result['risk_level']
        })
        
        print(f"\n{name}:")
        print(f"  Failure Probability: {result['failure_probability']:.2%}")
        print(f"  Risk Level: {result['risk_level']}")
    
    # Recommendation
    print("\n" + "─" * 70)
    print("RECOMMENDATION:")
    best_model = min(comparison, key=lambda x: x['fail_prob'])
    print(f"✅ Deploy: {best_model['name']}")
    print(f"   Lowest failure risk: {best_model['fail_prob']:.2%}")

# ============================================================================
# EXAMPLE 5: Green AI Impact Tracking
# ============================================================================

def example_5_green_ai_tracking():
    """Track environmental impact over time"""
    print("\n" + "="*70)
    print("EXAMPLE 5: Green AI Impact Tracking")
    print("="*70)
    
    calculator = GreenAIImpactCalculator()
    
    # Scenario 1: Small team (10 models/month)
    print("\n📈 Scenario 1: Small Team (10 models/month)")
    impact_small = calculator.calculate_impact(
        total_models=120,  # 1 year
        baseline_failure_rate=0.35,
        system_failure_rate=0.08
    )
    calculator.print_impact(impact_small)
    
    # Scenario 2: Enterprise (100 models/month)
    print("\n📈 Scenario 2: Enterprise (100 models/month)")
    impact_enterprise = calculator.calculate_impact(
        total_models=1200,  # 1 year
        baseline_failure_rate=0.40,
        system_failure_rate=0.10
    )
    calculator.print_impact(impact_enterprise)

# ============================================================================
# EXAMPLE 6: Custom Risk Thresholds
# ============================================================================

def example_6_custom_thresholds():
    """Use custom risk thresholds for different use cases"""
    print("\n" + "="*70)
    print("EXAMPLE 6: Custom Risk Thresholds")
    print("="*70)
    
    # Initialize MFPS
    mfps = ModelFailurePredictionSystem()
    try:
        mfps.load_model('mfps_model.pkl')
    except:
        X_meta, y_meta, _ = mfps.generate_training_data(n_models_per_type=30)
        mfps.train_meta_model(X_meta, y_meta)
    
    # Create test model
    X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LogisticRegression(C=1.0, max_iter=1000)
    model.fit(X_train, y_train)
    
    model_info = {
        'model': model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test
    }
    
    result = mfps.predict_failure(model_info)
    fail_prob = result['failure_probability']
    
    print(f"\nModel Failure Probability: {fail_prob:.2%}")
    print("\n" + "─" * 70)
    print("Risk Assessment for Different Use Cases:")
    print("─" * 70)
    
    # Different risk profiles
    use_cases = [
        ("Healthcare (Critical)", 0.10, 0.30),  # Very conservative
        ("Finance (High Stakes)", 0.20, 0.50),  # Conservative
        ("E-commerce (Standard)", 0.30, 0.70),  # Moderate
        ("Experimentation (Low Stakes)", 0.50, 0.80)  # Aggressive
    ]
    
    for use_case, safe_thresh, monitor_thresh in use_cases:
        if fail_prob < safe_thresh:
            decision = "✅ DEPLOY"
        elif fail_prob < monitor_thresh:
            decision = "⚠️  MONITOR"
        else:
            decision = "❌ BLOCK"
        
        print(f"{use_case:<30} {decision}")

# ============================================================================
# RUN ALL EXAMPLES
# ============================================================================

if __name__ == "__main__":
    print("\n" + "#"*70)
    print("#" + " "*68 + "#")
    print("#" + "MFPS USAGE EXAMPLES".center(68) + "#")
    print("#" + " "*68 + "#")
    print("#"*70)
    
    examples = [
        ("Evaluate Your Own Model", example_1_evaluate_your_model),
        ("Batch Model Evaluation", example_2_batch_evaluation),
        ("CI/CD Integration", example_3_cicd_integration),
        ("A/B Testing", example_4_ab_testing),
        ("Green AI Tracking", example_5_green_ai_tracking),
        ("Custom Thresholds", example_6_custom_thresholds),
    ]
    
    print("\nAvailable Examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nRunning all examples...\n")
    
    for name, func in examples:
        try:
            func()
        except Exception as e:
            print(f"\n❌ Error in {name}: {e}")
            continue
    
    print("\n" + "#"*70)
    print("#" + "ALL EXAMPLES COMPLETED".center(68) + "#")
    print("#"*70 + "\n")
