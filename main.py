
import pickle
import uvicorn
import traceback
import pandas as pd
import numpy as np
import joblib
from io import BytesIO
from sklearn.base import is_classifier, is_regressor
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from typing import Optional, Dict, Any
from datetime import datetime
from model_failure_prediction_system import ModelFailurePredictionSystem, MetaFeatureExtractor, Config, ModelSimulator

app = FastAPI(title="Model Failure Prediction System API", version="1.0")

# Load model on startup
mfps = ModelFailurePredictionSystem()
try:
    # Switch to joblib for the internal meta-model as well
    mfps.load_model('mfps_model.joblib')
    
    # Validation Layer: Ensure the loaded meta-model is exactly what we expect
    if not is_classifier(mfps.meta_model):
        print("CRITICAL: Meta-model validation failed! Not a classifier.")
        mfps.meta_model = None
    else:
        print(f"SUCCESS: MFPS Meta-Model ({type(mfps.meta_model).__name__}) loaded and validated.")
except Exception as e:
    print(f"Warning: Model could not be loaded: {e}")
    print("Some endpoints may fail. Use '/predict/data' for raw analysis if meta-model is present.")

class SimulationRequest(BaseModel):
    model_type: str  # HEALTHY, OVERFIT, UNDERFIT, DRIFT, IMBALANCED, NOISY

@app.get("/")
def home():
    return {"status": "online", "system": "MFPS", "version": "1.0"}

@app.post("/predict/simulation")
def predict_simulation(request: SimulationRequest):
    """
    Generate a synthetic model of the requested type and analyze it.
    """
    simulator = ModelSimulator()
    model_type = request.model_type.upper()
    
    if model_type == 'HEALTHY':
        model_info = simulator.create_healthy_model()
    elif model_type == 'OVERFIT':
        model_info = simulator.create_overfit_model()
    elif model_type == 'UNDERFIT':
        model_info = simulator.create_underfit_model()
    elif model_type == 'DRIFT':
        model_info = simulator.create_drift_model()
    elif model_type == 'IMBALANCED':
        model_info = simulator.create_imbalanced_model()
    elif model_type == 'NOISY':
        model_info = simulator.create_noisy_model()
    else:
        raise HTTPException(status_code=400, detail="Invalid model type. Choose from: HEALTHY, OVERFIT, UNDERFIT, DRIFT, IMBALANCED, NOISY")
    
    # Predict
    result = mfps.predict_failure(model_info)
    
    # Get feature importance for this specific prediction
    # We need to manually call what explain_prediction does to return data
    features = result['features']
    feature_importance = dict(zip(mfps.feature_names, mfps.meta_model.feature_importances_))
    contributions = {k: features[k] * feature_importance[k] for k in mfps.feature_names}
    sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
    
    # Top risk factors
    top_factors = [
        {"feature": k, "value": features[k], "weight": v} 
        for k, v in sorted_contrib[:10]
    ]

    return {
        "prediction": result,
        "explanation": top_factors,
        "model_type": model_info['type']
    }

@app.post("/predict/data")
def predict_data(features: Dict[str, float]):
    """
    Secure endpoint: Analyze model health using raw meta-features.
    Zero binary ingestion = Zero security risk.
    """
    if mfps.meta_model is None:
        raise HTTPException(status_code=503, detail="Meta-model not loaded on server.")
        
    try:
        # 1. Fill in missing features with defaults if needed (optional)
        # For simplicity, we assume the user provides all 17 features
        # or we could use the extractor's defaults.
        
        # 2. Convert to DataFrame in correct order
        X_new = pd.DataFrame([features])[mfps.feature_names]
        
        # 3. Predict using internal logic (duplicated here for visibility)
        fail_prob = mfps.meta_model.predict_proba(X_new)[0][1]
        prediction = mfps.meta_model.predict(X_new)[0]
        
        # Risk categorization
        if fail_prob < Config.SAFE_THRESHOLD:
            risk_level, rec, action = "SAFE", "✅ Auto-Deploy", "DEPLOY"
        elif fail_prob < Config.MONITOR_THRESHOLD:
            risk_level, rec, action = "MONITOR", "⚠️  Deploy with Monitoring", "DEPLOY_WITH_MONITORING"
        else:
            risk_level, rec, action = "FAILURE LIKELY", "❌ Block Deployment", "BLOCK"
            
        result = {
            'failure_probability': fail_prob,
            'prediction': "FAIL" if prediction == 1 else "PASS",
            'risk_level': risk_level,
            'recommendation': rec,
            'action': action,
            'features': features,
            'timestamp': datetime.now().isoformat()
        }
        
        # 4. Get explanation
        feature_importance = dict(zip(mfps.feature_names, mfps.meta_model.feature_importances_))
        contributions = {k: features.get(k, 0) * feature_importance[k] for k in mfps.feature_names}
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        top_factors = [
            {"feature": k, "value": features.get(k, 0), "weight": v} 
            for k, v in sorted_contrib[:10]
        ]
        
        return {
            "prediction": result,
            "explanation": top_factors,
            "mode": "SECURE_DATA_ONLY"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Feature processing error: {str(e)}")

@app.post("/predict/upload")
async def predict_upload(file: UploadFile = File(...)):
    """
    Analyze an uploaded pickled model_info dictionary.
    WARNING: Only use with trusted files.
    """
    try:
        contents = await file.read()
        print(f"DEBUG: Received file size: {len(contents)} bytes")
        print(f"DEBUG: First 50 bytes: {contents[:50]}")
        
        try:
            # Switch to joblib for more robust sklearn loading
            model_info = joblib.load(BytesIO(contents))
            print("DEBUG: Joblib loaded successfully.")
        except Exception as p_e:
            print(f"DEBUG: Joblib load failed: {str(p_e)}")
            
            # Still attempt CRLF repair just in case it's a raw binary corruption issue
            # joblib might be more sensitive or different about it
            try:
                print("DEBUG: Attempting CRLF -> LF repair (fallback)...")
                fixed_contents = contents.replace(b'\r\n', b'\n')
                if fixed_contents != contents:
                     model_info = joblib.load(BytesIO(fixed_contents))
                     print("DEBUG: Joblib loaded successfully after CRLF repair.")
                else:
                    raise Exception("No CRLF corruption found to repair")
            except Exception as repair_e:
                print(f"DEBUG: Repair fallback failed: {str(repair_e)}")
                
                # Save to disk for inspection
                with open("failed_upload.joblib", "wb") as f:
                    f.write(contents)
                print("DEBUG: Saved failed_upload.joblib for inspection")
                raise p_e
        
        # Validation - Part 1: Smart Structure Check
        # If it's already a dict, we just validate it.
        # If it's a raw estimator, we auto-wrap it.
        if not isinstance(model_info, dict):
             loaded_type = type(model_info).__name__
             
             if is_classifier(model_info) or is_regressor(model_info):
                 print(f"DEBUG: Auto-wrapping raw model instance: {loaded_type}")
                 model_info = {
                     'model': model_info,
                     'X_train': None,
                     'X_test': None,
                     'y_train': None,
                     'y_test': None,
                     'mode': 'AUTO_WRAPPED'
                 }
             else:
                 # Defensive Checks for known mismatch types
                 message = f"Invalid file format: Uploaded file must be a dictionary or a raw model, but received {loaded_type}."
                 if "LabelEncoder" in loaded_type:
                     message += " 💡 It looks like you uploaded a LabelEncoder. Deployment analysis requires a prediction model (like RandomForest)."
                 
                 print(f"DEBUG: Validation failed. Expected dict or model, got {loaded_type}")
                 raise HTTPException(status_code=400, detail=message)
        
        # Validation - Part 2: Required Contents Check
        required_keys = ['model', 'X_train', 'X_test', 'y_train', 'y_test']
        for key in required_keys:
            if key not in model_info:
                # If we auto-wrapped, these are None, so we just ensure the keys exist.
                # If they were missing in a user dict, this is an error.
                return {"error": f"Missing key in model_info: {key}"}

        # Validation - Part 3: Nested Security & Type Check
        internal_model = model_info['model']
        if not (is_classifier(internal_model) or is_regressor(internal_model)):
             print(f"DEBUG: Invalid nested model type: {type(internal_model)}")
             raise HTTPException(status_code=400, detail=f"Invalid model content: {type(internal_model).__name__}. Must contain a valid scikit-learn estimator.")
        
        # INVESTIGATION LAYER: Log shapes and types before analysis
        try:
            print("--- PREDICTION INVESTIGATION ---")
            print(f"DEBUG: Model Type: {type(internal_model)}")
            print(f"DEBUG: X_train shape: {getattr(model_info['X_train'], 'shape', 'N/A')}")
            print(f"DEBUG: X_test shape: {getattr(model_info['X_test'], 'shape', 'N/A')}")
            print(f"DEBUG: y_train shape: {getattr(model_info['y_train'], 'shape', 'N/A')}")
            print(f"DEBUG: y_test shape: {getattr(model_info['y_test'], 'shape', 'N/A')}")
            print("--------------------------------")
        except Exception as log_e:
            print(f"DEBUG: Logging failed (ignoring): {log_e}")

        try:
            result = mfps.predict_failure(model_info)
        except ValueError as ve:
            print("❌ PREDICTION FAILED: ValueError")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Model execution failed: {str(ve)}")
        except AttributeError as ae:
            print("❌ PREDICTION FAILED: AttributeError")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Model execution failed: {str(ae)}")
        except Exception as e:
            print("❌ PREDICTION FAILED: General Error")
            traceback.print_exc()
            raise HTTPException(status_code=400, detail=f"Model processing failed: {str(e)}")
        
        # Get explanation
        features = result['features']
        feature_importance = dict(zip(mfps.feature_names, mfps.meta_model.feature_importances_))
        contributions = {k: features[k] * feature_importance[k] for k in mfps.feature_names}
        sorted_contrib = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
        
        top_factors = [
            {"feature": k, "value": features[k], "weight": v} 
            for k, v in sorted_contrib[:10]
        ]
        
        return {
            "prediction": result,
            "explanation": top_factors,
            "filename": file.filename
        }

    except HTTPException:
        raise
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Server Error: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
