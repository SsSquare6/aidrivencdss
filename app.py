import os
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS  # Allows your HTML file to talk to this API
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- 1. GLOBAL MODEL SETUP ---
# We train the model once when the server starts
rf_model = None
label_encoders = {}
feature_columns = []

def train_model():
    global rf_model, label_encoders, feature_columns
    file_path = 'Athlete_recovery_dataset.csv'
    
    if not os.path.exists(file_path):
        print(f"CRITICAL ERROR: {file_path} not found.")
        return False

    df = pd.read_csv(file_path)

    # Cleaning (matching your original logic)
    cols_to_drop = ['Athlete_ID', 'Recovery_Time', 'Heart_Rate', 'Blood_Pressure', 
                    'Muscle_Recovery_Status', 'POMS_Score', 'Training_Intensity', 
                    'Sleep_Hours', 'Dietary_Intake', 'Training_Days_per_Week', 
                    'Recovery_Days_per_Week', 'Imaging_Report_Severity']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Encoding
    for col in df.select_dtypes(include=['object', 'string']).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

    X = df.drop(columns=['Recovery_Success'])
    y = df['Recovery_Success']
    feature_columns = X.columns.tolist()

    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    rf_model.fit(X, y)
    print("AI Brain Trained Successfully!")
    return True

# Initialize the model
train_model()

# --- 2. API ENDPOINTS ---

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json
        
        # Extract inputs from the frontend request
        injury = data.get('injuryType', '').lower()
        severity = data.get('severityLabel', 'Mild')
        confidence = int(data.get('confidence', 5))

        # Encode inputs for the model
        try:
            injury_encoded = label_encoders['Injury_Type'].transform([injury])[0]
        except (KeyError, ValueError):
            injury_encoded = 0 
            
        try:
            severity_encoded = label_encoders['Injury_Severity'].transform([severity])[0]
        except (KeyError, ValueError):
            severity_encoded = 0

        # Build feature array: [Injury_Type, Injury_Severity, (Placeholder), Confidence]
        # Matching your model's 4-feature expectation
        user_features = np.array([[injury_encoded, severity_encoded, 1, confidence]])
        
        # Get probability
        prob_success = rf_model.predict_proba(user_features)[0][1]
        
        # Identify top feature importance for the "Insight" section
        importances = rf_model.feature_importances_
        top_feature = feature_columns[np.argmax(importances)].replace('_', ' ')

        return jsonify({
            "success": True,
            "probability": float(prob_success),
            "top_feature": top_feature
        })

    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 400

@app.route('/', methods=['GET'])
def health_check():
    return "MuscleMend API is running!"

if __name__ == "__main__":
    # Use the port assigned by the hosting provider, or default to 5000
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)