import os
import json
import numpy as np
import joblib
from flask import Flask, request, jsonify, render_template, send_from_directory

app = Flask(__name__)

EDA_DIR = os.path.join(os.path.dirname(__file__), 'eda_output')

@app.route('/eda-images/<path:filename>')
def eda_images(filename):
    return send_from_directory(EDA_DIR, filename)

# ─── Load model ──────────────────────────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'house_model.pkl')
model_data = None

def load_model():
    global model_data
    if os.path.exists(MODEL_PATH):
        model_data = joblib.load(MODEL_PATH)
        print(f"✅ Loaded model: {model_data['best_name']}")
    else:
        print("⚠️  Model not found. Run model/train.py first.")

load_model()

# ─── Routes ──────────────────────────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/eda')
def eda():
    if model_data is None:
        return "Model not loaded. Please run model/train.py first.", 503
    return render_template(
        'eda.html',
        best_name=model_data['best_name'],
        metrics=model_data['metrics'],
        all_results=model_data['all_results'],
    )

@app.route('/model-info')
def model_info():
    if model_data is None:
        return jsonify({'error': 'Model not loaded'}), 503
    return jsonify({
        'best_model':  model_data['best_name'],
        'metrics':     model_data['metrics'],
        'all_results': model_data['all_results'],
    })

@app.route('/predict', methods=['POST'])
def predict():
    if model_data is None:
        return jsonify({'error': 'Model not loaded. Run model/train.py first.'}), 503

    try:
        data = request.get_json(force=True)

        # Raw inputs
        MedInc      = float(data.get('MedInc', 3.0))
        HouseAge    = float(data.get('HouseAge', 20.0))
        AveRooms    = float(data.get('AveRooms', 5.0))
        AveBedrms   = float(data.get('AveBedrms', 1.0))
        Population  = float(data.get('Population', 1000.0))
        AveOccup    = float(data.get('AveOccup', 3.0))
        Latitude    = float(data.get('Latitude', 35.0))
        Longitude   = float(data.get('Longitude', -119.0))

        # Engineered features
        RoomsPerHousehold      = AveRooms / (HouseAge + 1)
        BedroomsPerRoom        = AveBedrms / (AveRooms + 1e-5)
        PopulationPerHousehold = Population / (AveOccup + 1e-5)

        feature_cols = model_data['feature_cols']
        raw = {
            'MedInc':                 MedInc,
            'HouseAge':               HouseAge,
            'AveRooms':               AveRooms,
            'AveBedrms':              AveBedrms,
            'Population':             Population,
            'AveOccup':               AveOccup,
            'Latitude':               Latitude,
            'Longitude':              Longitude,
            'RoomsPerHousehold':      RoomsPerHousehold,
            'BedroomsPerRoom':        BedroomsPerRoom,
            'PopulationPerHousehold': PopulationPerHousehold,
        }
        X = np.array([[raw[c] for c in feature_cols]])
        X_scaled = model_data['scaler'].transform(X)
        pred = model_data['model'].predict(X_scaled)[0]

        # California Housing prices are in $100k units
        price_usd = round(float(pred) * 100_000, 2)

        return jsonify({
            'predicted_price':    price_usd,
            'predicted_price_fmt': f"${price_usd:,.0f}",
            'model_used':         model_data['best_name'],
            'r2_score':           model_data['metrics']['r2'],
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 400


if __name__ == '__main__':
    app.run(debug=True, port=5000)
