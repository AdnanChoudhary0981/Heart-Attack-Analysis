from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np
import shap
app = Flask(__name__)
model = joblib.load('heart_model_pipeline.joblib')
# load explainer lazily
explainer = None

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    probs = model.predict_proba(df)[:,1]
    preds = (probs >= 0.5).astype(int)
    return jsonify({'predictions': preds.tolist(), 'probabilities': probs.tolist()})

@app.route('/explain', methods=['POST'])
def explain():
    global explainer
    data = request.get_json()
    if isinstance(data, dict):
        df = pd.DataFrame([data])
    else:
        df = pd.DataFrame(data)
    # create explainer on first call (TreeExplainer for tree models)
    if explainer is None:
        try:
            explainer = shap.TreeExplainer(model.named_steps['classifier'])
        except Exception as e:
            return jsonify({'error': 'Failed to create SHAP explainer: ' + str(e)}), 500
    # transform the input
    transformed = model.named_steps['preprocessor'].transform(df)
    shap_values = explainer.shap_values(transformed)
    # return top features for positive class
    # Note: we return only indexes and approximate importance magnitude
    if isinstance(shap_values, list):
        sv = shap_values[1]
    else:
        sv = shap_values
    topn = 5
    feature_importances = np.abs(sv).mean(axis=0)
    top_idx = np.argsort(feature_importances)[-topn:][::-1].tolist()
    return jsonify({'top_feature_indices': top_idx, 'importances': feature_importances[top_idx].tolist()})
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
