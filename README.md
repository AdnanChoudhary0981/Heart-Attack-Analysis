# Heart Attack Analysis - Upgraded Project

This upgraded package includes:
- Trained model pipeline (Gradient Boosting) saved as `heart_model_pipeline.joblib`
- Flask app (`app.py`) to serve predictions and basic SHAP-based explanations
- Dockerfile and requirements.txt for containerization
- Sample predictions CSV
- Jupyter notebook (expanded) with EDA, modeling, SHAP, and deployment snippets

## How to run locally
1. Create virtual environment and install requirements:
   ```bash
   python -m venv venv
   source venv/bin/activate  # on Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. Run the Flask app:
   ```bash
   python app.py
   ```
3. Send POST requests to `/predict` with JSON payload like:
   ```json
   { "age": 54, "sex": 1, "cp": 3, "trestbps": 160, "chol": 240, "fbs": 0, "restecg": 1, "thalach": 140, "exang": 1, "oldpeak": 1.2, "slope": 2, "ca": 1, "thal": 2 }
   ```

## Docker
Build and run:
```bash
docker build -t heart-app .
docker run -p 5000:5000 heart-app
```
