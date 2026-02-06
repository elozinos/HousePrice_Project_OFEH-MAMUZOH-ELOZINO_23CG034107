from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

# Load the trained Random Forest model (relative path for Render)
MODEL_PATH = os.path.join("model", "model_random_forest.pkl")
model = joblib.load(MODEL_PATH)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        features = [
            float(request.form['OverallQual']),
            float(request.form['GrLivArea']),
            float(request.form['GarageCars']),
            float(request.form['TotalBsmtSF']),
            float(request.form['FullBath']),
            float(request.form['YearBuilt'])
        ]

        features_array = np.array([features])
        prediction = model.predict(features_array)[0]

        return render_template(
            "index.html",
            prediction_text=f"Predicted House Price: ${prediction:,.2f}"
        )

    except Exception as e:
        return render_template(
            "index.html",
            prediction_text=f"Error: {str(e)}"
        )

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
