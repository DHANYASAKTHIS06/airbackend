from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

classifier = joblib.load(
    os.path.join(MODEL_DIR, "air_quality_classifier.pkl")
)

regressor = joblib.load(
    os.path.join(MODEL_DIR, "pollution_regressor.pkl")
)

kmeans = joblib.load(
    os.path.join(MODEL_DIR, "pollution_kmeans.pkl")
)

scaler = joblib.load(
    os.path.join(MODEL_DIR, "feature_scaler.pkl")
)

pca = joblib.load(
    os.path.join(MODEL_DIR, "pca_model.pkl")
)

cluster_scaler = joblib.load(
    os.path.join(MODEL_DIR, "cluster_scaler.pkl")
)

model_features = joblib.load(
    os.path.join(MODEL_DIR, "model_features.pkl")
)


def prepare_data(data):

    required = [
        "so2",
        "no2",
        "rspm",
        "spm",
        "pm2_5"
    ]

    for field in required:
        if field not in data:
            raise ValueError(f"Missing field: {field}")

    values = pd.DataFrame([{
        "so2": float(data["so2"]),
        "no2": float(data["no2"]),
        "rspm": float(data["rspm"]),
        "spm": float(data["spm"]),
        "pm2_5": float(data["pm2_5"]),
        "year": int(data.get("year", 2026)),
        "month": int(data.get("month", 1))
    }])

    pollutant_features = [
        "so2",
        "no2",
        "rspm",
        "spm",
        "pm2_5"
    ]

    scaled = scaler.transform(
        values[pollutant_features]
    )

    pca_values = pca.transform(scaled)

    values["PC1"] = pca_values[:, 0]
    values["PC2"] = pca_values[:, 1]

    return values


def get_recommendation(quality):

    recommendations = {

        "Good": [
            "Air quality is satisfactory.",
            "Normal outdoor activities are suitable.",
            "Continue monitoring air quality."
        ],

        "Moderate": [
            "Sensitive people should reduce prolonged outdoor exposure.",
            "Avoid unnecessary vehicle usage.",
            "Monitor air quality regularly."
        ],

        "Poor": [
            "Reduce prolonged outdoor activities.",
            "Sensitive people should avoid heavy outdoor exercise.",
            "Use public transportation when possible."
        ],

        "Very Poor": [
            "Avoid prolonged outdoor exposure.",
            "Sensitive groups should remain indoors when possible.",
            "Reduce pollution-producing activities."
        ],

        "Severe": [
            "Avoid outdoor activities whenever possible.",
            "Keep windows closed during high pollution periods.",
            "Use appropriate air filtration indoors."
        ]
    }

    return recommendations.get(
        quality,
        ["Monitor air quality regularly."]
    )


@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "status": "success",
        "message": "Air Pollution Prediction API is running"
    })


@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy"
    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()

        values = prepare_data(data)

        X = values[model_features]

        quality = classifier.predict(X)[0]

        probabilities = classifier.predict_proba(X)[0]

        confidence = float(
            np.max(probabilities) * 100
        )

        pollution_score = regressor.predict(X)[0]

        cluster_input = values[
            [
                "so2",
                "no2",
                "rspm",
                "spm",
                "pm2_5"
            ]
        ]

        cluster_scaled = cluster_scaler.transform(
            cluster_input
        )

        pattern = int(
            kmeans.predict(cluster_scaled)[0]
        )

        pca_values = {
            "PC1": round(float(values["PC1"].iloc[0]), 4),
            "PC2": round(float(values["PC2"].iloc[0]), 4)
        }

        return jsonify({

            "success": True,

            "air_quality": quality,

            "confidence": round(
                confidence,
                2
            ),

            "pollution_score": round(
                float(pollution_score),
                4
            ),

            "pollution_pattern": pattern,

            "feature_extraction": pca_values,

            "recommendations":
                get_recommendation(quality)

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/pattern", methods=["POST"])
def pattern():

    try:

        data = request.get_json()

        values = prepare_data(data)

        cluster_input = values[
            [
                "so2",
                "no2",
                "rspm",
                "spm",
                "pm2_5"
            ]
        ]

        cluster_scaled = cluster_scaler.transform(
            cluster_input
        )

        pattern_number = int(
            kmeans.predict(cluster_scaled)[0]
        )

        return jsonify({

            "success": True,

            "pollution_pattern":
                pattern_number,

            "message":
                f"Input belongs to pollution pattern {pattern_number}"

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/features", methods=["POST"])
def features():

    try:

        data = request.get_json()

        values = prepare_data(data)

        return jsonify({

            "success": True,

            "PC1": round(
                float(values["PC1"].iloc[0]),
                4
            ),

            "PC2": round(
                float(values["PC2"].iloc[0]),
                4
            )

        })

    except Exception as e:

        return jsonify({
            "success": False,
            "error": str(e)
        }), 400


@app.route("/model-info", methods=["GET"])
def model_info():

    return jsonify({

        "classification": "Random Forest",

        "regression": "Random Forest Regressor",

        "pattern_matching": "K-Means",

        "feature_extraction": "PCA",

        "pollutants": [
            "SO2",
            "NO2",
            "RSPM",
            "SPM",
            "PM2.5"
        ]

    })


if __name__ == "__main__":

    port = int(
        os.environ.get("PORT", 5000)
    )

    app.run(
        host="0.0.0.0",
        port=port
    )
