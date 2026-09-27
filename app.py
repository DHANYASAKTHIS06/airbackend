from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
from pymongo import MongoClient


app = Flask(__name__)

CORS(app)


BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)


classifier = joblib.load(
    os.path.join(
        BASE_DIR,
        "air_quality_classifier.pkl"
    )
)


regressor = joblib.load(
    os.path.join(
        BASE_DIR,
        "pollution_regressor.pkl"
    )
)


kmeans = joblib.load(
    os.path.join(
        BASE_DIR,
        "pollution_kmeans.pkl"
    )
)


feature_scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "feature_scaler.pkl"
    )
)


pca = joblib.load(
    os.path.join(
        BASE_DIR,
        "pca_model.pkl"
    )
)


cluster_scaler = joblib.load(
    os.path.join(
        BASE_DIR,
        "cluster_scaler.pkl"
    )
)


model_features = joblib.load(
    os.path.join(
        BASE_DIR,
        "model_features.pkl"
    )
)


# -------------------------------------------------
# MongoDB connection
# -------------------------------------------------

MONGO_URI = os.environ.get(
    "MONGO_URI",
    "mongodb+srv://2403717620522007_db_user:PFmfqoMhDMxMsAY2@cluster0.ohomtdx.mongodb.net/?appName=Cluster0"
)

mongo_client = MongoClient(MONGO_URI)

db = mongo_client["air_quality_db"]

predictions_collection = db["predictions"]


def prepare_data(data):

    required_fields = [
        "so2",
        "no2",
        "rspm",
        "spm",
        "pm2_5"
    ]

    for field in required_fields:

        if field not in data:

            raise ValueError(
                f"Missing field: {field}"
            )


    values = pd.DataFrame([{

        "so2": float(data["so2"]),

        "no2": float(data["no2"]),

        "rspm": float(data["rspm"]),

        "spm": float(data["spm"]),

        "pm2_5": float(data["pm2_5"]),

        "year": int(
            data.get("year", 2026)
        ),

        "month": int(
            data.get("month", 1)
        )

    }])


    pollutant_features = [

        "so2",
        "no2",
        "rspm",
        "spm",
        "pm2_5"

    ]


    scaled_data = feature_scaler.transform(
        values[pollutant_features]
    )


    pca_data = pca.transform(
        scaled_data
    )


    values["PC1"] = pca_data[:, 0]

    values["PC2"] = pca_data[:, 1]


    return values


def get_recommendations(quality):

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

            "Prefer public transportation when possible."

        ],


        "Very Poor": [

            "Avoid prolonged outdoor exposure.",

            "Sensitive groups should remain indoors when possible.",

            "Reduce pollution-producing activities."

        ],


        "Severe": [

            "Avoid outdoor activities whenever possible.",

            "Keep windows closed during high pollution periods.",

            "Use appropriate indoor air filtration."

        ]

    }


    return recommendations.get(

        quality,

        ["Monitor air quality regularly."]

    )


@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "success": True,

        "message":
            "Air Pollution ML Backend is running",

        "modules": [

            "Classification",

            "Regression",

            "Pattern Matching",

            "Feature Extraction"

        ]

    })


@app.route("/health", methods=["GET"])
def health():

    mongo_status = "connected"

    try:

        mongo_client.admin.command("ping")

    except Exception:

        mongo_status = "disconnected"

    return jsonify({

        "success": True,

        "status": "healthy",

        "mongodb": mongo_status

    })


@app.route("/model-info", methods=["GET"])
def model_info():

    return jsonify({

        "classification":
            "Random Forest",

        "regression":
            "Random Forest Regressor",

        "pattern_matching":
            "K-Means",

        "feature_extraction":
            "PCA",

        "features": [

            "SO2",

            "NO2",

            "RSPM",

            "SPM",

            "PM2.5"

        ]

    })


@app.route("/predict", methods=["POST"])
def predict():

    try:

        data = request.get_json()


        if not data:

            return jsonify({

                "success": False,

                "error":
                    "No JSON data received"

            }), 400


        values = prepare_data(data)


        X = values[model_features]


        # Classification

        predicted_quality = classifier.predict(X)[0]


        probabilities = classifier.predict_proba(X)[0]


        confidence = float(
            np.max(probabilities) * 100
        )


        # Regression

        pollution_score = regressor.predict(X)[0]


        # Pattern Matching

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

            kmeans.predict(
                cluster_scaled
            )[0]

        )


        # PCA Feature Extraction

        pc1 = float(
            values["PC1"].iloc[0]
        )

        pc2 = float(
            values["PC2"].iloc[0]
        )


        # Response

        result = {

            "success": True,

            "air_quality":
                predicted_quality,

            "confidence":
                round(
                    confidence,
                    2
                ),

            "pollution_score":
                round(
                    float(pollution_score),
                    4
                ),

            "pollution_pattern":
                pattern,

            "feature_extraction": {

                "PC1":
                    round(
                        pc1,
                        4
                    ),

                "PC2":
                    round(
                        pc2,
                        4
                    )

            },

            "recommendations":
                get_recommendations(
                    predicted_quality
                )

        }


        # Save prediction to MongoDB

        try:

            log_entry = dict(result)

            log_entry["input"] = data

            log_entry["timestamp"] = datetime.utcnow()

            predictions_collection.insert_one(log_entry)

        except Exception as db_error:

            print(f"MongoDB insert failed: {db_error}")


        return jsonify(result)


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


@app.route("/classification", methods=["POST"])
def classification():

    try:

        data = request.get_json()

        values = prepare_data(data)

        X = values[model_features]

        prediction = classifier.predict(X)[0]

        probabilities = classifier.predict_proba(X)[0]

        confidence = float(
            np.max(probabilities) * 100
        )

        return jsonify({

            "success": True,

            "air_quality":
                prediction,

            "confidence":
                round(
                    confidence,
                    2
                )

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


@app.route("/regression", methods=["POST"])
def regression():

    try:

        data = request.get_json()

        values = prepare_data(data)

        X = values[model_features]

        prediction = regressor.predict(X)[0]

        return jsonify({

            "success": True,

            "pollution_score":
                round(
                    float(prediction),
                    4
                )

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


@app.route("/pattern", methods=["POST"])
def pattern_prediction():

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


        pattern = int(

            kmeans.predict(
                cluster_scaled
            )[0]

        )


        return jsonify({

            "success": True,

            "pollution_pattern":
                pattern

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


@app.route("/features", methods=["POST"])
def feature_extraction():

    try:

        data = request.get_json()

        values = prepare_data(data)


        return jsonify({

            "success": True,

            "PC1":
                round(
                    float(
                        values[
                            "PC1"
                        ].iloc[0]
                    ),
                    4
                ),

            "PC2":
                round(
                    float(
                        values[
                            "PC2"
                        ].iloc[0]
                    ),
                    4
                )

        })


    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


@app.route("/history", methods=["GET"])
def get_history():

    try:

        limit = int(request.args.get("limit", 20))

        records = list(

            predictions_collection.find(
                {},
                {"_id": 0}
            ).sort(
                "timestamp", -1
            ).limit(limit)

        )

        for record in records:

            if "timestamp" in record and isinstance(record["timestamp"], datetime):

                record["timestamp"] = record["timestamp"].isoformat()

        return jsonify({

            "success": True,

            "count": len(records),

            "history": records

        })

    except Exception as e:

        return jsonify({

            "success": False,

            "error": str(e)

        }), 400


if __name__ == "__main__":

    port = int(

        os.environ.get(
            "PORT",
            5000
        )

    )


    app.run(

        host="0.0.0.0",

        port=port

    )
