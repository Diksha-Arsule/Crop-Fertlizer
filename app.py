from flask import Flask, request, jsonify
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
with open("crop_model.pkl", "rb") as f:
    model = pickle.load(f)

# Fertilizer recommendation logic
def recommend_fertilizer(n, p, k):
    if n < 50:
        return "Urea (N-rich)"
    elif p < 30:
        return "Super Phosphate (P-rich)"
    elif k < 30:
        return "Muriate of Potash (K-rich)"
    else:
        return "Balanced Fertilizer"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = np.array([[data["N"], data["P"], data["K"], data["temperature"], 
                          data["humidity"], data["ph"], data["rainfall"]]])
    
    # Predict crop
    crop = model.predict(features)[0]
    
    # Recommend fertilizer
    fertilizer = recommend_fertilizer(data["N"], data["P"], data["K"])

    return jsonify({"Recommended Crop": crop, "Recommended Fertilizer": fertilizer})

if __name__ == "__main__":
    app.run(debug=True)
