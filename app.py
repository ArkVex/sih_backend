
import joblib
from flask import Flask, request, jsonify

app = Flask(__name__)
model = joblib.load("model.pkl")



@app.route('/')
def home():
    return "hello"

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    # Expecting: {"features": [val1, val2, ...]}
    features = [data['features']]
    prediction = model.predict_proba(features)
    # prediction[0] is a numpy array of probabilities for each class
    pred_list = prediction[0].tolist()  # Convert to list of floats
    return jsonify({'prediction': pred_list})

if __name__ == "__main__":
    app.run(debug=True, port=5500)
    