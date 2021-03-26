from flask import Flask, render_template, request, jsonify, redirect
from PIL import Image
import numpy as np
import joblib
import os

# __name__ is equal to app.py
app = Flask(__name__)


# load model from model.pkl
model = joblib.load('model.pkl')
tfidv = joblib.load('tfidv.pkl')



@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')



@app.route("/predict", methods=["POST"])
def predict():

	if not request.form.get("tweet", False):
		return redirect("/")

	text = request.form.get("tweet")
	prediction = model.predict(tfidv.transform( [text ] ) )

	return render_template("index.html", twt= "Positive" if prediction[0] else "Negative")	




@app.route("/api/predict", methods=["POST"])
def api_predict():

	if not request.form.get("tweet", False):
		return jsonify({"prediction": "Error"})

	text = request.form.get("tweet")
	prediction = model.predict(tfidv.transform( [text ] ) )

	return jsonify({"prediction": "Positive" if prediction[0] else "Negative"})


if __name__ == "__main__":
    app.run(debug=True)
