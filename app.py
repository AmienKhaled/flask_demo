from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import joblib
import os

# __name__ is equal to app.py
app = Flask(__name__)

UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# load model from model.pkl
model = joblib.load('model.pkl')



@app.route("/", methods=['GET'])
def home():
    return render_template('index.html')



@app.route("/predict", methods=["POST"])
def predict():
	print(request.files)
	if 'file' not in request.files:
		return "Error"
	# Save File
	file = request.files['file']
	file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
	file.save(file_path)
	print(file_path)
	img = Image.open(file_path)
	my_img_resized = np.array(img.resize((28,28)).convert('L'))
	print(my_img_resized)
	prediction = model.predict(my_img_resized.reshape(-1, 28*28)/255)

	return render_template("index.html", digit=prediction[0])	




if __name__ == "__main__":
    app.run(debug=True)
