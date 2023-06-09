import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

# Create flask app
flask_app = Flask(__name__, template_folder='template')
model = pickle.load(open("model.pkl", "rb"))

@flask_app.route("/")
def Home():
    return render_template("trial.html")

@flask_app.route("/predict", methods = ["POST"])
def predict():
    given_features = [int(x) for x in request.form.values()]
    features = [np.array(given_features)]
    prediction = model.predict(features)
    if prediction[0] == 0:
        text="Edible"
    else:
        text="Poisonous"
    return render_template("trial.html", prediction_text = " mushroom is  {}".format(text))

if __name__ == "__main__":
    flask_app.run(debug=True)