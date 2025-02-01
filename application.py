import pickle
import os
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

application = Flask(__name__)
app = application

# Define paths correctly
ridge_path = os.path.join(os.getcwd(), 'models', 'ridge.pkl')
scaler_path = os.path.join(os.getcwd(), 'models', 'scaler.pkl')

# Check if files exist before loading
if not os.path.exists(ridge_path) or not os.path.exists(scaler_path):
    raise FileNotFoundError("Model or scaler file not found!")

# Load Ridge regression model and StandardScaler
ridge_model = pickle.load(open(ridge_path, 'rb'))
standard_scaler = pickle.load(open(scaler_path, 'rb'))

@app.route("/")
def index():
    return render_template('index.html')

@app.route("/predictdata",methods=['GET','POST'])
def predict_datapoint():
    if request.method == "POST":
        Temperature = float(request.form.get('Temperature'))
        RH = float(request.form.get('RH'))
        WS = float(request.form.get('ws'))
        Rain = float(request.form.get('Rain'))
        FFMC = float(request.form.get('FFMC'))
        DMC = float(request.form.get('DMC'))
        ISI = float(request.form.get('ISI'))
        Classes = float(request.form.get('Classes'))
        Region = float(request.form.get('Region'))

        new_data_scaled=standard_scaler.transform([[Temperature,RH,WS,Rain,FFMC,DMC,ISI,Classes,Region]])
        result=ridge_model.predict(new_data_scaled)
        return render_template('home.html',results=result[0])

    else:
        return render_template('home.html')

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
