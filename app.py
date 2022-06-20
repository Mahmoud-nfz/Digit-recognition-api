import pandas as pd
import numpy as np


import joblib


#load saved model
xgb = joblib.load("XGBoost.pkl")


import json
from flask import Flask, request, jsonify
from flask_cors import CORS


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

@app.route('/',methods=['POST'])
def index():
    data = json.loads(request.data)
    
    sample = np.array(data)
    sample.resize((1,64))
    print(sample.shape)
    
    print(xgb.predict(sample)[0])
    return jsonify({
                'prediction' : int(xgb.predict(sample)[0])
                       })


@app.route('/',methods=['GET'])
def sayhello():
    return "<h1>Welcome to our server !!</h1>"


if __name__ == '__main__':
    # Threaded option to enable multiple instances for multiple user access support
    app.run(threaded=True, port=3000)