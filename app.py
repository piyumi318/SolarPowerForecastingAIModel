from flask import Flask, request, abort, jsonify
import pandas as pd
from catboost import CatBoostRegressor
import pickle
import requests
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

model = pickle.load(open("SolarPowerForecastingFinalModelFinal.pkl", 'rb'))


@app.route('/predict', methods=['POST'])
def predict():
    try:
        prediction = 0
        windSpeed = request.json['WindSpeed']
        radiation = request.json['Radiation']
        airTemperature = request.json['AirTemperature']
        relativeAirHumidity = request.json['RelativeAirHumidity']
        hour = request.json['Hour']
        sunshine = request.json['Sunshine']
        airPressure = request.json['AirPressure']
        month = request.json['Month']
        day = request.json['Day']
        input = pd.DataFrame([[windSpeed, sunshine, airPressure, radiation, airTemperature,
                               relativeAirHumidity, hour, month, day]],
                             columns=['WindSpeed', 'Sunshine', 'AirPressure', 'Radiation',
                                      'AirTemperature', 'RelativeAirHumidity', 'hour', 'month', 'day'])
        print(input)
        prediction = model.predict(input)[0]
        prediction = round(prediction, 2)
        print(prediction)
        return jsonify({'prediction': prediction})

    except Exception as e:

        print(f"An error occurred: {str(e)}")
        return jsonify({'error': 'An error occurred during prediction'})


if __name__ == '__main__':
    app.run()
