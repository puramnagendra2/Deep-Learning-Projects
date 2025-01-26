from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the model and scaler
model = joblib.load('price_prediction_pipeline.h5')

@app.route('/')
def open():
    return render_template('open.html')


@app.route('/index')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data from the form
        data = request.json
        print("Received data:", data)
        
        # Prepare the features for prediction (as a DataFrame)
        features_dict = {
            'year': [data['year']],
            'cylinders': [data['cylinders']],
            'odometer': [data['odometer']],
            'condition': [data['condition']],
            'fuel': [data['fuel']],
            'title_status': [data['title_status']],
            'transmission': [data['transmission']],
            'drive': [data['drive']],
            'size': [data['size']],
            'type': [data['type']]
        }
        
        # Convert features into a pandas DataFrame
        features_df = pd.DataFrame(features_dict)
        print("Features DataFrame:", features_df)

        # Make prediction
        predicted_price = model.predict(features_df)[0]
        
        return jsonify({'predicted_price': float(predicted_price*84)})
    
    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)
