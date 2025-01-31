import os
import logging
import warnings
import absl.logging

# Suppress logging and warnings before TensorFlow is imported
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

absl.logging.set_verbosity(absl.logging.ERROR)
logging.getLogger('tensorflow').setLevel(logging.ERROR)

warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=DeprecationWarning)

import sys
sys.stderr = open(os.devnull, 'w')

from flask import Flask, render_template, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

tf.get_logger().setLevel(logging.ERROR)
tf.autograph.set_verbosity(0)
app = Flask(__name__)

# Load the model and tokenizer
model = load_model('lstm_toxicity_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))


@app.route('/')
def home():
    return render_template('home.html')


@app.route('/index')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data (raw text stored in a list)
        data = request.get_json()
        # print(data)
        
        # Extract the input text
        comment = data[0]
        # print(comment)
        if not isinstance(comment, str):
            return jsonify({'error': 'Input must be a string inside the list.'}), 400

        # Preprocess the input
        user_sequences = tokenizer.texts_to_sequences([comment])  # Convert to sequence
        # print("User seq: ", user_sequences)
        max_seq_len = 232
        user_padded = pad_sequences(user_sequences, maxlen=max_seq_len, padding='post', truncating='post')
        # print("Padded seq: ", user_padded)

        # Make prediction
        predictions = model.predict(user_padded)
        # print("Predictions:", predictions)

        threshold = 0.5
        classification = "Negative" if predictions[0][0] > threshold else "Positive"

        # Return the result
        return jsonify({'review': comment, 'classification': classification})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
