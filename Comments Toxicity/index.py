from flask import Flask, render_template, request, jsonify
import pickle
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# Load the model and tokenizer
model = load_model('lstm_toxicity_model.h5')
tokenizer = pickle.load(open('tokenizer.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get input data (raw text stored in a list)
        data = request.get_json()
        print(data)
        
        # Extract the input text
        comment = data[0]
        print(comment)
        if not isinstance(comment, str):
            return jsonify({'error': 'Input must be a string inside the list.'}), 400

        # Preprocess the input
        user_sequences = tokenizer.texts_to_sequences([comment])  # Convert to sequence
        print("User seq: ", user_sequences)
        user_padded = pad_sequences(user_sequences, maxlen=100, padding='post', truncating='post')
        print("Padded seq: ", user_padded)

        # Make prediction
        predictions = model.predict(user_padded)
        print("Predictions:", predictions)

        threshold = 0.5
        classification = "toxic" if predictions[0][0] > threshold else "non_toxic"

        # Return the result
        return jsonify({'comment': comment, 'classification': classification})

    except Exception as e:
        print("Error:", e)
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
