import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import pickle
import numpy as np

# Load the models and tokenizers
lstm_model = 'lstm_toxicity_model.h5'
tokenizer_model = 'tokenizer.pkl'

# Load LSTM model
lstm_model = load_model(lstm_model)

# Tokenizer
tokenizer = pickle.load(open("tokenizer.pkl", "rb"))

# Padded Sequence
pad_text_sequences = pickle.load(open("pad_text_sequences.pkl", "rb"))

###############################################

# Predict on new comments

def prediction_model(user_input):
    user_sequences = tokenizer.texts_to_sequences(user_input)
    label_names = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

    user_padded = pad_text_sequences(user_sequences, max_sequence_length=100, padding='post', truncating='post')

    predictions = lstm_model.predict(user_padded)
    binary_labels = (np.array(predictions) >= 0.5).astype(int)

    # Convert binary labels to a dictionary with their respective names
    predicted_labels = [label_names[i] for i in range(len(label_names)) if binary_labels[0][i] == 1]
    print(predictions)
    if not predicted_labels:
        return "Not Toxic"
    else:
        return predicted_labels


###############################################
# Streamlit interface
st.title("Toxic Comment Classifier")

st.markdown("Enter the text you want to classify:")
user_input = st.text_area(label="Enter Comment")
if st.button("Classify"):
    if user_input:
        
        result = prediction_model(user_input)
        print(result)

        if result is not None:
            # Display the results with larger font size
            st.markdown("### Results")
            st.markdown(f"**Prediction:** {result}", unsafe_allow_html=True)

            # Apply custom CSS for larger font size and responsive layout
            st.markdown(
                """
                <style>
                .big-font {
                    font-size: 24px !important;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Add a class to the output to make the font size bigger
            st.markdown(f'<p class="big-font">{result}</p>', unsafe_allow_html=True)
    else:
        st.warning("Please enter some text to classify.")
