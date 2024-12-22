import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
import pickle

# Load model
model = load_model("chatbot_model.keras")

# Load intents
with open("intents.json") as file:
    data = json.load(file)

# Load the tokenizer
with open("tokenizer.json") as file:
    tokenizer_data = json.load(file)  # The JSON object is now a string
tokenizer = tokenizer_from_json(tokenizer_data)

# Load label encoder
with open("label_encoder.pkl", "rb") as file:
    label_encoder = pickle.load(file)

# Responses dictionary
responses = {intent["tag"]: intent["responses"] for intent in data["intents"]}
max_len = 20

# Define get_response function
def get_response(user_input):
    # Preprocess user input
    sequence = tokenizer.texts_to_sequences([user_input])
    if not sequence or len(sequence[0]) == 0:
        return "I'm sorry, I didn't understand that. Can you rephrase?"

    padded_sequence = pad_sequences(sequence, maxlen=max_len, truncating="post")

    # Predict intent
    prediction = model.predict(padded_sequence)
    tag = label_encoder.inverse_transform([np.argmax(prediction)])

    # Return a random response
    return np.random.choice(responses[tag[0]])

# Streamlit UI
st.title("Chatbot with Streamlit")
st.write("Talk to the chatbot!")

user_input = st.text_input("You: ")
if user_input:
    response = get_response(user_input)
    st.write(f"Chatbot: {response}")
