import streamlit as st
import pandas as pd
import joblib
import requests
from streamlit_lottie import st_lottie

st.set_page_config(
    page_title="Chatbot", 
    page_icon="🤖",  
    layout="centered"  # 'wide' or 'centered'
)

def load_lottie(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

chatbot_loader1 = load_lottie(r"https://lottie.host/7bf2bffa-ea74-4655-b682-37ad8a673a34/zDJErf7ae2.json")
chatbot_loader2 = load_lottie(r"https://lottie.host/768a4667-75e4-4790-900d-c603ad558bc8/fcBHGhd42V.json")

col1, col2 = st.columns(2)
with col1:
    st_lottie(chatbot_loader1, height=150, reverse=True)
with col2:
    st_lottie(chatbot_loader2, height=150, reverse=True)
st.title("Rule Based Chatbot ")

# Load the saved pipeline
@st.cache_resource
def load_pipeline(name):
    return joblib.load(name)

@st.cache_resource
def load_data(file_path):
    return pd.read_csv(file_path)

pipe5 = "chatbot_pipeline5.pkl"
pipeline5 = load_pipeline(pipe5)

# Load the CSV dataset
csv_file_path = "dataset.csv"  # Replace with your dataset file path
df = load_data(csv_file_path)

# Select a column from the dataset
column_name = "question"  # Replace with the column name in your dataset
if column_name not in df.columns:
    st.error(f"Column '{column_name}' not found in the dataset.")
else:
    odd_rows_df = df.iloc[::2]
    options = odd_rows_df[column_name].dropna().unique()

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Create a container for the chat history at the top
    chat_placeholder = st.empty()

    # Display chat messages from history
    with chat_placeholder.container():
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    # Place the user input dropdown at the bottom
    with st.container():
        st.divider()  # Visual separator
        selected_input = st.selectbox("Choose a question to ask the chatbot:", options, key="dropdown_input")
        if st.button("Send"):
            # Add user message to chat history
            st.session_state.messages.append({"role": "user", "content": selected_input})

            try:
                # Predict using the pipeline
                prediction = pipeline5.predict([selected_input])[0]
                response = f"Prediction: {prediction}"
            except Exception as e:
                response = f"An error occurred during prediction: {e}"

            # Add assistant response to chat history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Update the chat history in the placeholder
            with chat_placeholder.container():
                for message in st.session_state.messages:
                    with st.chat_message(message["role"]):
                        st.markdown(message["content"])