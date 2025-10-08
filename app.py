import streamlit as st
import joblib

# Load the saved model pipeline
@st.cache_resource
def load_model():
    model = joblib.load('spam_model.joblib')
    return model

model = load_model()

# --- Streamlit App Interface ---

st.title("Spam Message Detector 📩")
st.write("Enter a message below to check if it's spam or not.")

# Text input box for user
user_input = st.text_area("Your Message", height=150)

# Prediction button
if st.button("Analyze Message"):
    if user_input:
        # The model expects a list of texts for prediction
        prediction = model.predict([user_input])

        # Display the result
        st.subheader("Analysis Result")
        if prediction[0] == 1:
            st.error("This looks like a SPAM message.")
            # NEW, WORKING SPAM IMAGE URL
            st.image("https://i.imgur.com/t2yG8V2.png", width=150)
        else:
            st.success("This looks like a legitimate message (HAM).")
            # NEW, WORKING HAM IMAGE URL
            st.image("https://i.imgur.com/1y3tFmI.png", width=150)
    else:
        st.warning("Please enter a message to analyze.")