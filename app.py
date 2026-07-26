import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# Page Config
st.set_page_config(
    page_title="LSTM Sentence Generator",
    page_icon="✨",
    layout="centered"
)


# loading files

@st.cache_resource
def load_files():
    model = load_model("lstm_model.h5")

    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("max_len.pkl", "rb") as f:
        max_len = pickle.load(f)

    return model, tokenizer, max_len


model, tokenizer, max_len = load_files()


# Create index -> word dictionary
index_word = {v: k for k, v in tokenizer.word_index.items()}


# Predict Next Word
def predict_word(text):

    seq = tokenizer.texts_to_sequences([text])[0]

    seq = pad_sequences(
        [seq],
        maxlen=max_len - 1,
        padding="pre"
    )

    pred = model.predict(seq, verbose=0)

    pred_index = np.argmax(pred)

    return index_word.get(pred_index, "")


# Generate Sentence
def generate_sentence(seed_text, num_words):

    generated = seed_text

    for _ in range(num_words):

        next_word = predict_word(generated)

        if next_word == "":
            break

        generated += " " + next_word

    return generated


#  UI
st.title("✨ LSTM Sentence Generator")


st.markdown("---")

seed_text = st.text_input(
    "Enter Starting Text",
    placeholder="Example: what are you"
)

num_words = st.slider(
    "Words to Generate",
    min_value=1,
    max_value=30,
    value=10
)

col1, col2 = st.columns([1, 1])

with col1:
    generate = st.button("🚀 Generate")

with col2:
    clear = st.button("🗑 Clear")

if clear:
    st.rerun()

if generate:

    if seed_text.strip() == "":
        st.warning("Please enter some text.")

    else:

        sentence = generate_sentence(seed_text, num_words)

        st.success("Sentence Generated")

        st.markdown("### Generated Sentence")

        st.info(sentence)


st.markdown(
    """
    
    <div style="text-align:center; color:gray; font-size:14px; padding:150px;">
        © 2026 Next Word/Quotes Prediction System | Developed by <b>Bimlesh Kumar Sah</b><br>
        All Rights Reserved.
    </div>
    """,
    unsafe_allow_html=True
)
