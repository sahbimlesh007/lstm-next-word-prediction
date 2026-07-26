# LSTM Next Word Prediction

This project is a **Next Word Prediction** application built using **Python, TensorFlow/Keras, LSTM, and Streamlit**. It predicts the next words from a given input text and generates sentence continuations using a trained LSTM model.

## Features

- Predicts the next word using an LSTM model
- Generates complete sentence continuations
- Simple and interactive Streamlit interface
- Fast model loading with caching

## Technologies Used

- Python
- TensorFlow / Keras
- Streamlit
- NumPy

## Project Structure

```
├── app.py
├── lstm_model.h5
├── tokenizer.pkl
├── max_len.pkl
├── requirements.txt
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Run the Application

```bash
streamlit run app.py
```

## Example

**Input:**
```
Imperfection is
```

**Output:**
```
Imperfection is beauty, madness is genius and it's better to be absolutely ridiculous than absolutely boring.
```

> *The generated sentence depends on the model and training dataset.*

## Author

**BIMLESH KUMAR SAH**
