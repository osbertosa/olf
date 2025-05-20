# app.py
from flask import Flask, request, jsonify, app, url_for, render_template

import pickle
import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences


# Initialize the app
app = Flask(__name__)

# Load the trained model
model = pickle.load(open('research/model.pkl', 'rb'))
label_encoder= pickle.load(open('research/label_encoder.pkl', 'rb'))
tokenizer = pickle.load(open('research/tokenizer.pkl', 'rb'))

MAXLEN = 100
@app.route('/')
def home():
    return 'LSTM Model for Case Status Prediction'


@app.route('/predict', methods=['POST'])
def predict():
    try:
        df = request.json['text']  # Expecting {"text": "job title or description"}
        sequence = tokenizer.texts_to_sequences([df])
        padded = pad_sequences(sequence, maxlen=MAXLEN)
        prediction = model.predict(padded)[0][0]
        label = 'Certified' if prediction >= 0.5 else 'Denied'
        return render_template('index.html', prediction_text= 'The predicted case status is: {}'.format(label))
    except Exception as e:
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)