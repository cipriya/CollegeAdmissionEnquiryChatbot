# from flask import Flask, render_template, request
# import pickle
# import json
# import random

# app = Flask(__name__)

# # Load the trained model and vectorizer
# with open('model/chatbot_model.pkl', 'rb') as f:
#     best_model = pickle.load(f)

# with open('model/vectorizer.pkl', 'rb') as f:
#     vectorizer = pickle.load(f)

# # Load the intents data
# with open('dataset/intents1.json', 'r') as f:
#     intents = json.load(f)

# def chatbot_response(user_input):
#     input_text = vectorizer.transform([user_input])
#     predicted_intent = best_model.predict(input_text)[0]

#     for intent in intents['intents']:
#         if intent['tag'] == predicted_intent:
#             response = random.choice(intent['responses'])
#             break

#     lines = response.split('\n')
#     formatted_response = "<br>".join([line for line in lines if line.strip()]) 

#     return formatted_response

# @app.route('/')
# def home():
#     return render_template('index.html')

# @app.route('/chat', methods=['POST'])
# def chat():
#     user_input = request.form['user_input']
#     response = chatbot_response(user_input)
#     return (response)

# if __name__ == '__main__':
#     app.run(debug=True)

import nltk
from nltk.stem import WordNetLemmatizer
import numpy as np
from tensorflow.keras.models import load_model
import random
import json
import pickle
from flask import Flask, render_template, request

# pip install nltk tensorflow flask numpy

app = Flask(__name__)

# Initialize the lemmatizer
lemmatizer = WordNetLemmatizer()

# Load the trained model
model = load_model('chatbot_model.h5')

# Load the intents data
with open('dataset/intents.json') as f:
    intents = json.load(f)

# Load the words and classes
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))

# Helper function to preprocess user input and predict response
def clean_up_sentence(sentence):
    # Tokenize the sentence and lemmatize each word
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words

def bow(sentence, words, show_details=True):
    # Create a bag of words based on the sentence and words
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    
    for s in sentence_words:
        for i, word in enumerate(words):
            if word == s:
                bag[i] = 1
    return(np.array(bag))

def chatbot_response(user_input):
    # Predict the intent of the user input
    p = bow(user_input, words, show_details=True)
    p = p.reshape(1, len(p))
    
    # Get model prediction
    prediction = model.predict(p)[0]
    predicted_class_index = np.argmax(prediction)
    
    # Get the predicted class tag
    predicted_class = classes[predicted_class_index]

    # Get a response based on the predicted intent
    for intent in intents['intents']:
        if intent['tag'] == predicted_class:
            response = random.choice(intent['responses'])
            break

    # Format the response to handle line breaks
    lines = response.split('\n')
    formatted_response = "<br>".join([line for line in lines if line.strip()])
    
    return formatted_response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']
    response = chatbot_response(user_input)
    return response

if __name__ == '__main__':
    app.run(debug=True)
