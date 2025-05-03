import random
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import pickle


intents = {
    "intents": [
        {"tag": "greeting", "patterns": ["Hi", "Hey", "Hello", "Good day", "Good morning"], "responses": ["Hello!", "Hi there!", "Greetings!"]},
        {"tag": "goodbye", "patterns": ["Bye", "See you", "Goodbye", "Catch you later"], "responses": ["Goodbye!", "See you later!", "Bye!"]},
        {"tag": "thanks", "patterns": ["Thanks", "Thank you", "Much appreciated", "Thanks a lot"], "responses": ["You're welcome!", "No problem!", "Anytime!"]},
        {"tag": "hours", "patterns": ["What are your hours?", "When are you open?", "Tell me your opening time", "What time do you close?"], "responses": ["We're open 9am-5pm every day!"]},
        {"tag": "name", "patterns": ["What is your name?", "Who are you?", "Tell me your name"], "responses": ["I'm ChatPy, your friendly assistant!"]},
        {"tag": "location", "patterns": ["Where are you located?", "What is your location?", "Where can I find you?"], "responses": ["We are located at 123 Main Street, Your City!"]},
        {"tag": "default", "patterns": ["", "What?", "I don't understand", "Help", "Can you assist me?"], "responses": ["Sorry, I didn't understand that. Can you rephrase?", "I'm not sure about that. Could you clarify?"]}
    ]
}


X = []
y = []

for intent in intents["intents"]:
    for pattern in intent["patterns"]:
        X.append(pattern)
        y.append(intent["tag"])


model = make_pipeline(TfidfVectorizer(), MultinomialNB())
model.fit(X, y)


with open('chat_model.pkl', 'wb') as file:
    pickle.dump(model, file)


def load_model():
    with open('chat_model.pkl', 'rb') as file:
        return pickle.load(file)


def chat():
    print("ChatPy: Hello! Type 'quit' to exit.")
    while True:
        user_input = input("You: ")
        if user_input.lower() == "quit":
            print("ChatPy: Goodbye!")
            break
      
        user_input = re.sub(r'\s+', ' ', user_input).strip().lower()

    
        prediction = model.predict([user_input])[0]
        
     
        response_found = False
        for intent in intents["intents"]:
            if intent["tag"] == prediction:
                print("ChatPy:", random.choice(intent["responses"]))
                response_found = True
                break
        
       
        if not response_found:
            print("ChatPy:", random.choice([response for response in intents["intents"] if response["tag"] == "default"][0]["responses"]))

if __name__ == "__main__":
    chat()
