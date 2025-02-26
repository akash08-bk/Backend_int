import speech_recognition as sr
from gtts import gTTS
import transformers
import os
import datetime
import numpy as np
import requests

class ChatBot():
    def __init__(self, name):
        self.text = None
        self.name = name

    def speech_to_text(self):
        recognizer = sr.Recognizer()
        try:
            with sr.Microphone() as mic:
                print("Checking microphone...")
                # Adjust the energy threshold as needed
                recognizer.adjust_for_ambient_noise(mic)
                print("Microphone is ready.")
                print("Listening...")
                audio = recognizer.listen(mic)
                print("Audio captured successfully.")
        except sr.RequestError as e:
            print("Failed to access the microphone; {0}".format(e))
            return
        except sr.UnknownValueError:
            print("Failed to capture audio; unknown value error.")
            return
        
        try:
            self.text = recognizer.recognize_google(audio)
            print("User said:", self.text)
            try:
                data = {'text': self.text}
                url = 'http://localhost:5000/python-data'  # Update the URL to match your server endpoint
                response = requests.post(url, json=data)

                if response.status_code == 200:
                    print("Data sent to server successfully")
                else:
                    print("Failed to send data to server:", response.text)
            except Exception as e:
                print("Error occurred while sending data to server:", e)
        except Exception as e:
            print("Error in speech recognition:", e)

if __name__ == "__main__":
    flag = 0
    text = "Maya"
    ai = ChatBot(text)

    nlp = transformers.pipeline("text-generation", model="microsoft/DialoGPT-medium")

    os.environ["TOKENIZERS_PARALLELISM"] = "true"

    while True:
        try:
            url = 'http://localhost:5000/get_flag'
            response = requests.get(url)

            if response.status_code == 200:
                data = response.json()
                flag = data.get("flag")
                print("Flag received:", flag)
            else:
                print("Failed to get data from server:", response.text)

        except Exception as e:
            print("Error occurred while getting data from server:", e)

        if int(flag) != 0:
            ai.speech_to_text()

            # Add further actions based on the recognized text, if necessary
            # Example: You can send the text to a conversational model for response

            # Uncomment the following code if you want to use the chatbot response
            """
            if ai.text:
                chat = nlp(transformers.Conversation(ai.text), pad_token_id=50256)
                res = str(chat)
                res = res[res.find("bot >> ") + 6:].strip()
                ai.text_to_speech(res)
            """
