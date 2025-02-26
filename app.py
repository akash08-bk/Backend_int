import os
from flask import Flask, make_response, render_template, Response, request
import cv2
import numpy as np
import requests
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.preprocessing.image import img_to_array
from trial import ChatBot
from flask_cors import CORS
import transformers
import shutil
import sys

current_length = 500
app = Flask(__name__)
CORS(app)




# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
          activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

model.load_weights('model.h5')

# Load the haarcascade classifier
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Define the emotion labels
emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful",
                3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

confidence = {}
flag = 0

# def check_and_append_labels(emotion_labels_list):
#     global current_length
                    
#     print(current_length)
                    
#     if current_length != 0:
#         current_length -= 1
#         with open(file_path, "a") as file:
#             for label in emotion_labels_list:
#                 file.write(label + "\n")
#                 check_and_append_labels(emotion_labels_list)
#     else:
#         max_emotion, confidence_level=analyze_emotions(file_path)
#         if confidence_level == "High":
#             print("HIGH")
#                 # if os.path.exists("E:/work/client/HIGH.txt"):
#                 #     os.remove("E:/work/client/HIGH.txt")
#                 #     shutil.copy("HIGH.txt", "E:/work/client")
#         else:
#             print("LOW")
#                     # if os.path.exists("E:/work/client/LOW.txt"):
#                     #     os.remove("E:/work/client/LOW.txt")
#                     #     shutil.copy("LOW.txt", "E:/work/client")
#         sys.exit()
            
def analyze_emotions(file_path, question):
    emotions = {'Neutral': 0, 'Happy': 0, 'Sad': 0, 'Angry': 0, "Disgusted": 0, "Fearful": 0, "Surprised": 0}

    with open(file_path, 'r') as file:
        for line in file:
            emotion = line.strip()
            emotions[emotion] += 1

        total_lines = sum(emotions.values())
        percentages = {emotion: (count / total_lines) * 100 for emotion, count in emotions.items()}

        max_emotion = max(percentages, key=percentages.get)
        if max_emotion in ['Happy']:
            confidence_level = 'HIGH'
        elif max_emotion in ['Neutral']:
            confidence_level = 'MEDIUM'
        else:
            confidence_level = 'LOW'

        confidence[f"Question_{question}"] = confidence_level

def check_flag():
    global flag
    try:
        url = 'http://localhost:5000/get_flag_camera'
        response = requests.get(url)
        
        if response.status_code == 200:
            # print("Data received from server.")
            data = response.json()
            flag = data.get("flag")
            print(flag)
        else:
            print("Failed to get data from server:", response.text)

    except Exception as e:
        print("Error occurred while geting data from server:", e)
    
def set_flag():
    try:
        data = { "flag" : "0" }
        print(data)
        response = requests.post(url="http://localhost:5000/stop-camera", json=data);
        if response.status_code == 200:
            print("Data sent to server successfully")
        else:
            print("Failed to send data to server:", response.text)
    except Exception as e:
        print('An error occurred', e)
        
def detect_emotions():
    global flag
    file_path = ["emotion_labels_1.txt", "emotion_labels_2.txt", "emotion_labels_3.txt"]
    camera = cv2.VideoCapture(0)
    count = 1
    emotion_labels_list=[]
    while True:
       
        _, frame = camera.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype("float") / 255.0
            roi_gray = img_to_array(roi_gray)
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)
            prediction = model.predict(roi_gray)[0]
            maxindex = int(np.argmax(prediction))
            emotion_label = emotion_dict[maxindex]
          
            emotion_label = emotion_dict[maxindex]
            print(emotion_label)
            emotion_labels_list.append(emotion_label)
            
            print(" length :",len(emotion_labels_list))
            check_flag()
            
            if  int(flag) == 1 and len(emotion_labels_list) >= 70:
                print("inside creating flag ")
                with open(file_path[count-1], 'w') as file:
                    for number in emotion_labels_list:
                        file.write(f"{number}\n")
                analyze_emotions(file_path=file_path[count-1], question=count)
                set_flag()
                count += 1
                emotion_labels_list.clear()
                
            if count > len(file_path):
                try:
                    data = {"confidence" : confidence}
                    print(data)
                    url = 'http://localhost:5000/confidence-level'  
                    response = requests.post(url, json=data)
                    set_flag()
                    if response.status_code == 200:
                        print("Data sent to server successfully")
                        sys.exit()
                    else:
                        print("Failed to send data to server:", response.text)
                except Exception as e:
                    print("Error occurred while sending data to server:", e)
        
                
                
            print(emotion_label)

            cv2.putText(frame, emotion_label, (x + 20, y - 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2,
                        cv2.LINE_AA)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

            _, jpeg = cv2.imencode('.jpg', frame)
            frame = jpeg.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


@app.route('/audio')
def index1():
    return render_template('audio.html')
# Route for the home page

# ...existing code...




@app.route('/video')
def index2():
    return render_template('index.html')


@app.route('/')
def index3():
    return render_template('home.html')


@app.route('/contact')
def index4():
    return render_template('contact.html')


# Route for video feed
@app.route('/video_feed')
def video_feed(): 
        return  Response(detect_emotions(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


if __name__ == '__main__':
    app.run(debug=True)
