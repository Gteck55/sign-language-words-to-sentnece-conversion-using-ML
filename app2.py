from flask import Flask, render_template, Response, jsonify, request
import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math
import time
import pyttsx3

app = Flask(__name__)

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")
offset = 20
imgSize = 300
sentence = ""
current_word = None
word_start_time = None
is_audio_enabled = False  # Variable to track audio state
audio_engine = pyttsx3.init()  # Initialize text-to-speech engine
labels = ["Hello", "I love you", "No", "Okay", "Please", "Thank you", "Yes"]

# Initialize text-to-speech engine
engine = pyttsx3.init()

def generate_frames():
    global cap, detector, classifier, offset, imgSize, current_word, word_start_time, sentence

    while True:
        success, img = cap.read()
        img_output = img.copy()
        hands, img = detector.findHands(img)
        
        if hands:
            hand = hands[0]
            x, y, w, h = hand['bbox']

            img_white = np.ones((imgSize, imgSize, 3), np.uint8) * 255

            img_crop = img[y - offset:y + h + offset, x - offset:x + w + offset]
            img_crop_shape = img_crop.shape

            # Add a check to ensure that imgCrop is not empty
            if img_crop_shape[0] > 0 and img_crop_shape[1] > 0:
                aspect_ratio = h / w

                if aspect_ratio > 1:
                    k = imgSize / h
                    w_cal = math.ceil(k * w)
                    img_resize = cv2.resize(img_crop, (w_cal, imgSize))
                    img_resize_shape = img_resize.shape
                    w_gap = math.ceil((imgSize - w_cal) / 2)
                    img_white[:, w_gap: w_cal + w_gap] = img_resize
                    prediction, index = classifier.getPrediction(img_white, draw=False)

                else:
                    k = imgSize / w
                    h_cal = math.ceil(k * h)
                    img_resize = cv2.resize(img_crop, (imgSize, h_cal))
                    img_resize_shape = img_resize.shape
                    h_gap = math.ceil((imgSize - h_cal) / 2)
                    img_white[h_gap: h_cal + h_gap, :] = img_resize
                    prediction, index = classifier.getPrediction(img_white, draw=False)

                if current_word != labels[index]:
                    current_word = labels[index]
                    word_start_time = time.time()
                else:
                    if current_word and time.time() - word_start_time >= 8:
                        sentence += current_word + " "
                        # Spell out the word when it's added to the sentence
                        if is_audio_enabled:
                            audio_engine.say(current_word)
                            audio_engine.runAndWait()
                        current_word = None
                        word_start_time = None
                
                cv2.rectangle(img_output, (x - offset, y - offset - 70), (x - offset + 400, y - offset + 60 - 50), (0, 255, 0),
                            cv2.FILLED)
                cv2.putText(img_output, labels[index], (x, y - 30), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 0, 0), 2)
                cv2.rectangle(img_output, (x - offset, y - offset), (x + w + offset, y + h + offset), (0, 255, 0), 4)

        ret, frame = cv2.imencode('.jpg', img_output)
        data = frame.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + data + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_sentence')
def get_sentence():
    global sentence, is_audio_enabled
    return jsonify({'sentence': sentence, 'is_audio_enabled': is_audio_enabled})

@app.route('/toggle_audio', methods=['POST'])
def toggle_audio():
    global is_audio_enabled
    is_audio_enabled = not is_audio_enabled  # Toggle audio state
    return jsonify({'success': True})

@app.route('/blog')
def blog():
    return render_template('blog.html')


@app.route('/clear', methods=['POST'])
def clear_sentence():
    global sentence
    sentence = ""
    return "Sentence cleared successfully"

if __name__ == "__main__":
    app.run(debug=True)

