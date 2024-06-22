# sign-language-words-to-sentnece-conversion-using-ML
Sign Language Recognition with Flask

This project leverages computer vision and machine learning to recognize and interpret hand signs for basic phrases in sign language. The system captures video from a webcam, processes the hand gestures using the cvzone library, and classifies them into predefined labels using a trained model.

Key Features:

    Hand Detection and Tracking: Utilizes cvzone.HandTrackingModule to detect and track hand movements.
    Gesture Classification: Uses a trained classifier (keras_model.h5 and labels.txt) to recognize specific gestures.
    Real-time Video Streaming: Streams live video with gesture annotations via Flask web server.
    Text-to-Speech: Optionally converts recognized gestures to spoken words using pyttsx3.
    Toggleable Audio Output: Allows enabling or disabling audio output dynamically through the web interface.
    Dynamic Sentence Formation: Accumulates recognized words into sentences over time.

Technologies Used:

    Python
    Flask
    OpenCV
    cvzone
    NumPy
    pyttsx3

Usage:

    Install the required dependencies.
    Run the Flask server.
    Access the web interface to view live video feed and interact with the application.

Installation:
Follow the instructions in the README to set up the environment and run the application.
