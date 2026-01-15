from flask import Flask, render_template, Response, jsonify
import cv2 as cv
import os
from flask_cors import CORS
from datetime import datetime
import atexit
import time
import numpy as np
import threading
import dlib
from imutils import face_utils
import pandas as pd
import smtplib
from email.message import EmailMessage

app = Flask(__name__)
CORS(app)

MODEL_PATH = "front_face/face_trained.yml"
HAAR_CASCADE_PATH = "front_face/haar_face.xml"
PERSONS_DIR = "persons"

haar_cascade = cv.CascadeClassifier(HAAR_CASCADE_PATH)
if haar_cascade.empty():
    raise IOError(f"Haar cascade XML file not found at {HAAR_CASCADE_PATH}")

face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

people = [name for name in os.listdir(PERSONS_DIR) if os.path.isdir(os.path.join(PERSONS_DIR, name))]

recognized_names = []
detected_faces = False
all_captured_names = set()

class VideoStream:
    def __init__(self):
        self.cap = cv.VideoCapture(0)
        self.frame = None
        self.lock = threading.Lock()
        self.running = True

        self.thread = threading.Thread(target=self.update_frames, daemon=True)
        self.thread.start()

    def update_frames(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.frame = frame

    def get_frame(self):
        with self.lock:
            return self.frame

    def stop(self):
        self.running = False
        self.cap.release()

video_stream = VideoStream()

# Blink detection setup
predictor_path = "shape_predictor_68_face_landmarks.dat"
shape_predictor = dlib.shape_predictor(predictor_path)

# Constants for blink detection
EYE_AR_THRESH = 0.2  # Eye Aspect Ratio threshold
EYE_AR_CONSEC_FRAMES = 3  # Minimum consecutive frames for a blink

# Indices for the eyes in the 68-point facial landmarks
(left_start, left_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(right_start, right_end) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

blink_counter = 0
blink_confirmed = False

# FPS tracking
fps_start_time = time.time()
fps_frame_count = 0
current_fps = 0
TARGET_FPS = 30  # Target frame rate cap


def calculate_eye_aspect_ratio(eye):
    """Calculate the Eye Aspect Ratio (EAR) for blink detection"""
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear



def generate_frames():
    global recognized_names, detected_faces, all_captured_names, blink_counter, blink_confirmed
    global fps_start_time, fps_frame_count, current_fps

    recognized_person = None
    start_time = None
    frame_time = 1.0 / TARGET_FPS  # Time per frame for target FPS
    frame_border_color = (0, 0, 255)

    while True:
        frame = video_stream.get_frame()
        if frame is None:
            continue

        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        # Use Haar cascade for face detection (faster than dlib)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        detected_faces = len(faces_rect) > 0
        recognized_names_temp = []

        for (x, y, w, h) in faces_rect:
            face_roi = gray[y:y + h, x:x + w]

            # Predict face label
            label, confidence = face_recognizer.predict(face_roi)

            if 0 <= label < len(people) and confidence < 100:
                name = people[label]

                # Blink detection using dlib landmarks
                dlib_rect = dlib.rectangle(x, y, x + w, y + h)
                shape = shape_predictor(gray, dlib_rect)
                shape = face_utils.shape_to_np(shape)

                left_eye = shape[left_start:left_end]
                right_eye = shape[right_start:right_end]

                left_ear = calculate_eye_aspect_ratio(left_eye)
                right_ear = calculate_eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2.0

                if ear < EYE_AR_THRESH:
                    blink_counter += 1
                else:
                    if blink_counter >= EYE_AR_CONSEC_FRAMES:
                        blink_confirmed = True
                    blink_counter = 0

                # Only proceed with recognition if blink is confirmed
                if blink_confirmed:
                    if recognized_person == name:
                        if start_time is None:
                            start_time = time.time()
                        elif time.time() - start_time >= 3:
                            if name not in recognized_names_temp:
                                recognized_names_temp.append(name)
                                all_captured_names.add(name)
                                frame_border_color = (0, 255, 0)
                    else:
                        recognized_person = name
                        start_time = time.time()
                else:
                    recognized_person = None
                    start_time = None

                # Draw rectangle around face
                color = (0, 255, 0) if blink_confirmed else (0, 255, 255)
                cv.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                status = "Blink!" if blink_confirmed else "Blink to verify"
                cv.putText(frame, f"{name} ({confidence:.0f}) - {status}", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                recognized_person = None
                start_time = None

                if "Unknown" not in recognized_names_temp:
                    recognized_names_temp.append("Unknown")
                    all_captured_names.add("Unknown")
                
                # Draw rectangle for unknown face
                cv.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv.putText(frame, "Unknown", (x, y - 10),
                           cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        recognized_names = recognized_names_temp

        if not recognized_names_temp:
            frame_border_color = (0, 0, 255)

        # Calculate FPS
        fps_frame_count += 1
        elapsed = time.time() - fps_start_time
        if elapsed >= 1.0:
            current_fps = fps_frame_count / elapsed
            fps_frame_count = 0
            fps_start_time = time.time()

        # Display FPS on frame
        cv.putText(frame, f"FPS: {current_fps:.1f}", (10, 30), 
                   cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Draw border
        frame = cv.copyMakeBorder(frame, 10, 10, 10, 10, cv.BORDER_CONSTANT, value=frame_border_color)
        ret, buffer = cv.imencode('.jpg', frame)

        if not ret:
            continue

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')

        # Frame rate limiting
        time.sleep(max(0, frame_time - (time.time() - fps_start_time) / max(fps_frame_count, 1)))
        
def get_class_info():
    # Read the Excel file
    file_path = 'class_timetable.xlsx'
    df = pd.read_excel(file_path)

    # Ensure the column names match the actual Excel structure
    df.columns = df.columns.str.strip().str.lower()

    # Convert 'start_time' and 'end_time' columns to datetime.time
    df['start_time'] = pd.to_datetime(df['start_time'], format='%H:%M:%S').dt.time
    df['end_time'] = pd.to_datetime(df['end_time'], format='%H:%M:%S').dt.time

    # Get the current time as a datetime.time object
    current_time = datetime.now().time()

    # Filter for the class based on current time between start and end
    current_class = df[(df['start_time'] <= current_time) & (df['end_time'] > current_time)]

    if not current_class.empty:
        return {
            'start_time': current_class.iloc[0]['start_time'].strftime('%H:%M:%S'),
            'end_time': current_class.iloc[0]['end_time'].strftime('%H:%M:%S'),
            'course_name': current_class.iloc[0]['course_name'],
            'instructor': current_class.iloc[0]['instructor']
        }
    else:
        return {'start_time': 'N/A', 'end_time': 'N/A', 'course_name': 'No Class', 'instructor': 'N/A'}
    
def update_attendance():
    global all_captured_names
    
    # Path to the attendance Excel file
    file_path = 'ATTENDANCE.xlsx'  
    updated_file_path = 'Updated_ATTENDANCE.xlsx'  
    
    # Load the attendance file
    attendance_df = pd.read_excel(file_path)
    
    # Ensure column names are standardized
    attendance_df.columns = attendance_df.columns.str.strip().str.lower()
    
    # Mark attendance
    def mark_attendance(row):
        if row['students'].strip().lower() in all_captured_names:
            return "Present"
        else:
            return "Absent"

    attendance_df['absent/present'] = attendance_df.apply(mark_attendance, axis=1)
    
    # Save the updated file
    attendance_df.to_excel(updated_file_path, index=False)
    print(f"Attendance updated and saved to {updated_file_path}")


def send_email_with_attendance():
    # Read the class timetable Excel file
    timetable_df = pd.read_excel('class_timetable.xlsx')
    timetable_df.columns = timetable_df.columns.str.strip().str.lower()

    # Convert start and end times to datetime.time
    timetable_df['start_time'] = pd.to_datetime(timetable_df['start_time'], format='%H:%M:%S').dt.time
    timetable_df['end_time'] = pd.to_datetime(timetable_df['end_time'], format='%H:%M:%S').dt.time

    # Get the current time
    current_time = datetime.now().time()

    # Identify the ongoing class based on current time
    current_class = timetable_df[
        (timetable_df['start_time'] <= current_time) & (timetable_df['end_time'] > current_time)
    ]

    if current_class.empty:
        print("No ongoing class. Email will not be sent.")
        return

    # Extract instructor email and course name
    instructor_email = current_class.iloc[0]['email']
    course_name = current_class.iloc[0]['course_name']

    if pd.isna(instructor_email):
        print(f"No email available for the instructor of {course_name}.")
        return

    # Prepare email
    email_sender = "hrudhaygirish9@gmail.com"  # Replace with your email
    email_password = "rkdy pxke xemr elep"  # Replace with your email password
    email_receiver = instructor_email.strip()

    subject = f"Attendance Report for {course_name}"
    body = f"Dear Instructor,\n\nPlease find attached the attendance report for your {course_name} class.\n\nRegards,\nAttendance System"

    # Create the email
    message = EmailMessage()
    message['From'] = email_sender
    message['To'] = email_receiver
    message['Subject'] = subject
    message.set_content(body)

    # Attach the attendance file
    with open('Updated_ATTENDANCE.xlsx', 'rb') as file:
        file_data = file.read()
        file_name = 'Updated_ATTENDANCE.xlsx'
        message.add_attachment(file_data, maintype='application', subtype='octet-stream', filename=file_name)

    # Send the email
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
            server.login(email_sender, email_password)
            server.send_message(message)
            print(f"Attendance report sent to {email_receiver}.")
    except Exception as e:
        print(f"Failed to send email: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recognized_names', methods=['GET'])
def get_recognized_names():
    global recognized_names, detected_faces
    return jsonify({
        "names": ", ".join(recognized_names),
        "detected": detected_faces
    })


@app.route('/get-class-info', methods=['GET'])
def class_info():
    try:
        info = get_class_info()
        return jsonify(info)
    except Exception as e:
        return jsonify({'error': str(e)})


def on_exit():
    global all_captured_names
    print("\n--- Session Summary ---")
    print("All captured names:")
    print(", ".join(all_captured_names))
    update_attendance()
    send_email_with_attendance()

atexit.register(on_exit)

if __name__ == '__main__':
    app.run(debug=False,threaded = True,host='0.0.0.0', port=5000)
