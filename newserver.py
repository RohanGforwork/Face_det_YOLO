from flask import Flask, render_template, Response, jsonify
import cv2
import os
from flask_cors import CORS
import threading
import time
import queue

app = Flask(__name__)
CORS(app)

# Configuration
MODEL_PATH = "front_face/face_trained.yml"
HAAR_CASCADE_PATH = "front_face/haar_face.xml"
PERSONS_DIR = "persons"

# Initialize face detection
haar_cascade = cv2.CascadeClassifier(HAAR_CASCADE_PATH)
if haar_cascade.empty():
    raise IOError(f"Haar cascade XML file not found at {HAAR_CASCADE_PATH}")

face_recognizer = cv2.face.LBPHFaceRecognizer_create()
face_recognizer.read(MODEL_PATH)

people = [name for name in os.listdir(PERSONS_DIR) if os.path.isdir(os.path.join(PERSONS_DIR, name))]

# Shared variables
frame_queue = queue.Queue(maxsize=1)
recognized_names = []
detected_faces = False
last_frame_time = time.time()

def video_capture_thread():
    global recognized_names, detected_faces, last_frame_time
    
    # Try different backends
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Prefer DSHOW on Windows
    if not cap.isOpened():
        cap = cv2.VideoCapture(0)  # Fallback to default backend
    
    if not cap.isOpened():
        raise RuntimeError("Could not open video capture device")

    # Set reasonable resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame, retrying...")
            time.sleep(0.1)
            continue

        # Process frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces_rect = haar_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7)
        
        detected_faces = len(faces_rect) > 0
        current_names = []
        
        for (x, y, w, h) in faces_rect[:3]:  # Limit to 3 faces
            face_roi = gray[y:y+h, x:x+w]
            label, confidence = face_recognizer.predict(face_roi)
            
            name = people[label] if (0 <= label < len(people) and confidence < 100) else "Unknown"
            current_names.append(name)
            
            # Draw on frame
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, name, (x, y-10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        recognized_names = current_names
        last_frame_time = time.time()
        
        # Update frame queue
        if frame_queue.empty():
            try:
                frame_queue.put(frame.copy())
            except queue.Full:
                pass
        
        time.sleep(0.03)  # Control frame rate

    cap.release()

# Start video thread
video_thread = threading.Thread(target=video_capture_thread)
video_thread.daemon = True
video_thread.start()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def generate():
        while True:
            try:
                if not frame_queue.empty():
                    frame = frame_queue.get()
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
                else:
                    # Send black frame if no frames available
                    if time.time() - last_frame_time > 2:
                        black_frame = cv2.imread('static/black.jpg') if os.path.exists('static/black.jpg') else None
                        if black_frame is None:
                            black_frame = cv2.imread('black.jpg') if os.path.exists('black.jpg') else None
                        if black_frame is None:
                            black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
                        ret, buffer = cv2.imencode('.jpg', black_frame)
                        if ret:
                            yield (b'--frame\r\n'
                                   b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n\r\n')
                    time.sleep(0.1)
            except Exception as e:
                print(f"Stream error: {str(e)}")
                time.sleep(1)

    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/get_recognized_names')
def get_recognized_names():
    return jsonify({
        "names": ", ".join(recognized_names) if recognized_names else "No faces detected",
        "detected": detected_faces
    })

if __name__ == '__main__':
    app.run(debug=False, threaded=True)