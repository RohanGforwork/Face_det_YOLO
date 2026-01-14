import cv2
from ultralytics import YOLO
import time
import numpy as np
from scipy.spatial import distance as dist
from imutils import face_utils
import dlib

MODEL_PATH = r"C:\Users\Roshan\runs\classify\train16\weights\last.pt"
model = YOLO(MODEL_PATH)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

EYE_AR_THRESH = 0.45
EYE_AR_CONSEC_FRAMES = 5

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

recognized_person = None
start_time = None
frame_border_color = (0, 0, 255)
CONFIDENCE_THRESHOLD = 0.93

person_blink_count = {}
person_recognition_time = {}
present_list = []

last_face_position = None
movement_threshold = 20

cap = cv2.VideoCapture(0)

EAR_CHANGE_THRESHOLD = 0.03
previous_ear = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame)

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    current_face_position = None

    if len(rects) > 1:
        cv2.putText(frame, "Multiple people detected, please look at the camera one at a time.", 
                    (20, frame.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        frame_border_color = (0, 0, 255)
        recognized_person = "Unknown"
    else:
        if results and results[0].probs is not None:
            name_dict = results[0].names
            probs = results[0].probs.data.cpu().numpy()
            top_class_index = np.argmax(probs)
            top_class = name_dict[top_class_index]
            top_confidence = probs[top_class_index]

            if top_confidence >= CONFIDENCE_THRESHOLD:
                if recognized_person == top_class:
                    if start_time is None:
                        start_time = time.time()
                    elif time.time() - start_time >= 3:
                        if top_class not in person_recognition_time:
                            person_recognition_time[top_class] = time.time()
                else:
                    recognized_person = top_class
                    start_time = time.time()
                    if top_class not in person_blink_count:
                        person_blink_count[top_class] = 0
                        person_recognition_time[top_class] = 0
            else:
                recognized_person = "Unknown"
                start_time = None
                frame_border_color = (0, 0, 255)

        for rect in rects:
            shape = predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)

            current_face_position = (rect.left(), rect.top(), rect.right(), rect.bottom())

            if last_face_position:
                x_diff = abs(current_face_position[0] - last_face_position[0])
                y_diff = abs(current_face_position[1] - last_face_position[1])

                if x_diff > movement_threshold or y_diff > movement_threshold:
                    face_moving = True
                else:
                    face_moving = False
            else:
                face_moving = True

            last_face_position = current_face_position

            if face_moving:
                leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
                rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]

                leftEAR = eye_aspect_ratio(leftEye)
                rightEAR = eye_aspect_ratio(rightEye)

                ear = (leftEAR + rightEAR) / 2.0

                if recognized_person != "Unknown":
                    if previous_ear is not None:
                        ear_diff = abs(previous_ear - ear)
                        if ear_diff > EAR_CHANGE_THRESHOLD and ear < EYE_AR_THRESH:
                            person_blink_count[recognized_person] += 1
                            print(f"Blink detected for {recognized_person}")

                    previous_ear = ear

                    if person_blink_count[recognized_person] >= 3:
                        if recognized_person not in present_list and time.time() - person_recognition_time.get(recognized_person, 0) >= 3:
                            present_list.append(recognized_person)
                            frame_border_color = (0, 255, 0)

    recognized_text = f"Recognized: {recognized_person}, Blinks: {person_blink_count.get(recognized_person, 0)}"
    cv2.putText(frame, recognized_text, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, frame_border_color, 2)

    frame = cv2.copyMakeBorder(frame, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=frame_border_color)

    cv2.imshow("YOLO Real-Time Recognition with Liveness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print("Present People:", present_list)
