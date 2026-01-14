import cv2
import os
import time

def record_and_process_video(name, base_path):

    if not os.path.exists(base_path):
        raise ValueError(f"The base path '{base_path}' does not exist. Please create it first or specify the correct path.")

    person_path = os.path.join(base_path, name)
    if not os.path.exists(person_path):
        os.makedirs(person_path)

    cap = cv2.VideoCapture(0)

    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_file = os.path.join(person_path, 'video.avi')
    out = cv2.VideoWriter(output_file, fourcc, 20.0, (640, 480))

    start_time = time.time()
    end_time = start_time + 10

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if time.time() >= end_time:
            break

        cv2.imshow('Video Feed', frame)

        out.write(frame)

        frame_path = os.path.join(person_path, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_path, frame)

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame_path = os.path.join(person_path, f'gray_frame_{frame_count}.jpg')
        cv2.imwrite(gray_frame_path, gray_frame)

        resized_frame = cv2.resize(frame, (224, 224))
        resized_frame_path = os.path.join(person_path, f'resized_frame_{frame_count}.jpg')
        cv2.imwrite(resized_frame_path, resized_frame)

        frame_count += 1

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyAllWindows()

record_and_process_video('farhan', r'persons')
