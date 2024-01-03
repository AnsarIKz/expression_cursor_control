import cv2
import dlib
import numpy as np
import pyautogui
import time
import math

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("./model/shape_predictor_68_face_landmarks.dat")

pyautogui.FAILSAFE = False
screen_width, screen_height = pyautogui.size()

# Калибровка
calibration_points = {
    'left': None,
    'right': None,
    'up': None,
    'down': None
}

calibration_directions = ["left", "right", "up", "down"]
calibration_direction = calibration_directions[0]

cap = cv2.VideoCapture(0)

def face_calibration():
    global calibration_direction
    while not all(calibration_points.values()):
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = detector(gray)
        if faces:
            landmarks = predictor(gray, faces[0])

        
            for i in range(68):  # Iterate through all 68 face landmarks
                x, y = landmarks.part(i).x, landmarks.part(i).y
                cv2.putText(frame, f"{i}", (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)

            
            left_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
            right_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])

            # Вектор, соединяющий центры глаз
            vector_between_eyes = right_eye - left_eye

            # Угол между глазами с использованием арктангенса и векторного произведения
            angle_radians = np.arctan2(vector_between_eyes[1], vector_between_eyes[0])

            # Преобразование угла в положительное значение
            angle_radians = (angle_radians + 2 * np.pi) % (2 * np.pi)

            # Перевод угла из радиан в градусы

            nose_nasal = (landmarks.part(30).x, landmarks.part(30).y)
            nose_septum = (landmarks.part(27).x, landmarks.part(27).y)
            

            # horizontal_tilt = np.degrees(np.arctan2(nose_nasal[1], right_eye[0] - left_eye[0]))
            horizontal_tilt = math.degrees(angle_radians)
            vertical_tilt = nose_septum[1] - nose_nasal[1]

            cv2.putText(frame, f"Horizontal Tilt: {horizontal_tilt:.2f} degrees", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            cv2.putText(frame, f"Vertical Tilt: {vertical_tilt:.2f} degrees", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.putText(frame, f"Calibrating {calibration_direction}. Press 'Space' to save calibration point.", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

            cv2.imshow("Head Tilt Tracking", frame)

            key = cv2.waitKey(1)
            

            if key == ord(' '):
                if calibration_direction in ["left", "right"]:
                    calibration_points[calibration_direction] = horizontal_tilt
                else:
                    calibration_points[calibration_direction] = vertical_tilt
                current_index = calibration_directions.index(calibration_direction)
                next_index = (current_index + 1) % len(calibration_directions)
                calibration_direction = calibration_directions[next_index]
    cv2.destroyAllWindows()


face_calibration()
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = detector(gray)
    if faces:
        landmarks = predictor(gray, faces[0])

        left_eye = np.array([landmarks.part(45).x, landmarks.part(45).y])
        right_eye = np.array([landmarks.part(36).x, landmarks.part(36).y])

            # Вектор, соединяющий центры глаз
        vector_between_eyes = right_eye - left_eye

            # Угол между глазами с использованием арктангенса и векторного произведения
        angle_radians = np.arctan2(vector_between_eyes[1], vector_between_eyes[0])

            # Преобразование угла в положительное значение
        angle_radians = (angle_radians + 2 * np.pi) % (2 * np.pi)

            # Перевод угла из радиан в градусы

        nose_nasal = (landmarks.part(30).x, landmarks.part(30).y)
        nose_septum = (landmarks.part(27).x, landmarks.part(27).y)
            
        horizontal_tilt = math.degrees(angle_radians)
        vertical_tilt = nose_septum[1] - nose_nasal[1]

        
        x_movement = np.interp(horizontal_tilt, [calibration_points['right'], calibration_points['left']], [0, screen_width])
        y_movement = np.interp(np.abs(vertical_tilt), [np.abs(calibration_points['up']), np.abs(calibration_points['down'])], [0, screen_height])

        print(horizontal_tilt, calibration_points['left'], calibration_points['right'], x_movement)
        pyautogui.moveTo(x_movement, y_movement, 0.2)  


        cv2.imshow('Window', frame)
        
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

        



# Освобождение ресурсов
cap.release()
cv2.destroyAllWindows()