import cv2
import numpy as np
import os
import mediapipe as mp
import json

"""Keypoints using MP Holistic"""
mp_holistic = mp.solutions.holistic  # Holistic model
mp_drawing = mp.solutions.drawing_utils  # Drawing utilities
# contenido = os.listdir('E:/Uni/Videos_TFG/Videos')

# Normal
# contenido = os.listdir('/home/sergio/.virtualenvs/Videos_TFG/Videos')

# Test
contenido = os.listdir('/home/sergio/.virtualenvs/Videos_TFG/videos_test')


def mediapipe_detection(image, model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # COLOR CONVERSION BGR 2 RGB
    image.flags.writeable = False  # Image is no longer writeable
    results = model.process(image)  # Make prediction
    image.flags.writeable = True  # Image is now writeable
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)  # COLOR COVERSION RGB 2 BGR
    return image, results


def draw_landmarks(image, results):
    # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION) # Draw face connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)  # Draw pose connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks,
                              mp_holistic.HAND_CONNECTIONS)  # Draw right hand connections


def draw_styled_landmarks(image, results):
    # Draw face connections
    mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
                              mp_drawing.DrawingSpec(color=(80, 110, 10), thickness=1, circle_radius=1),
                              mp_drawing.DrawingSpec(color=(80, 256, 121), thickness=1, circle_radius=1)
                              )
    # Draw pose connections
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(80, 22, 10), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(80, 44, 121), thickness=2, circle_radius=2)
                              )
    # Draw left hand connections
    mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(121, 44, 250), thickness=2, circle_radius=2)
                              )
    # Draw right hand connections
    mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=4),
                              mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                              )


"""Extract Keypoint Values"""

"""Con face"""


def extract_keypoints(results):
    pose = np.array([[res.x, res.y, res.z, res.visibility] for res in
                     results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33 * 4)
    face = np.array([[res.x, res.y, res.z] for res in
                     results.face_landmarks.landmark]).flatten() if results.face_landmarks else np.zeros(468 * 3)
    lh = np.array([[res.x, res.y, res.z] for res in
                   results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21 * 3)
    rh = np.array([[res.x, res.y, res.z] for res in
                   results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(
        21 * 3)
    return np.concatenate([pose, face, lh, rh])


"""Sin cara"""


# def extract_keypoints(results):
#       pose = np.array([[res.x, res.y, res.z, res.visibility] for res in results.pose_landmarks.landmark]).flatten() if results.pose_landmarks else np.zeros(33*4)
#       lh = np.array([[res.x, res.y, res.z] for res in results.left_hand_landmarks.landmark]).flatten() if results.left_hand_landmarks else np.zeros(21*3)
#       rh = np.array([[res.x, res.y, res.z] for res in results.right_hand_landmarks.landmark]).flatten() if results.right_hand_landmarks else np.zeros(21*3)
#       return np.concatenate([pose, lh, rh])

def safe_dangerous(n_frame, nombre):
    with open('jsons/' + nombre) as f:
        data1 = json.load(f)
    lugar = ""
    encontrado = False
    i = 0
    while not encontrado and i < len(data1["vcd"]["actions"]):
        for j in (data1["vcd"]["actions"][str(i)]["frame_intervals"]):
            if j["frame_start"] < n_frame and n_frame < j["frame_end"]:
                lugar = i
                encontrado = True
                break
        i += 1

    if lugar == "":
        lugar = i - 1

    if data1["vcd"]["actions"][str(lugar)]["type"] == "driver_actions/talking_to_passenger" or \
            data1["vcd"]["actions"][str(lugar)]["type"] == "driver_actions/reach_side" or \
            data1["vcd"]["actions"][str(lugar)]["type"] == "driver_actions/radio" or \
            data1["vcd"]["actions"][str(lugar)]["type"] == "driver_actions/safe_drive" or \
            data1["vcd"]["actions"][str(lugar)]["type"] == "hands_using_wheel/only_left" or \
            data1["vcd"]["actions"][str(lugar)]["type"] == "hands_using_wheel/both" or \
            data1["vcd"]["actions"][str(lugar)]["type"] == "talking/talking" or data1["vcd"]["actions"][str(lugar)][
        "type"] == "gaze_on_road/looking_road":
        # if data1["vcd"]["actions"][str(lugar)]["type"] == "gaze_on_road/looking_road" or data1["vcd"]["actions"][str(lugar)]["type"] == "driver_actions/safe_drive" or data1["vcd"]["actions"][str(lugar)]["type"] == "hands_using_wheel/both":
        return "Safe"

    elif data1["vcd"]["actions"][str(lugar)]["type"] == "driver_actions/unclassified":
        return "unclassified"

    else:
        return "Dangerous"
    # or data1["vcd"]["actions"][str(lugar)]["type"] == "hands_using_wheel/none"
    # or data1["vcd"]["actions"][str(lugar)]["type"] == "gaze_on_road/not_looking_road"
    # or data1["vcd"]["actions"][str(lugar)]["type"] == "driver_actions/drinking"


# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Test_30')

num2 = 1
N_frame = 0
sec_frame = 0
for video in contenido:
    num = 1
    print("Procesando video:", num2, video)
    cap = cv2.VideoCapture('videos_test/' + video)
    contenido_jsons = os.listdir('/home/sergio/.virtualenvs/Videos_TFG/jsons')
    for i in contenido_jsons:
        if i[0:10] == video[0:10]:
            nombre = i
    # CUANDO TE QEDAS A MEDIAS EN LA EXTRACCION
    hacer = True
    # n_vid = os.listdir(DATA_PATH+"/Safe")
    # for i in n_vid:
    #     if i[0:8] == video[0:8]:
    #         hacer = False
    #         break

    # Set mediapipe model
    if hacer != False:
        with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while cap.isOpened():
                # Read feed
                ret, frame = cap.read()

                # Make detections
                image, results = mediapipe_detection(frame, holistic)

                # Draw landmarks
                draw_styled_landmarks(image, results)

                # NEW Export keypoints
                keypoints = extract_keypoints(results)

                if N_frame == 0:
                    clasi = safe_dangerous(num, nombre)
                    path = DATA_PATH + "/" + clasi + str(video[0:8]) + str(sec_frame)

                    os.makedirs(os.path.join(DATA_PATH, clasi, str(video[0:8]) + str(sec_frame)))

                if N_frame < 30:
                    npy_path = os.path.join(DATA_PATH, clasi, str(video[0:8]) + str(sec_frame), video[0:7] + str(num))

                N_frame += 1
                if N_frame == 30:  # numero de frames por secuencia
                    N_frame = 0
                    sec_frame += 1
                num += 1

                np.save(npy_path, keypoints)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q') or cap.get(cv2.CAP_PROP_POS_FRAMES) == cap.get(
                        cv2.CAP_PROP_FRAME_COUNT):
                    break
            N_frame = 0
            sec_frame = 0
            num2 += 1
            cap.release()
            cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()