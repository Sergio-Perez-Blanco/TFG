import os
import cv2
import numpy as np

import json

# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 15

# Specify the directory containing the dataset.
DATASET_DIR = "/home/sergio/.virtualenvs/Videos_TFG/videos_test/"
contenido = os.listdir(DATASET_DIR)

# Specify the list containing the names of the classes used for training. Feel free to choose any set of classes.
CLASSES_LIST = ["Safe", "Dangerous", "uncalssified"]


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


DATA_PATH = os.path.join('CNN_Test_15')
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
        while cap.isOpened():

            # Reading the frame from the video.
            success, frame = cap.read()

            # Check if Video frame is not successfully read then break the loop
            if not success:
                break

            # Resize the Frame to fixed height and width.
            resized_frame = cv2.resize(frame, (IMAGE_HEIGHT, IMAGE_WIDTH))

            # Normalize the resized frame by dividing it with 255 so that each pixel value then lies between 0 and 1
            normalized_frame = resized_frame / 255

            if N_frame == 0:
                clasi = safe_dangerous(num, nombre)
                path = DATA_PATH + "/" + clasi + str(video[0:8]) + str(sec_frame)

                os.makedirs(os.path.join(DATA_PATH, clasi, str(video[0:8]) + str(sec_frame)))

            if N_frame < 15:
                npy_path = os.path.join(DATA_PATH, clasi, str(video[0:8]) + str(sec_frame), video[0:7] + str(num))

            N_frame += 1
            if N_frame == 15:  # numero de frames por secuencia
                N_frame = 0
                sec_frame += 1
            num += 1

            np.save(npy_path, normalized_frame)

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
