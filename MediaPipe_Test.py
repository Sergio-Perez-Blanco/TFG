from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import os

import tensorflow as tf
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score

actions = np.array(['Safe', 'Dangerous'])
label_map = {label: num for num, label in enumerate(actions)}

DATA_PATH = os.path.join('MP_Test_30')
sequences, labels = [], []
for danger in actions:
    contenido = os.listdir(DATA_PATH +"/"+danger)
    rango_tot = os.listdir(DATA_PATH + "/"+"Dangerous")
    for sequence in range(len(contenido)):
        window = []
        contenido2 = os.listdir(DATA_PATH + "/" + danger+"/" + contenido[sequence])
        if len(contenido2) == 30:
            for frame_num in range(len(contenido2)):
                res = np.load(os.path.join(DATA_PATH, danger, contenido[sequence],contenido2[frame_num]))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[danger])



X = np.array(sequences)
print(X.shape)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.99)
model = tf.keras.models.load_model('MP_Modelos/MP_modelo_30x.h5')
yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))