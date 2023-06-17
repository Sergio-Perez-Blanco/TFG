from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
import numpy as np
import os

from keras.models import Sequential
from keras.layers import LSTM, Dense
from keras.callbacks import TensorBoard
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
from keras.callbacks import EarlyStopping

actions = np.array(['Safe', 'Dangerous'])
label_map = {label: num for num, label in enumerate(actions)}

# CAMBIAR PATHS A LAS CARPETAS QUE TOQUE
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('MP_Data_5_SC')
sequences, labels = [], []
for danger in actions:
    contenido = os.listdir(DATA_PATH+"/"+danger)
    rango_tot = os.listdir(DATA_PATH + "/"+"Dangerous")
    for sequence in range(len(rango_tot)):
        window = []
        contenido2 = os.listdir(DATA_PATH+"/" + danger+"/" + contenido[sequence])
        if len(contenido2) == 5:
            for frame_num in range(len(contenido2)):
                res = np.load(os.path.join(DATA_PATH, danger, contenido[sequence],contenido2[frame_num]))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[danger])



X = np.array(sequences)
print(X.shape)
y = to_categorical(labels).astype(int)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.05)

log_dir = os.path.join('Logs')
early_stopping = EarlyStopping(monitor='loss', patience=11)
tb_callback = TensorBoard(log_dir=log_dir)
model = Sequential()
model.add(LSTM(64, return_sequences=True, activation='relu', input_shape=(5,258))) # 258 // 1662
#model.add(Dropout(0.2))
model.add(LSTM(128, return_sequences=True, activation='relu'))
#model.add(Dropout(0.5))
model.add(LSTM(64, return_sequences=False, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(actions.shape[0], activation='softmax'))

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

model.fit(X_train, y_train, epochs=130, callbacks=[tb_callback, early_stopping])

model.summary()

yhat = model.predict(X_test)

ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat))

print(accuracy_score(ytrue, yhat))

model.save('MP_modelo_5_SC_2.h5')