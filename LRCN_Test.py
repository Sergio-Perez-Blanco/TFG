import os
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import matplotlib.pyplot as plt


def plot_metric(model_training_history, metric_name_1, metric_name_2, plot_name):
    '''
    This function will plot the metrics passed to it in a graph.
    Args:
        model_training_history: A history object containing a record of training and validation
                                loss values and metrics values at successive epochs
        metric_name_1:          The name of the first metric that needs to be plotted in the graph.
        metric_name_2:          The name of the second metric that needs to be plotted in the graph.
        plot_name:              The title of the graph.
    '''

    # Get metric values using metric names as identifiers.
    metric_value_1 = model_training_history.history[metric_name_1]
    metric_value_2 = model_training_history.history[metric_name_2]

    # Construct a range object which will be used as x-axis (horizontal plane) of the graph.
    epochs = range(len(metric_value_1))

    # Plot the Graph.
    plt.plot(epochs, metric_value_1, 'blue', label=metric_name_1)
    plt.plot(epochs, metric_value_2, 'red', label=metric_name_2)

    # Add title to the plot.
    plt.title(str(plot_name))

    # Add legend to the plot.
    plt.legend()


seed_constant = 27
# Specify the height and width to which each video frame will be resized in our dataset.
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64

# Specify the number of frames of a video that will be fed to the model as one sequence.
SEQUENCE_LENGTH = 5
actions = np.array(['Safe', 'Dangerous'])
label_map = {label: num for num, label in enumerate(actions)}

# CAMBIAR PATHS A LAS CARPETAS QUE TOQUE
# Path for exported data, numpy arrays
DATA_PATH = os.path.join('CNN_Test_5')
sequences, labels = [], []
for danger in actions:
    contenido = os.listdir(DATA_PATH + "/" + danger)
    rango_tot = os.listdir(DATA_PATH + "/" + "Dangerous")
    for sequence in range(len(contenido)):
        window = []
        contenido2 = os.listdir(DATA_PATH + "/" + danger + "/" + contenido[sequence])
        if len(contenido2) == SEQUENCE_LENGTH:
            for frame_num in range(len(contenido2)):
                res = np.load(os.path.join(DATA_PATH, danger, contenido[sequence], contenido2[frame_num]))
                window.append(res)
            sequences.append(window)
            labels.append(label_map[danger])

X = np.asarray(sequences)

one_hot_encoded_labels = to_categorical(labels)
features_train, features_test, labels_train, labels_test = train_test_split(X, one_hot_encoded_labels,
                                                                            test_size=0.99, shuffle=True,
                                                                            random_state=seed_constant)
model = tf.keras.models.load_model(
    'LRCN_model__5_Date_Time_2023_04_26__14_05_30___Loss_0.41150423884391785___Accuracy_0.8195121884346008.h5')
predicted_labels = model.predict(features_test)

true_labels = np.argmax(labels_test, axis=1)
predicted_labels = np.argmax(predicted_labels, axis=1)

cm = confusion_matrix(true_labels, predicted_labels)
print(cm)

accuracy = np.sum(np.diag(cm)) / np.sum(cm)
print(f'Accuracy: {accuracy * 100:.2f} %')