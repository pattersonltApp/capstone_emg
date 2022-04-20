import datetime
import os
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers import PReLU
from keras.layers.convolutional import Conv1D
from keras.layers import MaxPooling1D
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.callbacks import ModelCheckpoint
from keras import optimizers
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.python.keras.callbacks import EarlyStopping

output_path = 'models/'

def load_file(filepath):
    df = pd.read_csv(filepath, header=None)
    return df.values


def load_group(directory):
    """
    load_group(directory)
    Loads all files in a directory and appends them to a list
        which gets converted into an np array and returned.
    """
    data = []
    labels = []
    for filename in os.listdir(directory):
         if os.path.isfile(directory + '/' + filename):
            values = load_file(directory + '/' + filename)
            data.append(values)
            labels.append(filename[0])
    data = np.array(data)
    labels = np.array([ord(label) - 97 for label in labels])
    return data, labels


def main():
    # Load training data.
    train_directory = 'BA_data/cleaned_data/train'
    train, train_labels = load_group(train_directory)

    train2, train_labels2 = load_group('MJ_data/cleaned_data/train')
    train = np.concatenate((train, train2))
    train_labels = np.concatenate((train_labels, train_labels2))

    # Load test data.
    test_directory = 'BA_data/cleaned_data/test'
    test, test_labels = load_group(test_directory)

    test2, test_labels2 = load_group('MJ_data/cleaned_data/train')
    test = np.concatenate((test, test2))
    test_labels = np.concatenate((test_labels, test_labels2))

    #optimizer = tf.keras.optimizers.Adam(lr=0.000011288378916846883)
    optimizer = 'Adamax'

    batch_size = 51
    model = Sequential()
    model.add(Conv1D(filters=32, kernel_size=4, activation='relu', input_shape=(4, 1000)))
    model.add(BatchNormalization())
    act = PReLU()
    model.add(act)
    model.add(Dropout(0.2))

    model.add(MaxPooling1D(padding='same'))

    model.add(Conv1D(filters=64, kernel_size=4, activation='relu', padding='same', input_shape=(4, 1000)))
    model.add(BatchNormalization())
    act = PReLU()
    model.add(act)
    model.add(Dropout(0.2))

    model.add(MaxPooling1D(padding='same'))

    model.add(Dense(256))
    act = PReLU()
    model.add(act)
    model.add(BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Dense(26, activation='softmax'))
    model.compile(loss='sparse_categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1)

    # setup early stopping with early stopping parameters
    early_stopping = EarlyStopping(monitor='val_accuracy', verbose=1, patience=100)
    # setup model checkpointing
    model_checkpoint = ModelCheckpoint(
        filepath=os.path.join(output_path, f'{datetime.datetime.now()}.h5'),  # always overwrite the existing model
        save_weights_only=False, save_freq='epoch',
        save_best_only=True, monitor='val_accuracy', verbose=1)  # only save models that improve the 'monitored' value
    callbacks = [early_stopping, model_checkpoint, tensorboard_callback]

    #model.fit(train, train_labels, epochs=100000, batch_size=5, verbose=1, callbacks=[checkpoint])
    model.fit(train, train_labels, validation_data=(test, test_labels), epochs=10000, batch_size=32, verbose=1, callbacks = callbacks)
    #_, accuracy = model.evaluate(test, test_labels, batch_size=25, verbose=1)
    #print(accuracy)


if __name__ == '__main__':
    main()
