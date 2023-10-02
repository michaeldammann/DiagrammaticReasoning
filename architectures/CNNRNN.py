import pickle
import numpy as np
import cv2
import os
import numpy as np
from tensorflow import keras
import random
from os import listdir
from os.path import isfile, join
from skimage import io
from matplotlib import pyplot as plt
import imageio
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt
import keras
from keras import Sequential
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, TimeDistributed, Flatten, LSTM, GRU, Dense, Reshape, \
    Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.models import load_model
from PIL import Image
import argparse

from matplotlib import pyplot as plt


def newseqs(seqy):
    newseqy = np.zeros((len(seqy), 64, 64, 1))
    for i in range(0, len(seqy)):
        newseqy[i] = seqy[i][4]
    return newseqy


def preprocess_seqs(directory_path, split=(.8, .9)):
    allxseqs = []
    allyseqs = []

    # Initialize a counter to keep track of the number of folders
    seq_count = 0
    # Iterate over items in the directory
    for item in os.listdir(directory_path):
        item_path = os.path.join(directory_path, item)

        # Check if the item is a directory
        if os.path.isdir(item_path):
            seq_count += 1
    for i in range(seq_count):
        seqx = []
        seqy = []
        for j in range(0, 5):
            B = np.zeros((64, 64))
            A = (cv2.imread(directory_path + '/' + str(i) + '/' + str(j) + '.png', 0) / 255)
            nb = B.shape[0]
            na = A.shape[0]
            lower = (nb) // 2 - (na // 2)
            upper = (nb // 2) + (na // 2)
            B[lower:upper, lower:upper] = A
            imgi = np.reshape(B, (64, 64, 1))
            seqx.append(imgi)
        for k in range(1, 6):
            B = np.zeros((64, 64))
            A = (cv2.imread(directory_path + '/' + str(i) + '/' + str(k) + '.png', 0) / 255)
            nb = B.shape[0]
            na = A.shape[0]
            lower = (nb) // 2 - (na // 2)
            upper = (nb // 2) + (na // 2)
            B[lower:upper, lower:upper] = A
            imgii = np.reshape(B, (64, 64, 1))
            seqy.append(imgii)
        allxseqs.append(seqx)
        allyseqs.append(seqy)

    allxseqs = np.array(allxseqs)
    allyseqs = np.array(allyseqs)

    trainseq, valseq, testseq = np.split(allxseqs, [int(len(allxseqs) * split[0]), int(len(allxseqs) * split[1])])
    trainseqy, valseqy, testseqy = np.split(allyseqs, [int(len(allyseqs) * split[0]), int(len(allyseqs) * split[1])])

    trainseqy = newseqs(trainseqy)
    valseqy = newseqs(valseqy)
    testseqy = newseqs(testseqy)

    a = [trainseq, trainseqy, valseq, valseqy, testseq, testseqy]
    return a


def define_generator(input_img, lstm_units, latent_units, multiplier=32):
    x = TimeDistributed(Conv2D(1 * multiplier, (3, 3), padding='same'))(input_img)  # 32 x 32 x 32
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)  # 16 x 16 x 32
    x = TimeDistributed(Conv2D(2 * multiplier, (3, 3), padding='same'))(x)  # 16 x 16 x 64
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)  # 8 x 8 x 64
    x = TimeDistributed(Conv2D(4 * multiplier, (3, 3), padding='same'))(x)  # 8 x 8 x 128
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)
    x = TimeDistributed(Conv2D(8 * multiplier, (3, 3), padding='same'))(x)  # 8 x 8 x 128
    x = TimeDistributed(Activation('relu'))(x)
    x = TimeDistributed(BatchNormalization())(x)
    x = TimeDistributed(MaxPooling2D(pool_size=(2, 2)))(x)  #

    x = TimeDistributed(Flatten())(x)
    x = TimeDistributed(Dense(latent_units))(x)
    x = LSTM(lstm_units, return_sequences=False, name='LSTM')(x)
    x = Dense(4 * 4 * 8 * multiplier)(x)
    x = Reshape((4, 4, 8 * multiplier))(x)

    # decoder

    x = Conv2D(8 * multiplier, (3, 3), padding='same')(x)  # 4 x 4 x 128
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 16x16x128
    x = Conv2D(4 * multiplier, (3, 3), padding='same')(x)  # 4 x 4 x 128
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 16x16x128
    x = Conv2D(2 * multiplier, (3, 3), padding='same')(x)  # 4 x 4 x 128
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 16x16x128
    x = Conv2D(1 * multiplier, (3, 3), padding='same')(x)  # 4 x 4 x 128
    x = Activation('relu')(x)
    x = BatchNormalization()(x)
    x = UpSampling2D((2, 2))(x)  # 16x16x128
    x = Conv2D(1, (3, 3), padding='same')(x)  # 4 x 4 x 128
    x = Activation('sigmoid')(x)

    return x


def train_model(model_name, save_path, lstm_units, latent_units, trainseq, trainseqy, valseq, valseqy, batch_size=32,
                epochs=50, save_history=True, loss='mae'):
    input_img = Input(shape=(5, 64, 64, 1))
    generator = Model(input_img, define_generator(input_img, lstm_units, latent_units))
    generator.compile(loss=loss, optimizer=Adam())
    name = model_name

    os.makedirs(os.path.dirname(join(save_path, name)), exist_ok=True)

    filepath = join(save_path, name, 'bestmodel.hdf5')
    checkpoint = ModelCheckpoint(filepath, verbose=1, save_best_only=True)
    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    callbacks_list = [checkpoint, early_stopping]

    generator_train = generator.fit(x=trainseq, y=trainseqy, batch_size=batch_size, epochs=epochs, verbose=1,
                                    validation_data=(valseq, valseqy), callbacks=callbacks_list)

    if save_history:
        with open(join(save_path, name, 'historydic.pickle'), 'wb') as handle:
            pickle.dump(generator_train.history, handle)

        loss = generator_train.history['loss']
        val_loss = generator_train.history['val_loss']
        epochs = range(len(loss))
        f = plt.figure()
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
        plt.plot(epochs, loss, 'o', label='Training loss')
        plt.plot(epochs, val_loss, 'x', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()
        # plt.show()

        f.savefig(join(save_path, name, 'historyplot.pdf'), bbox_inches='tight')


def test_model(model_name, save_path, testseq, testseqy):
    name = model_name
    print('Loading best model')

    generator = load_model(join(save_path, name, 'bestmodel.hdf5'))

    print('Evaluating best model, saving test results')
    test_performance = generator.evaluate(x=testseq, y=testseqy, batch_size=32)
    text_file = open(join(save_path, name, 'test_performance.txt'), "w")
    text_file.write(str(test_performance))
    text_file.close()
    print('Test MAE: ' + str(test_performance))

    model = generator

    print('Saving test images')
    os.makedirs(join(save_path, name, 'testimages'), exist_ok=True)

    for i in range(0, min(len(testseq), 30)):
        os.makedirs(join(save_path, name, 'testimages', str(i)), exist_ok=True)
        for j in range(0, 7):
            if j == 5:
                Image.fromarray((np.clip(testseqy[i].reshape(64, 64), 0.0, 1.0) * 255).astype(np.uint8)).convert(
                    'RGB').save(
                    join(save_path, name, 'testimages', str(i), f'{j}.png'))
            elif j == 6:
                Image.fromarray(
                    (np.clip(model.predict(testseq)[i].reshape(64, 64), 0.0, 1.0) * 255).astype(np.uint8)).convert(
                    'RGB').save(
                    join(save_path, name, 'testimages', str(i), f'{j}.png'))
            else:
                Image.fromarray((np.clip(testseq[i][j].reshape(64, 64), 0.0, 1.0) * 255).astype(np.uint8)).convert(
                    'RGB').save(
                    join(save_path, name, 'testimages', str(i), f'{j}.png'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True, type=str)
    parser.add_argument('--model_path', required=True, type=str)
    parser.add_argument('--model_name', required=True, type=str)

    parser.add_argument('--lstm_units', required=False, default=16, type=int)
    parser.add_argument('--latent_units', required=False, default=32, type=int)
    parser.add_argument('--batch_size', required=False, default=32, type=int)
    parser.add_argument('--epochs', required=False, default=50, type=int)
    parser.add_argument('--loss', required=False, default='mae', type=str)

    parser.add_argument('--save_history', required=False, default=True, type=bool)

    args = parser.parse_args()

    preprocessed_data = preprocess_seqs(args.dataset_path)
    trainseq, trainseqy, valseq, valseqy, testseq, testseqy = preprocessed_data[0], preprocessed_data[1], \
                                                              preprocessed_data[2], preprocessed_data[3], \
                                                              preprocessed_data[4], preprocessed_data[5]

    train_model(args.model_name, args.model_path, args.lstm_units, args.latent_units, trainseq, trainseqy, valseq,
                valseqy, batch_size=args.batch_size, epochs=args.epochs, save_history=args.save_history, loss=args.loss)
    test_model(args.model_name, args.model_path, testseq, testseqy)
