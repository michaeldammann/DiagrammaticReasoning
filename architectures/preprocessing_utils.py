import pickle
import numpy as np
import cv2
import os
import numpy as np
from os.path import isfile, join
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, TimeDistributed, Flatten, LSTM, GRU, Dense, Reshape, \
    Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop, Adam
from keras.models import load_model
from PIL import Image
import argparse

from matplotlib import pyplot as plt

def newseqs_timeseries(seqy):
    newseqy = np.zeros((len(seqy), 64, 64, 1))
    for i in range(0, len(seqy)):
        newseqy[i] = seqy[i][4]
    return newseqy

def rearrange_fullycnn(seq):
    seq=np.reshape(seq,(len(seq), 5, 64, 64))
    newseq=np.zeros((len(seq),64,64,5))
    for i in range(0, len(seq)):
        images=seq[i]
        for j in range(0,len(images)): #5
            newseq[i,:,:,j]=images[j]
    return newseq

def preprocess_seqs(directory_path, split=(.8, .9), is_timeseries=False, is_fullycnn=False):
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


    trainseqy = newseqs_timeseries(trainseqy)
    valseqy = newseqs_timeseries(valseqy)
    testseqy = newseqs_timeseries(testseqy)

    if is_fullycnn:
        trainseq = rearrange_fullycnn(trainseq)
        valseq = rearrange_fullycnn(valseq)
        testseq = rearrange_fullycnn(testseq)

        trainseq = np.reshape(np.reshape(trainseq, (len(trainseq), 5, 64, 64)), (len(trainseq), 64, 64, 5))
        valseq = np.reshape(np.reshape(valseq, (len(valseq), 5, 64, 64)), (len(valseq), 64, 64, 5))
        testseq = np.reshape(np.reshape(testseq, (len(testseq), 5, 64, 64)), (len(testseq), 64, 64, 5))

    a = [trainseq, trainseqy, valseq, valseqy, testseq, testseqy]
    return a