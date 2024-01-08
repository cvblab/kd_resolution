import os
import torch
import torchvision
import numpy as np
import random
import pandas as pd

from skimage import io
from skimage.filters import threshold_li
from skimage.measure import regionprops, label
from sklearn.utils.class_weight import compute_class_weight
from skimage import img_as_float, img_as_ubyte
from PIL import Image
import imutils


class Dataset(object):
    def __init__(self, dir_dataset, data_frame, classes, input_shape=(3, 512, 512), labels=5,
                 augmentation=False, preallocate=True):

        self.dir_dataset = dir_dataset
        self.data_frame = data_frame
        self.resize = torchvision.transforms.Resize((input_shape[-2], input_shape[-1]))
        self.labels = labels
        self.augmentation = augmentation
        self.preallocate = preallocate
        self.input_shape = input_shape

        # Load and pre-process images
        if self.data_frame is not None:
            self.images = data_frame['image_name'].to_list()
            self.gt = np.array(data_frame[classes])
        else:
            self.images = os.listdir(dir_dataset)
            self.images = random.sample(self.images, 500000)
            self.gt = np.zeros((len(self.images), len(classes)))

            # Remove other files
            self.images = [self.images[i] for i in range(self.images.__len__()) if 'Thumbs.db' not in self.images[i]]

        if self.preallocate:
            # Pre-load images
            self.X = np.zeros((len(self.images), input_shape[0], input_shape[1], input_shape[2])).astype(np.float32)
        self.Y = np.zeros((len(self.images), labels))
        for iImage in np.arange(0, len(self.images)):
            print(str(iImage) + '/' + str(len(self.images)), end='\r')

            if self.preallocate:
                id = self.dir_dataset + self.images[iImage]

                im = np.array(io.imread(id))
                im = imutils.resize(im, height=self.input_shape[1])
                im = np.transpose(im, (2, 0, 1))

                # Intensity normalization
                im = im / 255

                self.X[iImage, :, :, :] = im
            if self.data_frame is not None:
                self.Y[iImage, :] = self.gt[iImage, :]

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.images)

    def __getitem__(self, index):
        'Generates one sample of data'

        if self.preallocate:
            x = self.X[index, :, :, :]
        else:

            id = self.dir_dataset + self.images[index]

            im = np.array(io.imread(id))
            im = imutils.resize(im, height=self.input_shape[1])
            im = np.transpose(im, (2, 0, 1))

            # Intensity normalization
            im = im / 255
            x = im

        y = self.Y[index, :]

        x = torch.tensor(x).float()
        y = torch.tensor(y).float()

        return x, y


class Generator(object):
    def __init__(self, dataset, batch_size, shuffle=True, balance=False, augmentation=False):

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.augmentation = augmentation
        self.indexes = np.arange(0, len(self.dataset.images))
        self._idx = 0
        self.balance = balance

        if self.balance:
            self.indexes = balance_dataset(self.indexes, np.argmax(self.dataset.Y, axis=1),
                                           labels=dataset.labels)
        self._reset()

    def __len__(self):
        N = len(self.dataset.images)
        b = self.batch_size
        return N // b

    def __iter__(self):
        return self

    def __next__(self):

        if self._idx + self.batch_size > len(self.dataset.images):
            self._reset()
            raise StopIteration()

        # Load images and include into the batch
        X, Y = [], []
        for i in range(self._idx, self._idx + self.batch_size):
            x, y = self.dataset.__getitem__(self.indexes[i])

            X.append(x.unsqueeze(0))
            Y.append(y.unsqueeze(0))

        # Update index iterator
        self._idx += self.batch_size

        X = torch.cat(X, 0)
        Y = torch.cat(Y, 0)

        if self.augmentation:
            X = self.dataset.transforms(X)

        return X, Y

    def _reset(self):
        if self.shuffle:
            random.shuffle(self.indexes)
        self._idx = 0


def balance_dataset(indexes, Y, fixed=False, labels=5):
    classes = list(np.arange(0, labels))
    counts = np.bincount(Y)

    if fixed:
        upsampling = [1, 2, 1, 6, 7]
    else:
        upsampling = [round(np.max(counts)/counts[iClass]) for iClass in classes]

    indexes_new = []
    for iClass in classes:
        if upsampling[iClass] == 1:
            indexes_iclass = indexes[Y == classes[iClass]]
        else:
            indexes_iclass = np.random.choice(indexes[Y == classes[iClass]], counts[iClass]*upsampling[iClass])
        indexes_new.extend(indexes_iclass)

    indexes_new = np.array(indexes_new)

    return indexes_new


def categorical2onehot(y, labels):

    y_onehot = np.zeros((len(list(y)), labels))
    for i in np.arange(0, len(list(y))):
        y_onehot[i, y[i]] = 1

    return y_onehot