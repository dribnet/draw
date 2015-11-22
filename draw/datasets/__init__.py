from __future__ import division

import fuel
import os
import sys

supported_datasets = ['bmnist', 'silhouettes']
# ToDo: # 'mnist' and 'tfd' are not normalized (0<= x <=1.)

def get_data(data_name, channels=None, size=None, width=None, height=None):
    print("==============> Fetching Data")
    if data_name == 'mnist':
        from fuel.datasets import MNIST
        img_size = (28, 28)
        channels = 1
        data_train = MNIST(which_sets=["train"], sources=['features'])
        data_valid = MNIST(which_sets=["test"], sources=['features'])
        data_test  = MNIST(which_sets=["test"], sources=['features'])
    elif data_name == 'bmnist':
        from fuel.datasets.binarized_mnist import BinarizedMNIST
        img_size = (28, 28)
        channels = 1
        data_train = BinarizedMNIST(which_sets=['train'], sources=['features'])
        data_valid = BinarizedMNIST(which_sets=['valid'], sources=['features'])
        data_test  = BinarizedMNIST(which_sets=['test'], sources=['features'])
    elif data_name == 'cifar10':
        from fuel.datasets.cifar10 import CIFAR10
        img_size = (32, 32)
        channels = 3
        data_train = CIFAR10(which_sets=['train'], sources=['features'])
        data_valid = CIFAR10(which_sets=['test'], sources=['features'])
        data_test  = CIFAR10(which_sets=['test'], sources=['features'])
    elif data_name == 'svhn2':
        from fuel.datasets.svhn import SVHN
        img_size = (32, 32)
        channels = 3
        data_train = SVHN(which_format=2,which_sets=['train'], sources=['features'])
        data_valid = SVHN(which_format=2,which_sets=['test'], sources=['features'])
        data_test  = SVHN(which_format=2,which_sets=['test'], sources=['features'])
    elif data_name == 'silhouettes':
        from fuel.datasets.caltech101_silhouettes import CalTech101Silhouettes
        size = 28
        img_size = (size, size)
        channels = 1
        data_train = CalTech101Silhouettes(which_sets=['train'], size=size, sources=['features'])
        data_valid = CalTech101Silhouettes(which_sets=['valid'], size=size, sources=['features'])
        data_test  = CalTech101Silhouettes(which_sets=['test'], size=size, sources=['features'])
    elif data_name == 'tfd':
        from fuel.datasets.toronto_face_database import TorontoFaceDatabase
        img_size = (28, 28)
        channels = 1
        data_train = TorontoFaceDatabase(which_sets=['unlabeled'], size=size, sources=['features'])
        data_valid = TorontoFaceDatabase(which_sets=['valid'], size=size, sources=['features'])
        data_test  = TorontoFaceDatabase(which_sets=['test'], size=size, sources=['features'])
    else:
        from fuel.datasets import H5PYDataset
        from fuel.transformers.defaults import uint8_pixels_to_floatX
        if width is None and height is None and size is not None:
            width = size
            height = size
        if width is None or height is None or channels is None:
            raise Exception("Image size and channels must be specified for data source {}".format(data_name))
        img_size = (width, height)

        file_found = False
        cur_path_index = 0
        while not file_found and cur_path_index < len(fuel.config.data_path):
            data_path = fuel.config.data_path[cur_path_index]
            dataset_fname = os.path.join(data_path, data_name+'.hdf5')
            if os.path.isfile(dataset_fname):
                file_found = True
            cur_path_index = cur_path_index + 1
        if not file_found:
            print "data set {} not found, exiting".format(data_name)
            sys.exit(1)

        data_train = H5PYDataset(dataset_fname, which_sets=['train'], sources=['features'])
        try:
            data_valid = H5PYDataset(dataset_fname, which_sets=['valid'], sources=['features'])
        except ValueError as e:
            data_valid = H5PYDataset(dataset_fname, which_sets=['test'], sources=['features'])
        data_test = H5PYDataset(dataset_fname, which_sets=['test'], sources=['features'])
        # this maybe could be an option - for now skip in lab space
        if not "-lab" in data_name:
            print("==============> Scaling features")
            data_train.default_transformers = uint8_pixels_to_floatX(('features',))
            data_valid.default_transformers = uint8_pixels_to_floatX(('features',))
            data_test.default_transformers = uint8_pixels_to_floatX(('features',))

    return img_size, channels, data_train, data_valid, data_test
