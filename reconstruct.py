#!/usr/bin/env python
from __future__ import division, print_function

import logging
import numpy as np

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import sys
import os
import theano
import theano.tensor as T
import fuel
import ipdb
import time
import cPickle as pickle

from argparse import ArgumentParser

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

import draw.datasets as datasets
from draw.draw import *
from draw.labcolor import scaled_lab2rgb

from PIL import Image
from scipy.misc import imread, imsave
from scipy.misc import imresize 

import modelutil

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

def render_grid(cols, width, height, channels, bottom_pairs, lab):
    rows = 2
    # total_height = rows * height + (rows - 1)
    # total_width  = cols * width + (cols - 1)
    total_height = rows * height
    total_width  = cols * width

    # logic for borders
    I = np.zeros((channels, total_height, total_width))

    # test pairs at bottom
    if bottom_pairs is not None:
        for c in range(cols):
            for i in range(2):
                r = i
                # offset_y, offset_x = r * height + r, c * width + c
                offset_y, offset_x = r * height, c * width
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = bottom_pairs[c][i].reshape(channels,height,width)

    # convert everything to image
    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I)

    if(lab):
        out = scaled_lab2rgb(out)

    out = (255 * out).astype(np.uint8)

    return Image.fromarray(out)

def render_all(rows, cols, width, height, channels, bottom_pairs, lab):
    total_height = rows * height
    total_width  = cols * width

    # logic for borders
    I = np.zeros((channels, total_height, total_width))

    # test pairs at bottom
    for r in range(rows):
        for c in range(cols):
            i = 1
            # offset_y, offset_x = r * height + r, c * width + c
            offset_y, offset_x = r * height, c * width
            I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = bottom_pairs[r*cols+c][i].reshape(channels,height,width)

    # convert everything to image
    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I)

    if(lab):
        out = scaled_lab2rgb(out)

    out = (255 * out).astype(np.uint8)

    return Image.fromarray(out)

def printchar(i):
    n = i + 1
    if(n % 10 != 0):
        print(".", end="")
    else:
        n = n / 10
        if(n % 10 != 0):
            print("{:d}".format(int(n%10)), end="")
        else:
            n = n / 10
            if(n % 10 != 0):
                print("{:d}".format(int(n%10)), end="")
            else:
                print("\n{:d}".format(i + 1))
    sys.stdout.flush()

# build up a set of reconstructed pairs (training or test)
def build_reconstruct_pairs(data_stream, num, model, channels, image_size):
    f = modelutil.build_reconstruct_function(model)
    iterator = data_stream.get_epoch_iterator(as_dict=True)
    pairs = []
    target_shape = (channels, image_size[0], image_size[1])

    # first build a list of num images in datastream
    datastream_images = []
    for batch in iterator:
        for entry in batch['features']:
            datastream_images.append(entry)
            if len(datastream_images) >= num: break
        if len(datastream_images) >= num: break

    input_shape = tuple([1] + list(datastream_images[0].shape))

    for i in range(num):
        next_im = datastream_images[i].reshape(input_shape)
        printchar(i)
        recon_im, kterms = modelutil.reconstruct_image(f, next_im)
        pairs.append([next_im.reshape(target_shape),
                      recon_im.reshape(target_shape)])

    print("")
    return pairs

# build up a set of reconstructed pairs (training or test)
def build_reconstruct_image(fname, model, channels, image_size):
    rawim = imread(fname);
    print("shape: {}".format(rawim.shape))
    im_height, im_width = rawim.shape

    f = modelutil.build_reconstruct_function(model)

    pairs = []
    target_shape = (channels, image_size[0], image_size[1])
    height, width = image_size

    # first build a list of num images in datastream
    datastream_images = []
    steps_y = int(im_height / height)
    steps_x = int(im_width / width)
    # while cur_x + width <= im_width and len(datastream_images) < num:
    for j in range(steps_y):
        cur_y = j * height
        for i in range(steps_x):
            cur_x = i * width
            entry = (rawim[cur_y:cur_y+height, cur_x:cur_x+width] / 255.0).astype('float32').reshape(width*height)
            datastream_images.append(entry)

    input_shape = tuple([1] + list(datastream_images[0].shape))

    for i in range(steps_x*steps_y):
        next_im = datastream_images[i].reshape(input_shape)
        printchar(i)
        recon_im, kterms = modelutil.reconstruct_image(f, next_im)
        pairs.append([next_im.reshape(target_shape),
                      recon_im.reshape(target_shape)])

    print("")
    return steps_y, steps_x, pairs


def build_reconstruct_imagec(fname, model, channels, image_size):
    rawim = imread(fname);
    print("shape: {}".format(rawim.shape))
    im_height, im_width, im_channels = rawim.shape
    mixedim = np.asarray([rawim[:,:,0], rawim[:,:,1], rawim[:,:,2]])
    print("newshape: {}".format(mixedim.shape))

    f = modelutil.build_reconstruct_function(model)

    pairs = []
    target_shape = (channels, image_size[0], image_size[1])
    height, width = image_size

    # first build a list of num images in datastream
    datastream_images = []
    steps_y = int(im_height / height)
    steps_x = int(im_width / width)
    # while cur_x + width <= im_width and len(datastream_images) < num:
    for j in range(steps_y):
        cur_y = j * height
        for i in range(steps_x):
            cur_x = i * width
            entry = (mixedim[0:im_channels, cur_y:cur_y+height, cur_x:cur_x+width] / 255.0).astype('float32').reshape(im_channels*width*height)
            datastream_images.append(entry)

    input_shape = tuple([1] + list(datastream_images[0].shape))

    for i in range(steps_x*steps_y):
        next_im = datastream_images[i].reshape(input_shape)
        printchar(i)
        recon_im, kterms = modelutil.reconstruct_image(f, next_im)
        pairs.append([next_im.reshape(target_shape),
                      recon_im.reshape(target_shape)])

    print("")
    return steps_y, steps_x, pairs

def generate_tiles(model, subdir, image_size, channels, tilefile):
    lab = False
    logging.info("Generating tiles from {}".format(tilefile))
    if(channels == 1):
        rows, cols, test_pairs = build_reconstruct_image(tilefile, model, channels, image_size)
    else:
        rows, cols, test_pairs = build_reconstruct_imagec(tilefile, model, channels, image_size)
    logging.info("Rendering pairs")
    imgrid = render_grid(rows*cols, image_size[0], image_size[1], channels, test_pairs, lab)
    imgrid.save("{0}/tile_pairs.png".format(subdir))
    logging.info("Rendering grid")
    imgrid = render_all(rows, cols, image_size[0], image_size[1], channels, test_pairs, lab)
    imgrid.save("{0}/tile_grid.png".format(subdir))

def generate_stream(model, subdir, image_size, channels, input_stream, num, prefix):
    lab = False
    logging.info("Generating pairs")
    reconstruct_pairs = build_reconstruct_pairs(input_stream, num, model, channels, image_size)
    logging.info("Rendering pairs")
    imgrid = render_grid(num, image_size[0], image_size[1], channels, reconstruct_pairs, lab)
    imgrid.save("{0}/{1}_pairs.png".format(subdir, prefix))

# load model and run based on args
def unpack_and_run(subdir, args):
    lab = args.lab

    logging.info("Loading file %s..." % args.model_file)
    model = modelutil.load_file(args.model_file)

    image_size = (args.width, args.height)
    channels = args.channels

    if(args.trainpairs or args.testpairs):
        dataset = args.dataset
        logging.info("Loading dataset %s..." % dataset)
        image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset, args.channels, args.size, args.width, args.height)
        if(args.trainpairs):
            train_stream = Flatten(DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, 1)))
            generate_stream(model, subdir, image_size, channels, train_stream, args.numpairs, "train")
        if(args.testpairs):
            test_stream = Flatten(DataStream.default_stream(data_test, iteration_scheme=SequentialScheme(data_test.num_examples, 1)))
            generate_stream(model, subdir, image_size, channels, test_stream, args.numpairs, "test")

    if(args.tilefile):
        generate_tiles(model, subdir, image_size, channels, args.tilefile)

# main - setup args, make output directory, and the run
if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model_file", help="filename of a pickled DRAW model")
    parser.add_argument("--dataset", type=str, dest="dataset",
                default="bmnist", help="Dataset to use: [bmnist|mnist|cifar10]")
    parser.add_argument("--channels", type=int,
                default=None, help="number of channels (if custom dataset)")
    parser.add_argument("--size", type=int,
                default=None, help="image size (if custom dataset)")
    parser.add_argument("--width", type=int,
                default=None, help="image width (if custom dataset)")
    parser.add_argument("--height", type=int,
                default=None, help="image height (if custom dataset)")
    parser.add_argument('--lab', dest='lab', default=False,
                help="Lab Colorspace", action='store_true')
    parser.add_argument('--trainpairs', dest='trainpairs', default=False,
                help="Generate Training Pairs", action='store_true')
    parser.add_argument('--testpairs', dest='testpairs', default=False,
                help="Generate Testing Pairs", action='store_true')
    parser.add_argument("--numpairs", type=int,
                default=12, help="Number of dataset pairs to generate")
    parser.add_argument("--tilefile", type=str, dest="tilefile",
                default=None, help="Input file for tiling reconstruction")
    parser.add_argument("--subdir", type=str, dest="subdir",
                default="reconstruct", help="subdirectory output")

    args = parser.parse_args()

    subdir = args.subdir
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    unpack_and_run(subdir, args)
