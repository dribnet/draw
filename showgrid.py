#!/usr/bin/env python
from __future__ import division, print_function

import logging
import numpy as np

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

import os
import theano
import theano.tensor as T
import fuel
import ipdb
import time
import cPickle as pickle

from argparse import ArgumentParser
from theano import tensor

from fuel.streams import DataStream
from fuel.schemes import SequentialScheme
from fuel.transformers import Flatten

from blocks.algorithms import GradientDescent, CompositeRule, StepClipping, RMSProp, Adam
from blocks.bricks import Tanh, Identity
from blocks.bricks.cost import BinaryCrossEntropy
from blocks.bricks.recurrent import SimpleRecurrent, LSTM
from blocks.initialization import Constant, IsotropicGaussian, Orthogonal
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import PARAMETER
from blocks.monitoring import aggregation
from blocks.extensions import FinishAfter, Timing, Printing, ProgressBar
from blocks.extensions.saveload import Checkpoint
from blocks.extensions.monitoring import DataStreamMonitoring, TrainingDataMonitoring
from blocks.main_loop import MainLoop
from blocks.model import Model

import draw.datasets as datasets
from draw.draw import *
from draw.labcolor import scaled_lab2rgb

from PIL import Image, ImageDraw 
from blocks.main_loop import MainLoop
from blocks.model import AbstractModel
from blocks.config import config

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

def render_grid(rows, cols, width, height, channels, samples, lab):
    full_width = width
    full_height = height
    # width = 32
    # height = 32
    width = 48
    height = 48
    total_height = rows * height + (rows - 1)
    total_width  = cols * width + (cols - 1)

    # logic for borders
    I = np.zeros((channels, total_height, total_width))
    three_rows_over = height * 1 + 3
    three_cols_over = width * 1 + 3

    I.fill(1.0)
    # I[:,three_rows_over-1,:].fill(0)
    # I[:,total_height-three_rows_over,:].fill(0)
    # I[:,three_rows_over-1:total_height-three_rows_over,three_cols_over-1].fill(0)
    # I[:,three_rows_over-1:total_height-three_rows_over,total_width-three_cols_over].fill(0)

    # random samples in the center
    sample_rows = rows
    sample_cols = cols
    if samples is not None:
        samprect = samples[-1].reshape( (sample_cols, sample_rows, channels, full_height, full_width) )
        for c in range(sample_cols):
            cur_c = c
            for r in range(sample_rows):
                cur_r = r
                offset_y, offset_x = cur_r * height + cur_r, cur_c * width + cur_c
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = samprect[c,r,0:channels,0:height,0:width]

    # convert everything to image
    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I)

    if(lab):
        out = scaled_lab2rgb(out)

    out = (255 * out).astype(np.uint8)

    return Image.fromarray(out)

def encode(imga, normed_offset, normed_size):
    z = np.copy(imga)
    img = Image.fromarray(z)
    draw = ImageDraw.Draw(img)
    offset_box = (0, 35, 12, 47)
    offset_box_inner = (2, 37, 10, 45)
    if normed_offset == -1:
        # print("fill == 255")
        draw.rectangle(offset_box,fill=1)
    else:
        draw.rectangle(offset_box,fill=0)
        draw.rectangle(offset_box_inner,fill=normed_offset/255.0)
    size_box = (24, 35, 36, 47)
    size_box_inner = (26, 37, 34, 45)
    if normed_size == -1:
        draw.rectangle(size_box,fill=1)
    else:
        draw.rectangle(size_box,fill=0)
        draw.rectangle(size_box_inner,fill=normed_size/255.0)
    return np.asarray(img)

# build up a set of reconstructed pairs (training or test)
def build_it(data_stream, rows, cols, model, channels, image_size):
    num = rows * cols

    draw = model.get_top_bricks()[0]

    x = tensor.matrix('features')
    reconstruct_function = theano.function([x], draw.reconstruct(x))
    iterator = data_stream.get_epoch_iterator(as_dict=True)
    samples = []
    target_shape = (channels, image_size[0], image_size[1])
    target_pairs = (2, image_size[0] * image_size[1])

    # first build a list of num images in datastream
    datastream_images = []
    for batch in iterator:
        for entry in batch['features']:
            datastream_images.append(entry.reshape(target_pairs))
            if len(datastream_images) >= num: break
        if len(datastream_images) >= num: break

    input_shape = tuple([1] + list(datastream_images[0][0].shape))

    source_ims = []
    source_ims.append(datastream_images[16][1].reshape(input_shape))
    source_ims.append(datastream_images[2][1].reshape(input_shape))
    source_ims.append(datastream_images[8][1].reshape(input_shape))
    source_ims.append(datastream_images[1][1].reshape(input_shape))
    for c in range(cols):
        for r in range(rows):
            normed_offset = int(255.0 * (c - 1.0) / (cols - 3.0))
            normed_size = int(255.0 * (r - 1.0) / (rows - 3.0))
            if normed_offset < 128 and normed_size < 128:
                source_im = source_ims[0]
            elif normed_offset >= 128 and normed_size < 128:
                source_im = source_ims[1]
            elif normed_offset < 128 and normed_size >= 128:
                source_im = source_ims[2]
            else:
                source_im = source_ims[3]

            if (c == 0 or c == cols-1) and (r == 0 or r == rows-1):
                # print("case c=0,r=0")
                samples.append([source_im.reshape(target_shape)])
            else:
                if c == 0 or c == cols-1:
                    # print("case c=0: ", normed_size)
                    encoded_im = encode(source_im.reshape(image_size[0], image_size[1]), -1, normed_size)
                elif r == 0 or r == rows-1:
                    # print("case r=0:", normed_offset)
                    encoded_im = encode(source_im.reshape(image_size[0], image_size[1]), normed_offset, -1)
                else:
                    # print("case normal:", normed_offset, normed_size)
                    encoded_im = encode(source_im.reshape(image_size[0], image_size[1]), normed_offset, normed_size)
                encoded_im = encoded_im.reshape(input_shape)
                recon_im, kterms = reconstruct_function(encoded_im)
                samples.append([encoded_im.reshape(target_shape)])
                # samples.append([recon_im.reshape(target_shape)])

    samples = np.array(samples)
    target_shape = (1, num, channels, image_size[0], image_size[1])
    return samples.reshape(target_shape)

# generate entire dash and save
def generate_dash(model, subdir, image_size, channels, lab, rows, cols, train_stream, test_stream):
    samples = None

    logging.info("Generating example")
    samples = build_it(test_stream, rows, cols, model, channels, image_size)
    logging.info("Rendering grid")
    imgrid = render_grid(rows, cols, image_size[0], image_size[1], channels, samples, lab)
    imgrid.save("{0}/showgrid.png".format(subdir))

# load model and run based on args
def unpack_and_run(subdir, args):
    rows = args.rows
    cols = args.cols
    lab = args.lab

    logging.info("Loading file %s..." % args.model_file)
    with open(args.model_file, "rb") as f:
        p = pickle.load(f)

    if isinstance(p, AbstractModel):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return

    dataset = args.dataset
    logging.info("Loading dataset %s..." % dataset)
    image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset, args.channels, args.size, args.width, args.height)
    train_stream = Flatten(DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, 1)))
    test_stream  = Flatten(DataStream.default_stream(data_test,  iteration_scheme=SequentialScheme(data_test.num_examples, 1)))

    generate_dash(model, subdir, image_size, channels, lab, rows, cols, train_stream, test_stream)

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
    parser.add_argument("--cols", type=int,
                default=12, help="grid cols")
    parser.add_argument("--rows", type=int,
                default=8, help="grid rows")
    parser.add_argument('--lab', dest='lab', default=False,
                help="Lab Colorspace", action='store_true')

    args = parser.parse_args()

    subdir = "dashboard"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    unpack_and_run(subdir, args)
