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

#import blocks.extras

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
#from blocks.extras.extensions.plot import Plot
from blocks.main_loop import MainLoop
from blocks.model import Model

import draw.datasets as datasets
from draw.draw import *

from PIL import Image
from blocks.main_loop import MainLoop
from blocks.model import AbstractModel
from blocks.config import config

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

def render_grid(rows, cols, height, width, channels, top_pairs, bottom_pairs, samples, left_samp, right_samp):
    total_height = rows * height + (rows - 1)
    total_width  = cols * width + (cols - 1)

    I = np.zeros((channels, total_height, total_width))
    two_rows_over = height * 2 + 2
    three_cols_over = width * 3 + 3
    I.fill(0.25)
    I[:,two_rows_over-1,:].fill(1)
    I[:,total_height-two_rows_over,:].fill(1)
    I[:,two_rows_over-1:total_height-two_rows_over,three_cols_over-1].fill(1)
    I[:,two_rows_over-1:total_height-two_rows_over,total_width-three_cols_over].fill(1)
    # I[0][:two_rows_height].fill(0.2)
    # I[1][-two_rows_height:].fill(0.5)
    # I[0][:,:two_rows_height].fill(0.5)
    # I[0][:,-two_rows_height:].fill(0.5)

    if top_pairs is not None:
        for c in range(cols):
            for r in range(2):
                offset_y, offset_x = r * height + r, c * width + c
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = top_pairs[c][r]

    if bottom_pairs is not None:
        for c in range(cols):
            for i in range(2):
                r = rows - i - 1
                offset_y, offset_x = r * height + r, c * width + c
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = bottom_pairs[c][i]
    
    sample_rows = rows - 4
    sample_cols = cols - 6
    if samples is not None:
        samprect = samples[-1].reshape( (sample_cols, sample_rows, channels, width, height) )
        for c in range(sample_cols):
            cur_c = c + 3
            for r in range(sample_rows):
                cur_r = r + 2
                offset_y, offset_x = cur_r * height + cur_r, cur_c * width + cur_c
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = samprect[c][r]

    if left_samp is not None:
        for c in range(3):
            cur_c = c
            for r in range(sample_rows):
                cur_r = r + 2
                offset_y, offset_x = cur_r * height + cur_r, cur_c * width + cur_c
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = left_samp[r][c]

    if right_samp is not None:
        for c in range(3):
            cur_c = cols - c - 1
            for r in range(sample_rows):
                cur_r = r + 2
                offset_y, offset_x = cur_r * height + cur_r, cur_c * width + cur_c
                I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = right_samp[r][c]

    # cur_c = cols-2
    # for r in range(sample_rows):
    #     cur_r = r + 2
    #     offset_y, offset_x = cur_r * height + cur_r, cur_c * width + cur_c
    #     I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = right_samp[r]

    I = (255*I).astype(np.uint8)
    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I).astype(np.uint8)
    return Image.fromarray(out)

def build_reconstruct_pairs(data_stream, num, model, channels, size):
    draw = model.get_top_bricks()[0]
    x = tensor.matrix('features')
    reconstruct_function = theano.function([x], draw.reconstruct(x))    
    # EXPERIMENT reconstruct_function = theano.function([x], draw.reconstruct_flat(x))    
    iterator = data_stream.get_epoch_iterator(as_dict=True)
    pairs = []
    target_shape = (channels, size, size)

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
        recon_im, kterms = reconstruct_function(next_im)
        # EXPERIMENT recon_im, kterms, z, u = reconstruct_function(next_im)
        # print(kterms.shape, np.amin(kterms), np.amax(kterms))
        # print("Z", z.shape, z[0], z[1], np.amin(z), np.amax(z))
        # print("U", u.shape, u[0], u[1], np.amin(u), np.amax(u))
        pairs.append([next_im.reshape(target_shape), 
                      recon_im.reshape(target_shape)])

    # for i in range(num):
    #     batch = next(iterator)
    #     next_im = batch['features']
    #     recon_im, kterms = reconstruct_function(next_im)
    #     pairs.append([next_im.reshape(target_shape), 
    #                   recon_im.reshape(target_shape)])
    return pairs

def build_random_samples(model, num_rand, size, channels):
    # reset the random generator
    draw = model.get_top_bricks()[0]
    del draw._theano_rng
    del draw._theano_seed
    draw.seed_rng = np.random.RandomState(config.default_seed)

    #------------------------------------------------------------
    logging.info("Compiling sample function...")

    n_samples = T.iscalar("n_samples")
    samples = draw.sample(n_samples)

    do_sample = theano.function([n_samples], outputs=samples, allow_input_downcast=True)

    #------------------------------------------------------------
    logging.info("Sampling images...")
    samples = do_sample(num_rand)
    current_shape = samples.shape
    niter = current_shape[0]
    target_shape = (niter, num_rand, channels, size, size)
    return samples.reshape(target_shape)

def get_image_diff(im1, im2):
    diff = np.subtract(im1, im2)
    ab = np.absolute(diff)
    return np.sum(ab)

def gen_match_pairs(data_stream, size, channels, targets):
    target_shape = (channels, size, size)
    matches = []

    # first build a list of all images in datastream
    all_datastream_images = []
    iterator = data_stream.get_epoch_iterator(as_dict=True)
    for batch in iterator:
        for entry in batch['features']:
            all_datastream_images.append(entry)

    for im1 in targets:
        best_score = 1e100
        best_score2 = 1e100
        best_im = None
        best_im2 = None
        for tr_im in all_datastream_images:
            im2 =  tr_im.reshape(target_shape)
            score = get_image_diff(im1, im2)
            if(score < best_score):
                best_score2 = best_score
                best_score = score
                best_im2 = best_im
                best_im = im2
            elif(score < best_score2):
                best_score2 = score
                best_im2 = im2
        # print("Format from {} to {}", tr_im.shape, im2.shape)
        print("Neighbor processed, score {}".format(best_score))
        matches.append([best_im2, best_im])

    return matches

def attach_recon_to_neighbors(model, images_left, images_right, channels, size):
    logging.info("Compiling reconstruction function...")
    draw = model.get_top_bricks()[0]
    x = tensor.matrix('features')
    x_recons, kl_terms = draw.reconstruct(x)
    reconstruct_function = theano.function([x], draw.reconstruct(x))    

    input_shape = (1, channels * size * size)
    target_shape = (channels, size, size)

    for n in images_left:
        next_im = n[1].reshape(input_shape)
        recon_im, kterms = reconstruct_function(next_im)
        n.append(recon_im.reshape(target_shape))        

    for n in images_right:
        next_im = n[1].reshape(input_shape)
        recon_im, kterms = reconstruct_function(next_im)
        n.append(recon_im.reshape(target_shape))        


def build_neighbors(model, data_stream, size, channels, images_left, images_right):
    pairs_left = gen_match_pairs(data_stream, size, channels, images_left)
    pairs_right = gen_match_pairs(data_stream, size, channels, images_right)
    attach_recon_to_neighbors(model, pairs_left, pairs_right, channels, size)
    return [pairs_left, pairs_right]

def generate_dash(model, subdir, size, channels, rows, cols, train_stream, test_stream):
    sample_rows = rows - 4
    sample_cols = cols - 6
    num_rand = sample_rows * sample_cols

    logging.info("Generating random samples")    
    samples = build_random_samples(model, num_rand, size, channels)
    left_samp = samples[-1][:sample_rows]
    right_samp = samples[-1][-sample_rows:]
    logging.info("Generating neighbors")
    left_pairs, right_pairs = build_neighbors(model, train_stream, size, channels, left_samp, right_samp)
    logging.info("Generating pairs (training)")    
    train_pairs = build_reconstruct_pairs(train_stream, cols, model, channels, size)
    logging.info("Generating pairs (test)")    
    test_pairs = build_reconstruct_pairs(test_stream, cols, model, channels, size)
    logging.info("Rendering grid")
    imgrid = render_grid(rows, cols, size, size, channels, train_pairs, test_pairs, samples, left_pairs, right_pairs)
    imgrid.save("{0}/dash.png".format(subdir))

def do_sample(subdir, args):
    rows = args.rows
    cols = args.cols

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
    image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset, args.channels, args.size)
    train_stream = Flatten(DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, 1)))
    test_stream  = Flatten(DataStream.default_stream(data_test,  iteration_scheme=SequentialScheme(data_test.num_examples, 1)))
    size = image_size[0]

    generate_dash(model, subdir, size, channels, rows, cols, train_stream, test_stream)
    # generate_samples(p, rows, cols, train_pairs, subdir, size, channels)

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
    parser.add_argument("--cols", type=int,
                default=12, help="grid cols")
    parser.add_argument("--rows", type=int,
                default=8, help="grid rows")

    args = parser.parse_args()

    subdir = "dashboard"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    do_sample(subdir, args)
