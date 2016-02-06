#!/usr/bin/env python 

from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
import cPickle as pickle

import numpy as np
import os

from PIL import Image
from blocks.main_loop import MainLoop
from blocks.model import Model
from blocks.config import config

from draw.labcolor import scaled_lab2rgb
from scipy.special import ndtri

### Load a model from disk
def load_file(filename):
    with open(filename, "rb") as f:
        p = pickle.load(f)
    if isinstance(p, Model):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return

    main_model = model.get_top_bricks()[0]
    # reset the random generator
    try:
        del main_model._theano_rng
        del main_model._theano_seed
    except AttributeError:
        # Do nothing
        pass
    main_model.seed_rng = np.random.RandomState(config.default_seed)
    return main_model

### 
def make_flat(z_dim, cols, rows, gaussian_prior=True, interleaves=[], shuffles=[]):
    sqrt2 = 1.0
    def lerpTo(val, low, high):
        zeroToOne = np.clip((val + sqrt2) / (2 * sqrt2), 0, 1)
        return low + (high - low) * zeroToOne

    def lerp(val, low, high):
        return low + (high - low) * val

    def pol2cart(phi):
        x = np.cos(phi)
        y = np.sin(phi)
        return(x, y)

    #  http://stackoverflow.com/a/5347492
    # >>> interleave(np.array(range(6)))
    # array([0, 3, 1, 4, 2, 5])
    def interleave(offsets):
        shape = offsets.shape
        split_point = int(shape[0] / 2)
        a = np.array(offsets[:split_point])
        b = np.array(offsets[split_point:])
        c = np.empty(shape, dtype=a.dtype)
        c[0::2] = a
        c[1::2] = b
        return c

    def shuffle(offsets):
        np.random.shuffle(offsets)

    offsets = []
    for i in range(z_dim):
        offsets.append(pol2cart(i * np.pi / z_dim))
    offsets = np.array(offsets)

    for i in range(interleaves):
        offsets = interleave(offsets)

    for i in range(shuffles):
        shuffle(offsets)

    ul = []
    # range_high = 0.95
    # range_low = 1 - range_high
    range_high = 0.997  # 3 standard deviations
    range_low = 1 - range_high
    for r in range(rows):
        # xf = lerp(r / (rows-1.0), -1.0, 1.0)
        xf = (r - (rows / 2.0) + 0.5) / ((rows-1) / 2.0 + 0.5)
        for c in range(cols):
            # yf = lerp(c / (cols-1.0), -1.0, 1.0)
            yf = (c - (cols / 2.0) + 0.5) / ((cols-1) / 2.0 + 0.5)
            coords = map(lambda o: np.dot([xf, yf], o), offsets)
            ranged = map(lambda n:lerpTo(n, range_low, range_high), coords)
            # ranged = map(lambda n:lerpTo(n, range_low, range_high), [xf, yf])
            if(gaussian_prior):
                cdfed = map(ndtri, ranged)
            else:
                cdfed = ranged
            ul.append(cdfed)
    u = np.array(ul).reshape(rows,cols,z_dim).astype('float32')
    return u

def sample_at(model, locations):
    u_var = T.tensor3("u_var")
    sample = model.sample_given(u_var)
    do_sample = theano.function([u_var], outputs=sample, allow_input_downcast=True)
    #------------------------------------------------------------
    iters, dim = model.get_iters_and_dim()
    rows, cols, z_dim = locations.shape
    logging.info("Sampling {}x{} locations with {} iters and {}={} dim...".format(rows,cols,iters,dim,z_dim))
    numsamples = rows * cols
    u_list = np.zeros((iters, numsamples, dim))

    for y in range(rows):
        for x in range(cols):
            # xcur_ycur = np.random.normal(0, 1.0, (iters, 1, dim))
            xcur_ycur = np.zeros((iters, 1, dim))
            xcur_ycur[0,0,:] = locations[y][x].reshape(dim)
            n = y * cols + x
            # curu = rowmin
            u_list[:,n:n+1,:] = xcur_ycur

    samples = do_sample(u_list)
    print("Shape: {}".format(samples.shape))
    return samples

def sample_random_native(model, numsamples):
    n_samples = T.iscalar("n_samples")
    sample = model.sample(n_samples)
    do_sample = theano.function([n_samples], outputs=sample, allow_input_downcast=True)
    #------------------------------------------------------------
    logging.info("Sampling and saving images...")
    samples = do_sample(numsamples)
    return samples

def sample_random(model, numsamples):
    u_var = T.tensor3("u_var")
    sample = model.sample_given(u_var)
    do_sample = theano.function([u_var], outputs=sample, allow_input_downcast=True)
    #------------------------------------------------------------
    logging.info("Sampling images...")
    iters, dim = model.get_iters_and_dim()
    u = np.random.normal(0, 1, (iters, numsamples, dim))
    samples = do_sample(u)
    print("Shape: {}".format(samples.shape))
    return samples

def sample_gradient(model, rows, cols):
    u_var = T.tensor3("u_var")
    sample = model.sample_given(u_var)
    do_sample = theano.function([u_var], outputs=sample, allow_input_downcast=True)
    #------------------------------------------------------------
    logging.info("Sampling gradient...")
    iters, dim = model.get_iters_and_dim()

    numsamples = rows * cols
    u_list = np.zeros((iters, numsamples, dim))
    xmin_ymin = np.random.normal(0, 1, (iters, 1, dim))
    xmax_ymin = np.random.normal(0, 1, (iters, 1, dim))
    xmin_ymax = np.random.normal(0, 1, (iters, 1, dim))
    # A xmax_ymax = np.random.normal(0, 1, (iters, 1, dim))
    # B xmax_ymax = xmin_ymax + (xmax_ymin - xmin_ymin)
    xmax_ymax = xmin_ymax + (xmax_ymin - xmin_ymin)
    # C xmax_ymax = xmin_ymin
    # C xmin_ymax = xmax_ymin

    for y in range(rows):
        # xcur_ymin = ((1.0 * y * xmin_ymin) + ((rows - y - 1.0) * xmax_ymin)) / (rows - 1.0)
        # xcur_ymax = ((1.0 * y * xmin_ymax) + ((rows - y - 1.0) * xmax_ymax)) / (rows - 1.0)
        xmin_ycur = (((rows - y - 1.0) * xmin_ymin) + (1.0 * y * xmin_ymax)) / (rows - 1.0)
        xmax_ycur = (((rows - y - 1.0) * xmax_ymin) + (1.0 * y * xmax_ymax)) / (rows - 1.0)
        for x in range(cols):
            # xcur_ycur = ((1.0 * x * xcur_ymin) + ((cols - x - 1.0) * xcur_ymax)) / (cols - 1.0)
            xcur_ycur = (((cols - x - 1.0) * xmin_ycur) + (1.0 * x * xmax_ycur)) / (cols - 1.0)
            n = y * cols + x
            # curu = rowmin
            u_list[:,n:n+1,:] = xcur_ycur

    samples = do_sample(u_list)
    print("Shape: {}".format(samples.shape))
    return samples

def sample_at_new(model, locations):
    n_iter, rows, cols, z_dim = locations.shape
    flat_locations = locations.reshape(n_iter, rows*cols, z_dim)

    n_samples = T.iscalar("n_samples")
    u_var = T.matrix("u_var")
    samples_at = model.sample_at_new(n_samples, u_var)
    do_sample_at = theano.function([n_samples, u_var], outputs=samples_at, allow_input_downcast=True)
    #------------------------------------------------------------
    logging.info("Sampling and saving images...")
    samples = do_sample_at(rows*cols, flat_locations)
    return samples

def build_reconstruct_function(model):
    x = T.matrix('features')
    reconstruct_function = theano.function([x], model.reconstruct(x))
    return reconstruct_function

def reconstruct_image(reconstruct_function, source_im):
    recon_im, kterms = reconstruct_function(source_im)
    return recon_im, kterms
