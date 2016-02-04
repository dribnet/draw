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
def make_flat(z_dim, cols, rows, gaussian_prior="True", interleaves=[], shuffles=[]):
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
    range_high = 0.95
    range_low = 1 - range_high
    for c in range(cols):
        # yf = lerp(c / (cols-1.0), -1.0, 1.0)
        yf = (c - (cols / 2.0) + 0.5) / (cols / 2.0 + 0.5)
        for r in range(rows):
            # xf = lerp(r / (rows-1.0), -1.0, 1.0)
            xf = (r - (rows / 2.0) + 0.5) / (rows / 2.0 + 0.5)
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
    rows, cols, z_dim = locations.shape
    flat_locations = locations.reshape(rows*cols, z_dim)

    n_samples = T.iscalar("n_samples")
    u_var = T.matrix("u_var")
    samples_at = model.sample_at(n_samples, u_var)
    do_sample_at = theano.function([n_samples, u_var], outputs=samples_at, allow_input_downcast=True)
    #------------------------------------------------------------
    logging.info("Sampling and saving images...")
    samples, newu = do_sample_at(rows*cols, flat_locations)
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
    # print("Shape: {}".format(u.shape))
    samples = do_sample(u)
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
