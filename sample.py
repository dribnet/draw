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
from blocks.model import AbstractModel
from blocks.config import config

from draw.labcolor import scaled_lab2rgb

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)

# this is old and now unused
def scale_norm(arr):
    arr = arr - arr.min()
    scale = (arr.max() - arr.min())
    return arr / scale

def img_grid(arr, rows, cols, lab, with_space, global_scale=False):
    N, channels, height, width = arr.shape

    total_height = rows * height
    total_width  = cols * width

    if with_space:
        total_height = total_height + (rows - 1)
        total_width  = total_width + (cols - 1)

    if global_scale:
        arr = scale_norm(arr)

    I = np.zeros((channels, total_height, total_width))
    I.fill(1)

    for i in xrange(N):
        r = i // cols
        c = i % cols

        this = arr[i]
        # if global_scale:
        #     this = arr[i]
        # else:
        #     this = scale_norm(arr[i])

        if with_space:
            offset_y, offset_x = r*height+r, c*width+c
        else:
            offset_y, offset_x = r*height, c*width
        I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = this
    
    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I)

    if(lab):
        out = scaled_lab2rgb(out)

    out = (255 * out).astype(np.uint8)

    return Image.fromarray(out)

def pol2cart(phi):
    x = np.cos(phi)
    y = np.sin(phi)
    return(x, y)

from scipy.special import ndtri

# sqrt2 = 1.41421356237
sqrt2 = 1.0
def lerpTo(val, low, high):
    zeroToOne = np.clip((val + sqrt2) / (2 * sqrt2), 0, 1)
    return low + (high - low) * zeroToOne

def generate_samples(p, subdir, output_size, channels, lab, flat, rows, cols, dims, with_space):
    if isinstance(p, AbstractModel):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return

    draw = model.get_top_bricks()[0]
    # reset the random generator
    try:
        del draw._theano_rng
        del draw._theano_seed
    except AttributeError:
        # Do nothing
        pass
    draw.seed_rng = np.random.RandomState(config.default_seed)

    #------------------------------------------------------------
    logging.info("Compiling sample function...")
    n_samples = T.iscalar("n_samples")
    if flat:
        offsets = []
        for i in range(dims):
            offsets.append(pol2cart(i * np.pi / dims))

        #------------------------------------------------------------
        # this does 3.3 deviations (99.9)
        range_high = 0.999
        range_low = 1 - range_high
        ul = []
        for c in range(cols):
            yf = (c - (cols / 2.0) + 0.5) / (cols / 2.0 + 0.5)
            for r in range(rows):
                xf = (r - (rows / 2.0) + 0.5) / (rows / 2.0 + 0.5)
                coords = map(lambda o: np.dot([xf, yf], o), offsets)
                ranged = map(lambda n:lerpTo(n, range_low, range_high), coords)
                flatter = map(lambda n:lerpTo(n, -3, 3), coords)
                cdfed = map(ndtri, ranged)
                # u1 = (np.array(flatter) + np.array(cdfed)) / 2.0
                u1 = np.array(cdfed)
                ul.append(u1)
        u = np.array(ul)
        u_var = T.matrix("u_var")
        samples_at = draw.sample_at(n_samples, u_var)
        do_sample_at = theano.function([n_samples, u_var], outputs=samples_at, allow_input_downcast=True)
        #------------------------------------------------------------
        logging.info("Sampling and saving images...")
        samples, newu = do_sample_at(rows*cols, u)
        # print("NEWU: ", newu)
        # print("NEWU.s: ", newu.shape)
        # print("NEWU[0]: ", newu[0])
    else:
        samples = draw.sample(n_samples)
        do_sample = theano.function([n_samples], outputs=samples, allow_input_downcast=True)
        #------------------------------------------------------------
        logging.info("Sampling and saving images...")
        samples = do_sample(rows*cols)

    n_iter, N, D = samples.shape
    # logging.info("SHAPE IS: {}".format(samples.shape))
    samples = samples.reshape( (n_iter, N, channels, output_size, output_size) )

    if(n_iter > 0):
        img = img_grid(samples[n_iter-1,:,:,:], rows, cols, lab, with_space)
        img.save("{0}/sample.png".format(subdir))

    if(n_iter > 1):
        for i in xrange(n_iter-1):
            img = img_grid(samples[i,:,:,:], rows, cols, lab, with_space)
            img.save("{0}/time-{1:03d}.png".format(subdir, i))

        os.system("convert -delay 5 {0}/time-*.png -delay 300 {0}/sample.png {0}/sequence.gif".format(subdir))

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model_file", help="filename of a pickled DRAW model")
    parser.add_argument("--channels", type=int,
                default=1, help="number of channels")
    parser.add_argument("--size", type=int,
                default=28, help="Output image size (width and height)")
    parser.add_argument("--cols", type=int,
                default=12, help="grid cols")
    parser.add_argument("--rows", type=int,
                default=8, help="grid rows")
    parser.add_argument('--flat', dest='flat', default=False, action='store_true')
    parser.add_argument("--zdim", type=int,
                default=2, help="zdim (if flat)")
    parser.add_argument('--tight', dest='tight', default=False, action='store_true')
    parser.add_argument('--lab', dest='lab', default=False,
                help="Lab Colorspace", action='store_true')
    args = parser.parse_args()

    logging.info("Loading file %s..." % args.model_file)
    with open(args.model_file, "rb") as f:
        p = pickle.load(f)

    subdir = "sample"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    generate_samples(p, subdir, args.size, args.channels, args.lab, args.flat, args.rows, args.cols, args.zdim, not args.tight)
