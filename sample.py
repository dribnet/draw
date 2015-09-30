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

def img_grid(arr, rows, cols, lab, global_scale=False):
    N, channels, height, width = arr.shape

    total_height = rows * height + (rows - 1)
    total_width  = cols * width + (cols - 1)

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

        offset_y, offset_x = r*height+r, c*width+c
        I[0:channels, offset_y:(offset_y+height), offset_x:(offset_x+width)] = this
    
    if(channels == 1):
        out = I.reshape( (total_height, total_width) )
    else:
        out = np.dstack(I)

    if(lab):
        out = scaled_lab2rgb(out)

    out = (255 * out).astype(np.uint8)

    return Image.fromarray(out)

def generate_samples(p, subdir, output_size, channels, lab, rows, cols, flat):
    if isinstance(p, AbstractModel):
        model = p
    else:
        print("Don't know how to handle unpickled %s" % type(p))
        return

    draw = model.get_top_bricks()[0]
    # reset the random generator
    del draw._theano_rng
    del draw._theano_seed
    draw.seed_rng = np.random.RandomState(config.default_seed)

    #------------------------------------------------------------
    logging.info("Compiling sample function...")
    n_samples = T.iscalar("n_samples")
    if flat:
        #------------------------------------------------------------
        # this does 3 deviations (99.7)
        from scipy.special import ndtri
        range_high = 0.99865
        range_low = 1 - range_high
        rowspace = np.linspace(range_low,range_high,rows)
        colspace = np.linspace(range_low,range_high,cols)
        ul = []
        for c in range(cols):
            for r in range(rows):
                u1 = np.zeros(2)
                u1[0] = ndtri(rowspace[r])
                u1[1] = ndtri(colspace[c])
                # u1 = np.random.uniform(-1, 1, size=4)
                ul.append(u1)
                # ul.append([rowspace[r], colspace[c]])
        u = np.array(ul)
        print(u)
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
        img = img_grid(samples[n_iter-1,:,:,:], rows, cols, lab)
        img.save("{0}/sample.png".format(subdir))

    if(n_iter > 1):
        for i in xrange(n_iter-1):
            img = img_grid(samples[i,:,:,:], rows, cols, lab)
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
    parser.add_argument('--lab', dest='lab', default=False,
                help="Lab Colorspace", action='store_true')
    args = parser.parse_args()

    logging.info("Loading file %s..." % args.model_file)
    with open(args.model_file, "rb") as f:
        p = pickle.load(f)

    subdir = "sample"
    if not os.path.exists(subdir):
        os.makedirs(subdir)

    generate_samples(p, subdir, args.size, args.channels, args.lab, args.rows, args.cols, args.flat)
