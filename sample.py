#!/usr/bin/env python 

from __future__ import print_function, division

import logging
import theano
import theano.tensor as T
import cPickle as pickle

import numpy as np
import os

from PIL import Image

from draw.labcolor import scaled_lab2rgb
import modelutil

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


def generate_samples(draw, subdir, filename, width, height, channels, lab, flat, interleaves, shuffles, rows, cols, z_dim, with_space):
    #------------------------------------------------------------
    logging.info("Compiling sample function...")
    if flat:
        coords = modelutil.make_flat(z_dim, cols, rows, True, interleaves, shuffles)
        print("SUCCESSFUL COORDS IS: {}".format(coords.shape))
        samples = modelutil.sample_at(draw, coords)
        print("SUCCESSFUL SHAPE IS: {}".format(samples.shape))
    else:
        samples = modelutil.sample_random(draw, rows*cols)
        print("SUCCESSFUL SHAPE IS: {}".format(samples.shape))

    n_iter, N, D = samples.shape
    # logging.info("SHAPE IS: {}".format(samples.shape))
    samples = samples.reshape( (n_iter, N, channels, height, width) )

    if(n_iter > 0):
        img = img_grid(samples[n_iter-1,:,:,:], rows, cols, lab, with_space)
        img.save("{0}/{1}.png".format(subdir, filename))

    if(n_iter > 1):
        for i in xrange(n_iter-1):
            img = img_grid(samples[i,:,:,:], rows, cols, lab, with_space)
            img.save("{0}/time-{1:03d}.png".format(subdir, i))

        os.system("convert -delay 5 {0}/time-*.png -delay 300 {0}/{1}.png {0}/sequence.gif".format(subdir, filename))

if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model_file", help="filename of a pickled DRAW model")
    parser.add_argument("--channels", type=int,
                default=1, help="number of channels")
    parser.add_argument("--size", type=int,
                default=28, help="Output image size (width and height)")
    parser.add_argument("--width", type=int,
                default=None, help="image width (if custom dataset)")
    parser.add_argument("--height", type=int,
                default=None, help="image height (if custom dataset)")
    parser.add_argument("--cols", type=int,
                default=12, help="grid cols")
    parser.add_argument("--rows", type=int,
                default=8, help="grid rows")
    parser.add_argument('--flat', dest='flat', default=False, action='store_true')
    parser.add_argument("--interleaves", type=int,
                default=0, help="#interleaves if flat")
    parser.add_argument("--shuffles", type=int,
                default=0, help="#shuffles if flat")
    parser.add_argument("--z_dim", type=int,
                default=100, help="z_dim (if flat)")
    parser.add_argument('--tight', dest='tight', default=False, action='store_true')
    parser.add_argument('--lab', dest='lab', default=False,
                help="Lab Colorspace", action='store_true')
    parser.add_argument('--subdir', dest='subdir', default="sample")
    parser.add_argument('--filename', dest='filename', default="sample")
    args = parser.parse_args()

    logging.info("Loading file %s..." % args.model_file)
    main_model = modelutil.load_file(args.model_file)

    if not os.path.exists(args.subdir):
        os.makedirs(args.subdir)

    generate_samples(main_model, args.subdir, args.filename, args.width, args.height, args.channels, args.lab, args.flat, args.interleaves, args.shuffles,args.rows, args.cols, args.z_dim, not args.tight)
