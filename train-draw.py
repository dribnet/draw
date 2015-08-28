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
from blocks.bricks.cost import BinaryCrossEntropy, AbsoluteError, CostMatrix, SquaredError
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
from draw.samplecheckpoint import SampleCheckpoint
from draw.partsonlycheckpoint import PartsOnlyCheckpoint

#----------------------------------------------------------------------------

def main(name, dataset, channels, size, epochs, batch_size, learning_rate,
         attention, n_iter, enc_dim, dec_dim, z_dim, oldmodel, lab):

    image_size, channels, data_train, data_valid, data_test = datasets.get_data(dataset, channels, size)

    train_stream = Flatten(DataStream.default_stream(data_train, iteration_scheme=SequentialScheme(data_train.num_examples, batch_size)))
    valid_stream = Flatten(DataStream.default_stream(data_valid, iteration_scheme=SequentialScheme(data_valid.num_examples, batch_size)))
    test_stream  = Flatten(DataStream.default_stream(data_test,  iteration_scheme=SequentialScheme(data_test.num_examples, batch_size)))

    if name is None:
        name = dataset

    img_height, img_width = image_size
    x_dim = channels * img_height * img_width

    rnninits = {
        #'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }
    inits = {
        #'weights_init': Orthogonal(),
        'weights_init': IsotropicGaussian(0.01),
        'biases_init': Constant(0.),
    }

    # Configure attention mechanism
    if attention != "":
        read_N, write_N = attention.split(',')
    
        read_N = int(read_N)
        write_N = int(write_N)
        read_dim = 2 * channels * read_N ** 2

        reader = AttentionReader(x_dim=x_dim, dec_dim=dec_dim,
                                 channels=channels, width=img_width, height=img_height,
                                 N=read_N, **inits)
        writer = AttentionWriter(input_dim=dec_dim, output_dim=x_dim,
                                 channels=channels, width=img_width, height=img_height,
                                 N=write_N, **inits)
        attention_tag = "r%d-w%d" % (read_N, write_N)
    else:
        read_dim = 2*x_dim

        reader = Reader(x_dim=x_dim, dec_dim=dec_dim, **inits)
        writer = Writer(input_dim=dec_dim, output_dim=x_dim, **inits)

        attention_tag = "full"

    #----------------------------------------------------------------------

    if name is None:
        name = dataset

    # Learning rate
    def lr_tag(value):
        """ Convert a float into a short tag-usable string representation. E.g.:
            0.1   -> 11
            0.01  -> 12
            0.001 -> 13
            0.005 -> 53
        """
        exp = np.floor(np.log10(value))
        leading = ("%e"%value)[0]
        return "%s%d" % (leading, -exp)

    lr_str = lr_tag(learning_rate)

    subdir = name + "-" + time.strftime("%Y%m%d-%H%M%S");
    longname = "%s-%s-t%d-enc%d-dec%d-z%d-lr%s" % (dataset, attention_tag, n_iter, enc_dim, dec_dim, z_dim, lr_str)
    pickle_file = subdir + "/" + longname + ".pkl"

    print("\nRunning experiment %s" % longname)
    print("               dataset: %s" % dataset)
    print("              channels: %d" % channels)
    print("            image_size: %dx%d" % image_size)
    print("         learning rate: %g" % learning_rate)
    print("             attention: %s" % attention)
    print("          n_iterations: %d" % n_iter)
    print("     encoder dimension: %d" % enc_dim)
    print("           z dimension: %d" % z_dim)
    print("     decoder dimension: %d" % dec_dim)
    print("            batch size: %d" % batch_size)
    print("                epochs: %d" % epochs)
    print()

    #----------------------------------------------------------------------

    encoder_rnn = LSTM(dim=enc_dim, name="RNN_enc", **rnninits)
    decoder_rnn = LSTM(dim=dec_dim, name="RNN_dec", **rnninits)
    encoder_mlp = MLP([Identity()], [(read_dim+dec_dim), 4*enc_dim], name="MLP_enc", **inits)
    decoder_mlp = MLP([Identity()], [             z_dim, 4*dec_dim], name="MLP_dec", **inits)
    q_sampler = Qsampler(input_dim=enc_dim, output_dim=z_dim, **inits)

    draw = DrawModel(
                n_iter, 
                reader=reader,
                encoder_mlp=encoder_mlp,
                encoder_rnn=encoder_rnn,
                sampler=q_sampler,
                decoder_mlp=decoder_mlp,
                decoder_rnn=decoder_rnn,
                writer=writer)
    draw.initialize()

    #------------------------------------------------------------------------
    x = tensor.matrix('features')
    
    x_recons, kl_terms = draw.reconstruct(x)

    recons_term = BinaryCrossEntropy().apply(x, x_recons)
    recons_term.name = "recons_term"


    # # THESE MORE COMPLICATED COST FUNCTIONS ARE COMMENTED OUT FOR NOW
    # before_len = x.shape[0]
    # after_len = x_recons.shape[0]

    # # full split
    # before_split = T.reshape(x, (before_len, channels, img_height * img_width))
    # after_split = T.reshape(x_recons, (after_len, channels, img_height * img_width))
    # recons_term_single = BinaryCrossEntropy(name='recons_term_single').apply(before_split[0], after_split[0])

    # if(channels == 1):
    #     recons_term = 1.0 * recons_term_single
    # else:
    #     recons_term_color1 = BinaryCrossEntropy(name='recons_term_color1').apply(before_split[1], after_split[1])
    #     recons_term_color2 = BinaryCrossEntropy(name='recons_term_color2').apply(before_split[2], after_split[2])
    #     recons_term = 1.0 * recons_term_single + 0.1 * recons_term_color1 + 0.1 * recons_term_color2

    # recons_term.name = "recons_term"

    # # begin edge
    # before_unfolded = T.reshape(x, (before_len, channels, img_height, img_width))
    # after_unfolded = T.reshape(x_recons, (after_len, channels, img_height, img_width))

    # edge_matrix = tensor.constant([[0, 0.25, 0], [0.25, -1, 0.25], [0, 0.25, 0]], dtype='float32')
    # th_filter = T.reshape(edge_matrix, (1,1,3,3))

    # before_edge_image = theano.tensor.nnet.conv.conv2d(before_unfolded[:,0:1,:,:], th_filter, border_mode='valid') + 0.5
    # after_edge_image = theano.tensor.nnet.conv.conv2d(after_unfolded[:,0:1,:,:], th_filter, border_mode='valid') + 0.5

    # before_edge_flat = T.reshape(before_edge_image, (before_len, (img_height-2) * (img_width-2)))
    # after_edge_flat = T.reshape(after_edge_image, (after_len, (img_height-2) * (img_width-2)))

    # recons_term_edge = BinaryCrossEntropy(name='recons_term_edge').apply(before_edge_flat, after_edge_flat)
    # # recons_term_edge = SquaredError(name='diff_crossentropy').apply(edge_image1, edge_image2)
    # # recons_term_edge = AbsoluteError(name='diff_crossentropy').apply(edge_image1, edge_image2)
    # recons_term_edge.name = "recons_term_edge"
    # # END COMMENTED OUT COST FUNCTIONS

    kl_terms_sum = kl_terms.sum(axis=0).mean()
    kl_terms_sum.name = "kl_terms_sum"

    # cost = recons_term + 0.25 * recons_term_edge + kl_terms_sum
    cost = recons_term + kl_terms_sum
    cost.name = "nll_bound"

    cost_monitors = [cost, recons_term, kl_terms_sum]

    # cost_monitors = [cost, recons_term, recons_term_single, kl_terms_sum, recons_term_edge]
    # if (channels > 1):
    #     cost_monitors.extend([recons_term_color1, recons_term_color2])

    #------------------------------------------------------------
    cg = ComputationGraph([cost])
    params = VariableFilter(roles=[PARAMETER])(cg.variables)

    algorithm = GradientDescent(
        cost=cost, 
        parameters=params,
        step_rule=CompositeRule([
            StepClipping(10.), 
            Adam(learning_rate),
        ])
        #step_rule=RMSProp(learning_rate),
        #step_rule=Momentum(learning_rate=learning_rate, momentum=0.95)
    )

    #------------------------------------------------------------------------
    # Setup monitors
    monitors = cost_monitors
    for t in range(n_iter):
        kl_term_t = kl_terms[t,:].mean()
        kl_term_t.name = "kl_term_%d" % t

        #x_recons_t = T.nnet.sigmoid(c[t,:,:])
        #recons_term_t = BinaryCrossEntropy().apply(x, x_recons_t)
        #recons_term_t = recons_term_t.mean()
        #recons_term_t.name = "recons_term_%d" % t

        monitors +=[kl_term_t]

    train_monitors = monitors[:]
    train_monitors += [aggregation.mean(algorithm.total_gradient_norm)]
    train_monitors += [aggregation.mean(algorithm.total_step_norm)]
    # Live plotting...
    plot_channels = [
        ["train_nll_bound", "test_nll_bound"],
        ["train_kl_term_%d" % t for t in range(n_iter)],
        #["train_recons_term_%d" % t for t in range(n_iter)],
        ["train_total_gradient_norm", "train_total_step_norm"]
    ]

    #------------------------------------------------------------

    if not os.path.exists(subdir):
        os.makedirs(subdir)

    main_loop = MainLoop(
        model=Model(cost),
        data_stream=train_stream,
        algorithm=algorithm,
        extensions=[
            Timing(),
            FinishAfter(after_n_epochs=epochs),
            TrainingDataMonitoring(
                train_monitors, 
                prefix="train",
                after_epoch=True),
#            DataStreamMonitoring(
#                monitors,
#                valid_stream,
##                updates=scan_updates,
#                prefix="valid"),
            DataStreamMonitoring(
                monitors,
                test_stream,
#                updates=scan_updates, 
                prefix="test"),
            PartsOnlyCheckpoint("{}/{}".format(subdir,name), before_training=True, after_epoch=True, save_separately=['log', 'model']),
            SampleCheckpoint(image_size=image_size[0], channels=channels, lab=lab, save_subdir=subdir, \
                before_training=True, after_epoch=True, train_stream=train_stream, test_stream=test_stream),
            # Plot(name, channels=plot_channels),
            ProgressBar(),
            Printing()])

    if oldmodel is not None:
        print("Initializing parameters with old model %s"%oldmodel)
        with open(oldmodel, "rb") as f:
            oldmodel = pickle.load(f)
            main_loop.model.set_parameter_values(oldmodel.get_parameter_values())
        del oldmodel

    main_loop.run()

#-----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, dest="name",
                default=None, help="Name for this experiment")
    parser.add_argument("--dataset", type=str, dest="dataset",
                default="bmnist", help="Dataset to use: [bmnist|mnist|cifar10]")
    parser.add_argument("--channels", type=int,
                default=None, help="number of channels (if custom dataset)")
    parser.add_argument("--size", type=int,
                default=None, help="image size (if custom dataset)")
    parser.add_argument("--epochs", type=int, dest="epochs",
                default=100, help="Number of training epochs to do")
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--attention", "-a", type=str, default="",
                help="Use attention mechanism (read_window,write_window)")
    parser.add_argument("--niter", type=int, dest="n_iter",
                default=10, help="No. of iterations")
    parser.add_argument("--enc-dim", type=int, dest="enc_dim",
                default=256, help="Encoder RNN state dimension")
    parser.add_argument("--dec-dim", type=int, dest="dec_dim",
                default=256, help="Decoder  RNN state dimension")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                default=100, help="Z-vector dimension")
    parser.add_argument("--oldmodel", type=str,
                help="Use a model pkl file created by a previous run as a starting point for all parameters")
    parser.add_argument('--lab', dest='lab', default=False,
                help="Lab Colorspace", action='store_true')
    args = parser.parse_args()

    main(**vars(args))
