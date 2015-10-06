from __future__ import division, print_function

import os
import shutil
import theano
import theano.tensor as T

from blocks.extensions.saveload import Checkpoint

from sample import generate_samples
from dashboard import generate_dash

class SampleCheckpoint(Checkpoint):
    def __init__(self, image_size, channels, lab, flat, z_dim, save_subdir, train_stream, test_stream, **kwargs):
        super(SampleCheckpoint, self).__init__(path=None, **kwargs)
        self.image_size = image_size
        self.channels = channels
        self.lab = lab
        self.flat = flat
        self.z_dim = z_dim
        self.save_subdir = save_subdir
        self.iteration = 0
        self.train_stream = train_stream
        self.test_stream = test_stream
        self.epoch_src = "{0}/sample.png".format(save_subdir)
        self.dash_src = "{0}/dash.png".format(save_subdir)
        self.samples_every = 1
        if train_stream != None and test_stream != None:
            self.set_dash_params(every=10)
        else:
            self.set_dash_params(every=0)
        # fix image size
        self.set_dash_params(every=0)

    def set_dash_params(self, every=10, rows=12, cols=12):
        self.dash_every = every
        self.dash_rows = rows
        self.dash_cols = cols

    def do(self, callback_name, *args):
        """Sample the model and save images to disk
        """
        if self.samples_every != 0 and self.iteration % self.samples_every == 0:
            generate_samples(self.main_loop.model, self.save_subdir, self.image_size[0], self.image_size[1], self.channels, self.lab, self.flat, self.dash_rows, self.dash_cols, self.z_dim, True)
            if os.path.exists(self.epoch_src):
                epoch_dst = "{0}/epoch-{1:04d}.png".format(self.save_subdir, self.iteration)
                shutil.copy2(self.epoch_src, epoch_dst)
                os.system("convert -delay 5 -loop 1 {0}/epoch-*.png {0}/training.gif".format(self.save_subdir))
        if self.dash_every != 0 and self.iteration % self.dash_every == 0:
            generate_dash(self.main_loop.model, self.save_subdir, self.image_size, self.channels, \
                self.lab, self.dash_rows, self.dash_cols, self.train_stream, self.test_stream)
            if os.path.exists(self.epoch_src):
                dash_dst = "{0}/dash-{1:04d}.png".format(self.save_subdir, self.iteration)
                shutil.copy2(self.dash_src, dash_dst)
                os.system("convert -delay 100 -loop 1 {0}/dash-*.png {0}/traindash.gif".format(self.save_subdir))

        self.iteration = self.iteration + 1


