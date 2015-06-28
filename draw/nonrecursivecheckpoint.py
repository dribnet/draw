from __future__ import division, print_function

import os
import shutil
import theano
import tempfile
import zipfile
from six.moves import cPickle
from contextlib import closing
from theano.misc import pkl_utils

from blocks.extensions.saveload import Checkpoint

from sample import generate_samples

from blocks.serialization import PersistentParameterID, PicklerWithWarning, secure_dump

from pickle import HIGHEST_PROTOCOL
try:
    from pickle import DEFAULT_PROTOCOL
except ImportError:
    DEFAULT_PROTOCOL = HIGHEST_PROTOCOL

from nonrecursivepickler import NonrecursivePickler

SAVED_TO = "saved_to"

def nr_dump(obj, file_handler, protocol=DEFAULT_PROTOCOL,
         persistent_id=PersistentParameterID, use_cpickle=False):
    with closing(zipfile.ZipFile(file_handler, 'w', zipfile.ZIP_DEFLATED,
                                 allowZip64=True)) as zip_file:
        def func(f):
            # if use_cpickle:
            #     p = cPickle.Pickler(f, protocol=protocol)
            # else:
            #     p = PicklerWithWarning(f, protocol=protocol)
            p = NonrecursivePickler(f, protocol=protocol)
            p.persistent_id = persistent_id(zip_file)
            p.dump(obj)
        pkl_utils.zipadd(func, zip_file, 'pkl')


def nr_secure_dump(object_, path, dump_function=nr_dump, **kwargs):
    r"""Robust serialization - does not corrupt your files when failed.

    Parameters
    ----------
    object_ : object
        The object to be saved to the disk.
    path : str
        The destination path.
    dump_function : function
        The function that is used to perform the serialization. Must take
        an object and file object as arguments. By default, :func:`dump` is
        used. An alternative would be :func:`pickle.dump`.
    \*\*kwargs
        Keyword arguments to be passed to `dump_function`.

    """
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            dump_function(object_, temp, **kwargs)
        shutil.move(temp.name, path)
    except:
        if "temp" in locals():
            os.remove(temp.name)
        raise

class NonRecursiveCheckpoint(Checkpoint):
    def do(self, callback_name, *args):
        """Pickle the main loop object to the disk.

        If `*args` contain an argument from user, it is treated as
        saving path to be used instead of the one given at the
        construction stage.

        """
        _, from_user = self.parse_args(callback_name, args)
        try:
            path = self.path
            if from_user:
                path, = from_user
            nr_secure_dump(self.main_loop, path, use_cpickle=self.use_cpickle)
            filenames = self.save_separately_filenames(path)
            for attribute in self.save_separately:
                secure_dump(getattr(self.main_loop, attribute),
                            filenames[attribute], cPickle.dump)
        except Exception:
            path = None
            raise
        finally:
            already_saved_to = self.main_loop.log.current_row.get(SAVED_TO, ())
            self.main_loop.log.current_row[SAVED_TO] = (already_saved_to +
                                                        (path,))
