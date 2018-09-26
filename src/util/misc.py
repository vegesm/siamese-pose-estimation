import pickle
import os
import numpy as np


class Params(object):
    """ A simple dictionary that has its keys as attributes available. """

    def __init__(self):
        pass

    def __str__(self):
        s = ""
        for name in sorted(self.__dict__.keys()):
            s += "%-18s %s\n" % (name + ":", self.__dict__[name])
        return s

    def __repr__(self):
        return self.__str__()


def save(path, var):
    """
    Saves the variable ``var`` to the given path. The file format depends on the file extension.
    List of supported file types:

    - .pkl: pickle
    - .npy: numpy
    - .txt: text file, one element per line. ``var`` must be a string or list of strings.
    """
    if path.endswith(".pkl"):
        with open(path, 'wb') as f:
            pickle.dump(var, f, 2)
    elif path.endswith(".npy"):
        np.save(path, var)
    elif path.endswith(".txt"):
        with open(path, 'w') as f:
            if isinstance(var, basestring):
                f.write(var)
            else:
                for i in var:
                    f.write(i)
                    f.write('\n')
    else:
        raise NotImplementedError("Unknown extension: " + os.path.splitext(path)[1])


def load(path):
    """
    Loads the content of a file. It is mainly a convenience function to
    avoid adding the ``open()`` contexts. File type detection is based on extensions.
    Can handle the following types:

    - .pkl: pickles
    - .txt: text files, result is a list of strings ending whitespace removed

    :param path: path to the file
    """
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            return pickle.load(f)
    elif path.endswith('.txt'):
        with open(path, 'r') as f:
            return [x.rstrip('\n\r') for x in list(f)]
    else:
        raise NotImplementedError("Unknown extension: " + os.path.splitext(path)[1])


def ensuredir(path):
    """
    Creates a folder if it doesn't exist.

    :param path: path to the folder to create
    """
    if not os.path.exists(path):
        os.makedirs(path)
