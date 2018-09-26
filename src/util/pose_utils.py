import numpy as np


def get_pose_count(data_dict):
    """
    Calculates the number of poses in the dataset. A pose is one frame of a sequence, independent of cameras.
    In other words, each pose was taken from 4 cameras.
    """
    pose_lens = {}
    for key in data_dict.keys():
        subject, action, seq, camera = key
        if (subject, action, seq) in pose_lens:
            assert pose_lens[(subject, action, seq)] == len(data_dict[key]), \
                'Inconsistent number of poses at ' + str(key)
        else:
            pose_lens[(subject, action, seq)] = len(data_dict[key])

    return sum([x for x in pose_lens.values()])


def pose_index(data_dict):
    """
    Calculates a pose index for the given input dict.

    :returns: (poselist, keys) tuple, where poselist has a row for each pose and two columns.
        The first column is the frame number, the second an index into keys. keys is a list of tuples of 3 elements:
        (subject, action, sequence). Note that camera information is not present.
    """
    pose_lens = {}
    for key in data_dict.keys():
        subject, action, seq, camera = key
        if (subject, action, seq) in pose_lens:
            assert pose_lens[(subject, action, seq)] == len(data_dict[key])
        else:
            pose_lens[(subject, action, seq)] = len(data_dict[key])

    N = sum([x for x in pose_lens.values()])
    poselist = np.zeros((N, 2), 'int64')
    keys = sorted(pose_lens.keys())

    start = 0
    for i, k in enumerate(keys):
        n = pose_lens[k]
        poselist[start:start + n, 0] = np.arange(n)
        poselist[start:start + n, 1] = i

        start += n

    assert start == len(poselist)
    for i in range(len(keys) - 1):
        assert keys[i] < keys[i + 1]

    for i in range(len(poselist) - 1):
        assert poselist[i, 1] <= poselist[i + 1, 1]

    return poselist, keys


def keys_to_stack(x):
    """
    Creates an index that can be used to merge the data that is in baseline format.
    The output is deterministic.

    :returns: ``(framelist, keys)`` tuple, ``keys`` is an array of video sequence keys, ``framelist`` is a numpy array,
              each row contains a frame number and a key id which can be looked up in keys.
    """
    keys = sorted(x.keys())
    key2id = {k: i for i, k in enumerate(keys)}

    x_len = sum([v.shape[0] for v in x.values()])
    framelist = np.empty((x_len, 2), dtype='int32')

    start = 0
    for k in keys:
        cnt = x[k].shape[0]
        framelist[start:start + cnt, 1] = key2id[k]
        framelist[start:start + cnt, 0] = np.arange(cnt)
        start += cnt

    assert start == len(framelist)
    for i in range(len(keys) - 1):
        assert keys[i] < keys[i + 1]

    for i in range(len(framelist) - 1):
        assert framelist[i, 1] <= framelist[i + 1, 1]

    return framelist, keys


# stack y data according to framelist
def stack_by_keys(data_dict, framelist, keys):
    """ Stacks a pose dictionary according to ``framelist``. ``keys`` provides the keys to the video name. """
    shape = list(data_dict[keys[0]].shape)
    shape[0] = len(framelist)
    out = np.empty(shape)

    for i, key in enumerate(keys):
        out[framelist[:, 1] == i, :] = data_dict[key]

    return out
