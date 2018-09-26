import numpy as np
from keras import backend as K

from util.pose_utils import keys_to_stack, stack_by_keys, pose_index


class BaseGenerator(object):
    """ A generator that dynamically creates batches from a list of N examples. """

    def __init__(self, N, batch_size, create_y):
        """
        Parameters:
            N: total number of elements
            batch_size:  batch size
            create_y: True if generating target batches is needed
        """
        self.N = N
        self.batch_size = batch_size
        self.batch_start = 0
        self.create_y = create_y

    def __iter__(self):
        return self

    def next(self):
        self._advance_batch()
        if self.batch_start == 0:
            self.first_batch()

        x_batch = self.get_x_batch()
        if self.create_y:
            y_batch = self.get_y_batch()
            result = (x_batch, y_batch)
        else:
            result = x_batch

        self.batch_start = self.batch_end % self.N
        return result

    def _advance_batch(self):
        assert self.batch_start < self.N
        if self.batch_start + self.batch_size <= self.N:
            self.batch_end = self.batch_start + self.batch_size
        else:
            self.batch_end = self.N

    def first_batch(self):
        """ Called on the first batch when the iterator finished loading all examples. Useful for reshuffling data. """
        pass

    def get_x_batch(self):
        pass

    def get_y_batch(self):
        pass


class SplitBatchGenerator(BaseGenerator):
    """
    Generates batches. The input batch has two poses and a dummy placeholder for the rotation matrix.
    """

    def __init__(self, x, y, params, shuffle=False):
        self.x = x
        self.y = y
        self.params = params
        self.shuffle = shuffle

        # Dummy arrays
        self.empty_y = np.zeros((params.batch_size, 1), dtype=K.floatx())
        self.empty_rot = np.tile(np.eye(3), (params.batch_size, 1, 1))

        self.framelist, self.keys = keys_to_stack(x)

        super(SplitBatchGenerator, self).__init__(len(self.framelist) / 2, params.batch_size, y is not None)

    def first_batch(self):
        n = len(self.framelist)
        self.index_array = np.random.permutation(n) if self.shuffle else np.arange(n)

    def get_batch(self, data, width):
        current_batch_size = self.batch_end - self.batch_start
        batch_frames = self.framelist[self.index_array[2 * self.batch_start: 2 * self.batch_end]]

        # Generate batch of x values
        # Note that Keras expects list of numpy arrays for multiple inputs
        # and doesn't like a single numpy array with length of 2
        batch = [np.zeros((current_batch_size, width), dtype=K.floatx()),
                 np.zeros((current_batch_size, width), dtype=K.floatx())]

        for i, (fnum, vid_id) in enumerate(batch_frames):
            batch[i % 2][i / 2] = data[self.keys[vid_id]][fnum].flatten()

        return batch

    def get_x_batch(self):
        x_batch = self.get_batch(self.x, 32)
        x_batch.append(self.empty_rot[:(self.batch_end - self.batch_start), :, :])

        return x_batch

    def get_y_batch(self):
        y_batch = self.get_batch(self.y, 48)
        y_batch.append(self.empty_y[:(self.batch_end - self.batch_start), :])

        return y_batch


class PosNegBatchGenerator(BaseGenerator):
    """
    Generates batches. A batch have two images, they can be either the same pose from different camera angles
    or entirely different poses. There is also a third element in the input, a rotation matrix between the two
    cameras. The output batch is the two poses in 3D, and the distance of the embedding.
    """

    def __init__(self, x, y, params, shuffle=False, cams=None, train_dict_3d_absolute=None):
        assert 0 <= params.pos_pairs_ratio <= 1, "pos_pairs_ratio must be between 0 and 1"
        assert y is not None, "The generator is onnly usable for training"
        assert params.siamese_loss in ['pose_dist_loss', 'contrastive_loss', 'mean_squared_error']

        self.x = x
        self.y = y
        self.shuffle = shuffle
        self.params = params

        self.poselist, self.keys = pose_index(x)

        self.absolute_pose = stack_by_keys(train_dict_3d_absolute, self.poselist, self.keys)
        assert self.absolute_pose.shape[1:] == (16, 3)

        self.cam_cnt = len(params.train_camera_names)
        self.create_matrix_dict(cams)

        super(PosNegBatchGenerator, self).__init__(len(self.poselist), params.batch_size, y is not None)

    def create_matrix_dict(self, cams):
        self.rotation_mx = {}
        for k, v in cams.iteritems():
            name = v[-1]
            assert isinstance(name, basestring)
            self.rotation_mx[(k[0], name)] = v[0]

    @staticmethod
    def pose_diff(p1, p2):
        assert p1.shape == p2.shape
        assert p1.shape[1:] == (16, 3)

        errors = p1 - p2
        errors = np.mean(np.linalg.norm(errors, ord=2, axis=2), axis=1)
        return errors

    def first_batch(self):
        # Random permutation of poses
        self.pose1_indices = np.random.permutation(self.N) if self.shuffle else np.arange(self.N)
        self.pose2_indices = np.random.permutation(self.N)

        # Select cameras randomly
        self.c1 = np.random.randint(0, self.cam_cnt, size=(self.N,))
        self.c2 = np.mod(self.c1 + np.random.randint(1, self.cam_cnt, size=(self.N,)), self.cam_cnt)

        # Make random pairs matching
        matching = np.random.random(self.c1.shape) < self.params.pos_pairs_ratio
        self.pose2_indices[matching] = self.pose1_indices[matching]

        # Calculate difference between poses
        errors = PosNegBatchGenerator.pose_diff(self.absolute_pose[self.pose1_indices],
                                                self.absolute_pose[self.pose2_indices])

        self.pose_dist = errors * self.params.pose_dist_weight
        assert np.all((~matching) | (self.pose_dist == 0)), 'somewhere matching but not zero error'

    def get_batch(self, data, width):
        current_batch_size = self.batch_end - self.batch_start
        batch_pose1 = self.poselist[self.pose1_indices[self.batch_start: self.batch_end]]
        batch_pose2 = self.poselist[self.pose2_indices[self.batch_start: self.batch_end]]
        batch_c1 = self.c1[self.batch_start: self.batch_end]
        batch_c2 = self.c2[self.batch_start: self.batch_end]
        assert len(batch_pose1) == current_batch_size
        assert len(batch_pose2) == current_batch_size
        assert len(batch_c1) == current_batch_size
        assert len(batch_c2) == current_batch_size

        # Generate batch of x values
        # Note that keras expects list of numpy arrays for multiple inputs
        # and doesn't like a single numpy array with length of 2
        batch = [np.zeros((current_batch_size, width), dtype='float32'),
                 np.zeros((current_batch_size, width), dtype='float32')]

        for i in range(current_batch_size):
            batch[0][i] = data[self.keys[batch_pose1[i, 1]] +
                               (self.params.train_camera_names[batch_c1[i]],)][batch_pose1[i, 0]].flatten()
            batch[1][i] = data[self.keys[batch_pose2[i, 1]] +
                               (self.params.train_camera_names[batch_c2[i]],)][batch_pose2[i, 0]].flatten()
        return batch

    def rotation_matrix_batch(self):
        current_batch_size = self.batch_end - self.batch_start
        batch_pose1 = self.poselist[self.pose1_indices[self.batch_start: self.batch_end]]
        batch_pose2 = self.poselist[self.pose2_indices[self.batch_start: self.batch_end]]
        batch_c1 = self.c1[self.batch_start: self.batch_end]
        batch_c2 = self.c2[self.batch_start: self.batch_end]

        rs = np.empty((current_batch_size, 3, 3), dtype=K.floatx())
        for i in range(current_batch_size):
            k1 = (self.keys[batch_pose1[i, 1]][0], self.params.train_camera_names[batch_c1[i]])
            k2 = (self.keys[batch_pose2[i, 1]][0], self.params.train_camera_names[batch_c2[i]])
            rs[i] = np.dot(self.rotation_mx[k2], self.rotation_mx[k1].T)

        return rs

    def get_y_batch(self):
        y_batch = self.get_batch(self.y, 48)

        batch_pose_dist = self.pose_dist[self.batch_start:self.batch_end]
        assert len(y_batch[0]) == len(batch_pose_dist)
        y_batch.append(batch_pose_dist)

        return y_batch

    def get_x_batch(self):
        x_batch = self.get_batch(self.x, 32)
        x_batch.append(self.rotation_matrix_batch())
        return x_batch
