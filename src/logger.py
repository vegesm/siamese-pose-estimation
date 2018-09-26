import numpy as np
import keras
import data_sources.h36m_interface as H36M
from itertools import izip_longest


class LogAllMillimeterError(keras.callbacks.Callback):
    """
    This callback evaluates the model on every epoch, both actionwise and jointwise.
    Results can be saved optionally.
    """

    def __init__(self, std_3d, data, siamese=False, csv=None):
        """
        Parameters:
            std_3d: ndarray(16,3) of float
            data: dict{action_name - str: dict{'x':ndarray(nSample, self.inputLen) of float32,
                                               'y':ndarray(nSample, self.outputLen) of float32})
            siamese: if True, the model has two inputs and two examples are evaluated parallel
            csv: None or path to a csv file, results are also written to the given file if not none
        """
        super(LogAllMillimeterError, self).__init__()
        assert std_3d.shape == (16, 3), std_3d.shape

        self.std_3d = std_3d
        self.siamese = siamese
        self.csv = csv
        self.pctiles = [5, 10, 50, 90, 95, 99, 100]

        if self.csv is not None:
            with open(csv, 'w') as f:
                f.write('epoch,type,name,avg')
                f.write(''.join([',pct' + str(x) for x in self.pctiles]))
                f.write('\n')

        self.data_2d = {}
        self.data_3d_mm = {}
        for action_name in data.keys():
            self.data_2d[action_name] = data[action_name]['x']
            arr_3d = data[action_name]['y']
            arr_3d = arr_3d.reshape(arr_3d.shape[0], 16, 3)  # (nSample, 16,3)
            arr_3d = arr_3d * self.std_3d
            self.data_3d_mm[action_name] = arr_3d

            # Make the size of the data even by cutting off last element (only if needed)
            N = len(self.data_2d[action_name])
            if siamese and N % 2 == 1:
                self.data_2d[action_name] = self.data_2d[action_name][:N - 1, :]
                self.data_3d_mm[action_name] = self.data_3d_mm[action_name][:N - 1, :]

    def calculate_error(self, pred3d, gt3d):
        """ Calculates the error over all poses and joints, returns an ndarray(nSample,16) array"""

        pred3d = pred3d.reshape(gt3d.shape[0], 16, 3)  # (nSample, 16, 3)
        preds_mm = pred3d * self.std_3d
        err = np.linalg.norm(preds_mm - gt3d, axis=2, ord=2)  # (nSample, 16)

        return err

    def on_epoch_end(self, epoch, logs=None):
        print("    Predicting millimeter errors on validation set...")

        action_means = {}
        action_pctiles = {}
        all_errs = []

        for action_name in self.data_2d.keys():
            if self.siamese:
                # We are too lazy to cut the network in half and create a new model, instead just use
                # both branches of the siamese net
                N = len(self.data_2d[action_name])
                inputs = [self.data_2d[action_name][:N / 2, :], self.data_2d[action_name][N / 2:, :]]
                inputs.append(np.zeros((N / 2, 3, 3)))  # Add dummy values

                # Result is [pose1, pose2, embedding_dist]
                result = self.model.predict(inputs, batch_size=256, verbose=0)  # (nSample, 16*3)

                err1 = self.calculate_error(result[0], self.data_3d_mm[action_name][:N / 2, :])
                err2 = self.calculate_error(result[1], self.data_3d_mm[action_name][N / 2:, :])
                errs = np.concatenate([err1, err2], axis=0)
                assert len(errs.shape) == 2 and errs.shape[1] == 16, errs.shape
            else:
                preds = self.model.predict(self.data_2d[action_name], batch_size=256, verbose=0)  # (nSample, 16*3)
                errs = self.calculate_error(preds, self.data_3d_mm[action_name])

            action_pctiles[action_name] = np.percentile(errs, self.pctiles)
            action_means[action_name] = np.mean(errs) * float(16. / 17.)  # Multiplying by 16/17 to account for the hip
            all_errs.append(errs)

        all_errs = np.concatenate(all_errs)
        joint_means = np.mean(all_errs, axis=0)
        joint_pctiles = np.percentile(all_errs, self.pctiles, axis=0)

        assert len(all_errs.shape) == 2 and all_errs.shape[1] == 16, all_errs.shape
        assert joint_means.shape == (16,), joint_means.shape
        assert joint_pctiles.shape == (len(self.pctiles), 16), joint_pctiles.shape
        assert self.pctiles[-2] == 99, "Currently the second to last percentile is harcoded to be 99 for printing"

        if self.csv is not None:
            with open(self.csv, 'a') as f:
                for action_name in sorted(self.data_2d.keys()):
                    f.write('%d,%s,%s,%f' % (epoch, 'action', action_name, action_means[action_name]))
                    for i in range(len(self.pctiles)):
                        f.write(',%f' % action_pctiles[action_name][i])
                    f.write('\n')

                for joint_id in range(16):
                    f.write('%d,%s,%s,%f' % (epoch, 'joint', H36M.H36_JOINT_NAMES16_3D[joint_id], joint_means[joint_id]))
                    for i in range(len(self.pctiles)):
                        f.write(',%f' % joint_pctiles[i, joint_id])
                    f.write('\n')

        print(" ----- Per action and joint errors in millimeter on the validation set ----- ")
        print "    %12s   %6s      %6s   \t %16s  %6s      %6s" % ('Action', 'Avg', '99%', '', 'Avg', '99%')
        for action_name, joint_id in izip_longest(sorted(self.data_2d.keys()), range(16)):
            if action_name is not None:
                action_str = "    %-12s:  %6.2f mm   %6.2f mm\t " \
                             % (str(action_name), action_means[action_name], action_pctiles[action_name][-2])
            else:
                action_str = " " * 49

            print('%s%9s (#%2d):  %6.2f mm   %6.2f mm ' % (action_str, H36M.H36_JOINT_NAMES16_3D[joint_id], joint_id,
                                                           joint_means[joint_id], joint_pctiles[-2, joint_id]))

        mean_action_err = np.mean(np.asarray(action_means.values(), dtype=np.float32))
        self.mean_action_err = mean_action_err
        pctile99 = np.percentile(all_errs, 99)
        print("\n Mean action error is %6.2f mm, total 99%% prctile is %6.2f." % (mean_action_err, pctile99))
        print(" ---------------------------------------------------------------- ")
