# Interface to use data_utils.py
import data_sources.data_utils as data_utils
import data_sources.cameras as cameras
from util.misc import load
import numpy as np

H36_JOINT_NAMES17 = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax',
                     'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
H36_JOINT_NAMES16_2D = ['Hip', 'RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax',
                        'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']
H36_JOINT_NAMES16_3D = ['RHip', 'RKnee', 'RFoot', 'LHip', 'LKnee', 'LFoot', 'Spine', 'Thorax',
                        'Neck/Nose', 'Head', 'LShoulder', 'LElbow', 'LWrist', 'RShoulder', 'RElbow', 'RWrist']

H36_JOINT_SHORT_NAMES16_2D = ['hp', 'Rhp', 'Rkn', 'Rft', 'Lhp', 'Lkn', 'Lft', 'spi', 'tho',
                              'hed', 'Lsh', 'Lel', 'Lwr', 'Rsh', 'Rel', 'Rwr']
H36_JOINT_SHORT_NAMES16_3D = ['Rhp', 'Rkn', 'Rft', 'Lhp', 'Lkn', 'Lft', 'spi', 'tho',
                              'nn', 'hed', 'Lsh', 'Lel', 'Lwr', 'Rsh', 'Rel', 'Rwr']

KEEP_JOINTS_2D = np.asarray([0, 1, 2, 3, 6, 7, 8, 12, 13, 15, 17, 18, 19, 25, 26, 27], dtype=np.int32)
KEEP_JOINTS_3D = np.asarray([1, 2, 3, 6, 7, 8, 12, 13, 14, 15, 17, 18, 19, 25, 26, 27], dtype=np.int32)

# limb graphs must have the following form: list of tuple(j0 - int, j1 - int) 
#                                                   where values of j0 must not be present later in the list anywhere
LIMBGRAPH_17 = [(16, 15), (15, 14), (14, 8),
                (13, 12), (12, 11), (11, 8),
                (3, 2), (2, 1), (1, 0),
                (6, 5), (5, 4), (4, 0),
                (10, 9), (9, 8),
                (8, 7), (7, 0),
                ]  # r arm to thorax, l arm to thorax, r leg to hip, l leg to hip, head to thorax, thorax to hip
LIMBGRAPH_16_2D = [(15, 14), (14, 13), (13, 8),
                   (12, 11), (11, 10), (10, 8),
                   (3, 2), (2, 1), (1, 0),
                   (6, 5), (5, 4), (4, 0),
                   (9, 8),
                   (8, 7), (7, 0),
                   ]  # r arm to thorax, l arm to thorax, r leg to hip, l leg to hip, head to thorax, thorax to hip
LIMBGRAPH_16_3D = [(15, 14), (14, 13), (13, 7),
                   (12, 11), (11, 10), (10, 7),
                   (2, 1), (1, 0), (0, None),
                   (5, 4), (4, 3), (3, None),
                   (9, 8), (8, 7),
                   (7, 6), (6, None),
                   ]  # r arm to thorax, l arm to thorax, r leg to hip, l leg to hip, head to thorax, thorax to hip
#   None represents the origin point (hip)

SIDEDNESS_17 = np.asarray([0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
                          dtype=np.int32)  # for each joint: 0 for no side, 1 for left, 2 for right
SIDEDNESS_16_2D = np.asarray([0, 2, 2, 2, 1, 1, 1, 0, 0, 0, 1, 1, 1, 2, 2, 2],
                             dtype=np.int32)  # for each joint: 0 for no side, 1 for left, 2 for right
SIDEDNESS_16_3D = np.asarray([2, 2, 2, 1, 1, 1, 0, 0, 0, 0, 1, 1, 1, 2, 2, 2],
                             dtype=np.int32)  # for each joint: 0 for no side, 1 for left, 2 for right


class H36M():
    """
    Member fields:
        cameras_fpath: str; the cameras hdf5 archive file path
        database_folder: str; the folder to the database files

        all_action_names: list of str; the names of all actions; load with _loadActionsList()
        cam_dict: dict; the loaded cameras dict; load with _loadCameras()
        all_cam_names: list of str; the names of all cameras; load with _loadCameras()
        all_subject_ids: list of ints; the IDs of all subject IDs

        <DATA_ARRAYS>: all pose arrays are unnormalized, joints unfiltered (all 32 per coord),
                            3d pose arrays are zero centered at the hip joint

            pose_data_3d_orig: dict; the loaded 3d pose arrays; load with _loadPoseData(); HIP MOVED to origin.
            pose_data_2d_sh: dict; the loaded 2d Stacked Hourglass prediciton pose arrays; load with _loadPoseData()
            pose_data_3d_allcams: dict; the loaded 3d pose arrays transformed to each camera; load with _loadPoseData();
                                            HIP MOVED to origin.
            pose_data_2d_gt: dict; the loaded 3d pose arrays projected to each camera -> 2d gt; load with _loadPoseData()
            pose_data_3d_orig_rootpos: dict; original HIP joint positions of pose_data_3d_orig; load with _loadPoseData()
            pose_data_3d_allcams_rootpos: dict; original HIP joint positions of pose_data_3d_allcams; load with _loadPoseData()

    """

    def __init__(self, cameras_fpath, database_folder, augmented_cameras_path=None, frame_density=None, data_to_load=None):
        self.all_subject_ids = [1, 5, 6, 7, 8, 9, 11]
        self.cameras_fpath = cameras_fpath
        self.database_folder = database_folder
        self._load_actions_list()
        self._load_cameras(augmented_cameras_path)
        self._load_all_pose_data(frame_density, data_to_load)

    def _load_actions_list(self):
        """
        Loads the name of all actions.
        Sets members:
            self.all_action_names: list of str;
        """
        self.all_action_names = data_utils.define_actions("all")

    def _load_cameras(self, augmented_cameras_path=None):
        """
        Loads camera parameters for each subject with each camera.
        Sets members:
            self.cam_dict: dictionary of 4 tuples per subject ID containing its camera parameters for the 4 h36m cams
        """
        self.cam_dict = cameras.load_cameras(self.cameras_fpath, self.all_subject_ids)
        if augmented_cameras_path is not None:
            aug_cams = load(augmented_cameras_path)
            self.cam_dict.update(aug_cams)

        self.all_cam_names = []
        max_id = max([x[1] for x in self.cam_dict.keys()])  # largest camera id
        for c in range(1, max_id + 1):
            self.all_cam_names.append(self.cam_dict[(5, c)][6])

    def _load_all_pose_data(self, frame_density, data_to_load=None):
        """
        Loads all pose data.

        Sets members:
            self.pose_data_3d_orig: dict{ tuple(subject_id - int, action_name - str, filename - str):
                                    ndarray(nTimePoints, nUnfilteredJoints*nCoords=32*3=96) of float64 }
                            210 keys
                            an example key: (7, 'Purchases', 'Purchases 1.h5')
                            HIP joint moved to origin
            self.pose_data_2d_sh: dict{ tuple(subject_id - int, action_name - str, filename - str):
                                    ndarray(nTimePoints, nUnfilteredJoints*nCoords=32*2=64) of float64 }
                            839 keys - 210 for each of the 4 cameras, 1 missing because of damaged video: (11, 'Directions', ?)
                            an example key: (5, 'WalkTogether', 'WalkTogether 1.54138969.h5-sh')
            self.pose_data_3d_allcams: dict{ tuple(subject_id - int, action_name - str, filename - str):
                                    ndarray(nTimePoints, nUnfilteredJoints*nCoords=32*3=96) of float64 }
                            840 keys - 210 for each of the 4 cameras
                            an example key: (5, 'WalkTogether', 'WalkTogether 1.54138969.h5')
                            HIP joint moved to origin
            self.pose_data_2d_gt: dict{ tuple(subject_id - int, action_name - str, filename - str):
                                    ndarray(nTimePoints, nUnfilteredJoints*nCoords=32*2=64) of float64 }
                            840 keys - 210 for each of the 4 cameras
                            an example key: (5, 'WalkTogether', 'WalkTogether 1.54138969.h5')
            self.pose_data_3d_orig_rootpos: dict, same structure as pose_data_3d_orig,
                            with values shaped: ndarray(nTimePoints, nCoords=3); the original positions of the HIP joint
            self.pose_data_3d_allcams_rootpos: dict, same structure as pose_data_3d_allcams,
                            with values shaped: ndarray(nTimePoints, nCoords=3); the original positions of the HIP joint
        Parameters:
            frame_density: frequency to sample frames. Unnecessary frames are not loaded
            data_to_load: list of 2D representations to load. Array of following values: sh, shft, shft-aug.
                          If not specified, everything is loaded. Useful to save memory. Note that getDataDict
                          will throw an error if the data was not loaded beforehand.
        """
        if data_to_load is None:
            data_to_load = ['sh', 'shft-aug']

        ncams = max([x[1] for x in self.cam_dict.keys()])
        pose_data_3d_orig = data_utils.load_3d_data(self.database_folder, self.all_subject_ids, self.all_action_names,
                                                    frame_density=frame_density)
        if 'sh' in data_to_load:
            self.pose_data_2d_sh = data_utils.load_stacked_hourglass(self.database_folder, 'StackedHourglass', self.all_subject_ids,
                                                                     self.all_action_names, frame_density=frame_density)

        # shft-aug is a superset of shft, only load one of them
        if 'shft-aug' in data_to_load:
            self.pose_data_2d_sh_ft = data_utils.load_augmented_pose(self.database_folder, self.all_subject_ids, self.all_action_names,
                                                                     frame_density=frame_density)
        elif 'shft' in data_to_load:
            self.pose_data_2d_sh_ft = data_utils.load_stacked_hourglass(self.database_folder, 'StackedHourglassFineTuned240',
                                                                        self.all_subject_ids, self.all_action_names,
                                                                        frame_density=frame_density)

        pose_data_3d_allcams = data_utils.transform_world_to_camera(pose_data_3d_orig, self.cam_dict, ncams=ncams)
        self.pose_data_2d_gt = data_utils.project_to_cameras(pose_data_3d_orig, self.cam_dict, ncams=ncams)

        # only 3d is translated (hip joint moved to origin, and thus hip joint vlaues are not included in the final 16 joints for 3d)
        self.pose_data_3d_orig, self.pose_data_3d_orig_rootpos = data_utils.postprocess_3d(pose_data_3d_orig)
        self.pose_data_3d_allcams, self.pose_data_3d_allcams_rootpos = data_utils.postprocess_3d(pose_data_3d_allcams)

    def get_data_dict(self, data_name, action_names=None, camera_names=None, subject_ids=None):
        """
        Returns dict of pose arrays.
        Parameters:
            data_name: str; any of ['3dorig','3dall','2dall','2dsh','3dorig_rootpos','3dall_rootpos']
                        3dorig means absolute 3D pose, 3dall camera relative 3D pose
            action_names: None OR list of strs; None to use all actions
            camera_names: None OR list of strs; None to use all cameras
            subject_ids: None OR list of ints; None to use all subjects
        Returns:
            ret_dict: dict{original key tuple:
                            ndarray(nTimePoints, nJoints=16, nCoords=(2 or 3)) of float64
                         OR ndarray(nTimePoints, nCoords=(2 or 3)) of float64}      # if rootpos array is returned
                      DEEP COPY
        """
        DATA_SOURCES = {'3dorig': self.pose_data_3d_orig,
                        '3dgt': self.pose_data_3d_allcams,
                        '2dgt': self.pose_data_2d_gt,
                        '2dsh': self.pose_data_2d_sh,
                        '2dshft': self.pose_data_2d_sh_ft,
                        '3dorig_rootpos': self.pose_data_3d_orig_rootpos,
                        '3dgt_rootpos': self.pose_data_3d_allcams_rootpos}

        # check params
        assert data_name in DATA_SOURCES.keys()
        if data_name.startswith('3dorig'):
            assert camera_names is None, "If using camera independent coordinate system, camera_names must be None"

        if action_names is None:
            action_names = self.all_action_names
        if camera_names is None:
            camera_names = self.all_cam_names
        if subject_ids is None:
            subject_ids = self.all_subject_ids

        assert isinstance(action_names, (list, tuple))
        assert isinstance(camera_names, (list, tuple))
        assert isinstance(subject_ids, (list, tuple))
        assert set(action_names) <= set(self.all_action_names)
        assert set(camera_names) <= set(self.all_cam_names)
        assert set(subject_ids) <= set(self.all_subject_ids)

        # copy part of dict and edit dict value arrays
        source_dict = DATA_SOURCES[data_name]

        ret_dict = {}
        for key in source_dict.keys():
            subj_id, action_name, seq_name, camera_name = key

            if subj_id not in subject_ids:
                continue
            if action_name not in action_names:
                continue
            if camera_name != '' and camera_name not in camera_names:
                continue

            arr = source_dict[key].copy()
            if data_name not in ['3dorig_rootpos', '3dgt_rootpos']:
                # for shft, if loaded from augmented data source, the bad joints are already removed
                if data_name == '2dshft' and arr.shape[1] == 32:
                    arr = arr.reshape((-1, 16, 2))
                else:
                    arr = drop_unused_joints(arr)

            ret_dict[(subj_id, action_name, seq_name, camera_name)] = np.copy(arr)

        return ret_dict


def drop_unused_joints(arr):
    """
    Drops unused joints from pose array and reshapes it.

    Parameters:
        arr: ndarray(nTimePoints, nAllJoints*nCoords=32*(2 or 3))
    Returns:
        ret: ndarray(nTimePoints, nJoints=16, nCoords=(2 or 3)); a view of the original array
    """
    assert arr.ndim == 2
    assert arr.shape[1] in [64, 96]

    arr = arr.reshape((arr.shape[0], 32, -1))
    joint_idxs_to_keep = KEEP_JOINTS_2D if arr.shape[2] == 2 else KEEP_JOINTS_3D
    ret = arr[:, joint_idxs_to_keep, :]
    return ret


def pose_dict_mean_std(data_dict):
    """
    Calculates mean and std of input dictionary.

    Parameters:
        data_dict: dict{tuple_key: ndarray(nPoints, nJoints=16, nCoords=(2 or 3))} of float64
    Returns:
        data_mean, data_std: ndarray(nJoints=16, nCoords=(2 or 3)) of float64
    """
    EPSILON = .00001
    conc_data = np.concatenate(data_dict.values(), axis=0)
    data_mean = np.mean(conc_data, axis=0)
    data_std = np.std(conc_data, axis=0)
    if np.any(data_std < EPSILON):
        print("data_utils_interface.getDataDictMeanStd(): WARNING! Std array has values near or equal to zero. ")
    return data_mean, data_std


def apply_to_pose_dict(data_dict, func, *func_args):
    """
    Apply func on all arrays in the given pose dict.

    Parameters:
        data_dict: dict, dictionary whose values are updated by ``func``
        func, *func_args: function which accepts values of the dict as parameters and other args
    """
    for key in data_dict.keys():
        data_dict[key] = func(data_dict[key], *func_args)

    return data_dict


def normalize_pose_arr(data_arr, data_mean, data_std):
    """
    Normalizes a pose array.

    Parameters:
        data_arr: ndarray(nTimePoints, nJoints=16, nCoords=(2 or 3)) of float64
        data_mean, data_std: ndarray(nJoints=16, nCoords=(2 or 3)) of float64
    """
    return (data_arr - data_mean) / data_std


def create_dataset(pose_dict_2d, pose_dict_3d, all_cam_names, nSamples=None):
    """
    Creates a dataset for training/testing from a 2d and a 3d pose data dictionary. Arrays in the two dictionaries are matched
            using the key tuples. Subject ID and action name must match. Matching filenames are equal, or if no camera frame
            is used in 3d data, the 3d filename must be a substring of the 2d filename.

    Parameters:
        pose_dict_2d: dict{tuple_key: ndarray(nTimePoints, nJoints=16, nCoords=2)} of float64; the input 2d pose dict
        pose_dict_3d: dict{tuple_key: ndarray(nTimePoints, nJoints=16, nCoords=3)} of float64; the output 3d pose dict
            where tuple_key is (subject_id - int, action_name - str, filename - str)
        nSamples: int or None; if None and history is not enabled, the whole dataset is returned, otherwise specifies
    Returns:
        dataset_2d: ndarray(nSamples, hist_len, 16, 2) of float32
        dataset_3d: ndarray(nSamples, hist_len, 16, 3) of float32
    """

    # matching keys in 2d & 3d pose dicts
    dataset_arrays_2d = []
    dataset_arrays_3d = []
    for key3d in pose_dict_3d.keys():
        subj_id, action_name, fname3d, camera = key3d
        if camera != '':
            matching_keys_2d = []
            if key3d in pose_dict_2d.keys():
                # there is a single key missing in 2d sh, only then this 'if' branch is not executed
                matching_keys_2d.append(key3d)
        else:
            # 3d dict is not in camera frame, so each 3d array may match multiple 2d arrays (one for each camera)
            possible_keys_2d = [(subj_id, action_name, fname3d, cam_name) for cam_name in all_cam_names]
            matching_keys_2d = list(set(pose_dict_2d.keys()) & set(possible_keys_2d))

        # if there are multiple matching 2d keys for a single 3d key, add all corresponding 2d arrays to the list
        #            together with copies of the 3d array

        for key2d in matching_keys_2d:
            dataset_arrays_2d.append(pose_dict_2d[key2d])
            dataset_arrays_3d.append(pose_dict_3d[key3d])

    assert len(dataset_arrays_2d) == len(dataset_arrays_3d)

    dataset_2d = np.concatenate(dataset_arrays_2d, axis=0)  # (nTotalTimePoints, 16, 2)
    dataset_3d = np.concatenate(dataset_arrays_3d, axis=0)  # (nTotalTimePoints, 16, 3)

    assert dataset_2d.shape[1:] == (16, 2)
    assert dataset_3d.shape[1:] == (16, 3)

    if nSamples is not None:
        # random sample dataset
        idxs = np.random.randint(low=0, high=dataset_2d.shape[0], size=nSamples)
        dataset_2d = dataset_2d[idxs]  # (nSamples, 16, 2)
        dataset_3d = dataset_3d[idxs]  # (nSamples, 16, 3)

    assert dataset_2d.shape[0] == dataset_3d.shape[0]
    assert dataset_2d.ndim == dataset_3d.ndim == 3
    return dataset_2d.astype(np.float32), dataset_3d.astype(np.float32)
