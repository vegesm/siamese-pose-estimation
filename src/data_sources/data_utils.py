# The code in this file is based on https://github.com/una-dinosauria/3d-pose-baseline
"""Utility functions for dealing with human3.6m data."""

from __future__ import division

import os
import numpy as np
from data_sources import cameras

import h5py
import glob
import copy

# Human3.6m IDs for training and testing
TRAIN_SUBJECTS = [1, 5, 6, 7, 8]
TEST_SUBJECTS = [9, 11]

# Joints in H3.6M -- data has 32 joints, but only 17 that move; these are the indices.
H36M_NAMES = [''] * 32
H36M_NAMES[0] = 'Hip'
H36M_NAMES[1] = 'RHip'
H36M_NAMES[2] = 'RKnee'
H36M_NAMES[3] = 'RFoot'
H36M_NAMES[6] = 'LHip'
H36M_NAMES[7] = 'LKnee'
H36M_NAMES[8] = 'LFoot'
H36M_NAMES[12] = 'Spine'
H36M_NAMES[13] = 'Thorax'
H36M_NAMES[14] = 'Neck/Nose'
H36M_NAMES[15] = 'Head'
H36M_NAMES[17] = 'LShoulder'
H36M_NAMES[18] = 'LElbow'
H36M_NAMES[19] = 'LWrist'
H36M_NAMES[25] = 'RShoulder'
H36M_NAMES[26] = 'RElbow'
H36M_NAMES[27] = 'RWrist'

# Stacked Hourglass produces 16 joints. These are the names.
SH_NAMES = [''] * 16
SH_NAMES[0] = 'RFoot'  # sh -> 2dbaseline: 2d = sh[[6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10]]
SH_NAMES[1] = 'RKnee'
SH_NAMES[2] = 'RHip'
SH_NAMES[3] = 'LHip'
SH_NAMES[4] = 'LKnee'
SH_NAMES[5] = 'LFoot'
SH_NAMES[6] = 'Hip'
SH_NAMES[7] = 'Spine'
SH_NAMES[8] = 'Thorax'
SH_NAMES[9] = 'Head'
SH_NAMES[10] = 'RWrist'
SH_NAMES[11] = 'RElbow'
SH_NAMES[12] = 'RShoulder'
SH_NAMES[13] = 'LShoulder'
SH_NAMES[14] = 'LElbow'
SH_NAMES[15] = 'LWrist'


def load_3d_data(bpath, subjects, actions, frame_density=None):
    """
    Loads 2d ground truth from disk, and puts it in an easy-to-acess dictionary

    Args
      bpath: String. Path where to load the data from
      subjects: List of integers. Subjects whose data will be loaded
      actions: List of strings. The actions to load
      dim: Integer={2,3}. Load 2 or 3-dimensional data
    Returns:
      data: Dictionary with keys k=(subject, action, seqname)
        values v=(nx(32*2) matrix of 2d ground truth)
        There will be 2 entries per subject/action if loading 3d data
        There will be 8 entries per subject/action if loading 2d data
    """

    data = {}

    for subj in subjects:
        for action in actions:
            dpath = os.path.join(bpath, 'S{0}'.format(subj), 'MyPoses/3D_positions', '{0}*.h5'.format(action))
            fnames = glob.glob(dpath)

            loaded_seqs = 0
            for fname in fnames:
                vidname = os.path.basename(fname)
                parts = vidname.split('.')
                assert len(parts) == 2, "Incorrect filename: " + vidname
                seqname = parts[0]

                # This rule makes sure SittingDown is not loaded when Sitting is requested
                if action == "Sitting" and seqname.startswith("SittingDown"):
                    continue

                # This rule makes sure that WalkDog and WalkTogeter are not loaded when
                # Walking is requested.
                if seqname.startswith(action):
                    # print( fname )
                    loaded_seqs = loaded_seqs + 1

                    with h5py.File(fname, 'r') as h5f:
                        poses = h5f['3D_positions'][:]

                    poses = poses.T
                    # This filtering must be done here, otherwise a memory leak occurs
                    if frame_density is not None:
                        poses = poses[::frame_density, :].copy()

                    data[(subj, action, seqname, '')] = poses

            assert loaded_seqs == 2, "Expecting 2 sequences, found {0} instead".format(loaded_seqs)

    return data


def find_pose_files(data_dir, subtype, subjects, actions):
    """
    Loads the filenames for the given subjects, actions and pose type. Return value is a dictionary,
    where the keys are the tuple ``(subject, action, sequence name, camera name)``.
    """
    files = {}
    for subj in subjects:
        for action in actions:
            dpath = os.path.join(data_dir, 'S{0}'.format(subj), '{0}/{1}*.*'.format(subtype, action))
            full_names = glob.glob(dpath)

            files_found = 0
            for full_name in full_names:
                vidname = os.path.basename(full_name)
                parts = vidname.split('.')
                assert len(parts) == 3, "Incorrect filename: " + vidname
                seq_name = parts[0]
                seq_name = seq_name.replace('_', ' ')
                camera_name = parts[1]

                # This rule makes sure SittingDown is not loaded when Sitting is requested
                if action == "Sitting" and seq_name.startswith("SittingDown"):
                    continue

                # This rule makes sure that WalkDog and WalkTogeter are not loaded when
                # Walking is requested.
                if not seq_name.startswith(action):
                    continue

                files_found = files_found + 1
                files[(subj, action, seq_name, camera_name)] = full_name

            # Make sure we have found all the sequences
            if subj == 11 and action == 'Directions':  # <-- this video is damaged
                assert files_found in [7, 43], "Expecting 7 or 43sequences, found {0} instead. S:{1} {2}".format(files_found, subj, action)
            else:
                assert files_found in [8, 44], "Expecting 8 or 44 sequences, found {0} instead. S:{1} {2}".format(files_found, subj, action)

    return files


def load_stacked_hourglass(data_dir, subtype, subjects, actions, frame_density=None):
    """
    Load 2d detections from disk, and put it in an easy-to-acess dictionary.

    Args
      data_dir: string. Directory where to load the data from,
      subtype: string. Type of 2D detection, either StackedHourglass or StackedHourglassFineTuned240
      subjects: list of integers. Subjects whose data will be loaded.
      actions: list of strings. The actions to load.
    Returns
      data: dictionary with keys k=(subject, action, seqname)
            values v=(nx(32*2) matrix of 2d stacked hourglass detections)
            There will be 2 entries per subject/action if loading 3d data
            There will be 8 entries per subject/action if loading 2d data
    """
    # Permutation that goes from SH detections to H36M ordering.
    SH_TO_GT_PERM = np.array([SH_NAMES.index(h) for h in H36M_NAMES if h != '' and h in SH_NAMES])
    assert np.all(SH_TO_GT_PERM == np.array([6, 2, 1, 0, 3, 4, 5, 7, 8, 9, 13, 14, 15, 12, 11, 10]))
    assert subtype in ['StackedHourglass', 'StackedHourglassFineTuned240']

    files = find_pose_files(data_dir, subtype, subjects, actions)
    data = {}
    for key, fname in files.iteritems():
        # Load the poses from the .h5 file
        with h5py.File(fname, 'r') as h5f:
            poses = h5f['poses'][:]

            # Permute the loaded data to make it compatible with H36M
            poses = poses[:, SH_TO_GT_PERM, :]

            # Reshape into n x (32*2) matrix
            poses = np.reshape(poses, [poses.shape[0], -1])
            poses_final = np.zeros([poses.shape[0], len(H36M_NAMES) * 2])

            dim_to_use_x = np.where(np.array([x != '' and x != 'Neck/Nose' for x in H36M_NAMES]))[0] * 2
            dim_to_use_y = dim_to_use_x + 1

            dim_to_use = np.zeros(len(SH_NAMES) * 2, dtype=np.int32)
            dim_to_use[0::2] = dim_to_use_x
            dim_to_use[1::2] = dim_to_use_y
            poses_final[:, dim_to_use] = poses

            if frame_density is not None:
                poses_final = poses_final[::frame_density, :].copy()

            data[key] = poses_final

    return data


def load_augmented_pose(data_dir, subjects, actions, frame_density=None):
    """
    Load augmented 2d detections from disk, and put it in an easy-to-acess dictionary.

    Args
      data_dir: string. Directory where to load the data from,
      subtype: string. Type of 2D detection, either StackedHourglass or StackedHourglassFineTuned240
      subjects: list of integers. Subjects whose data will be loaded.
      actions: list of strings. The actions to load.
    Returns
      data: dictionary with keys k=(subject, action, seqname)
            values v=(nx(32*2) matrix of 2d stacked hourglass detections)
            There will be 2 entries per subject/action if loading 3d data
            There will be 8 entries per subject/action if loading 2d data
    """
    files = find_pose_files(data_dir, 'StackedHourglassFineTuned240Augmented', subjects, actions)

    data = {}
    for key, fname in files.iteritems():
        # Load the poses from the .npy file
        poses = np.load(fname)
        if frame_density is not None:
            poses = poses[::frame_density, :].copy()

        data[key] = poses

    return data


def transform_world_to_camera(poses_set, cams, ncams=4, frame_density=None):
    """
    Project 3d poses from world coordinate to camera coordinate system
    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with cameras
      ncams: number of cameras per subject
    Return:
      t3d_camera: dictionary with 3d poses in camera coordinate
    """
    t3d_camera = {}
    for t3dk in sorted(poses_set.keys()):

        subj, action, seqname, camera_name = t3dk
        assert camera_name == '', t3dk
        t3d_world = poses_set[t3dk]

        for c in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, c + 1)]
            camera_coord = cameras.world_to_camera_frame(np.reshape(t3d_world, [-1, 3]), R, T)
            camera_coord = np.reshape(camera_coord, [-1, len(H36M_NAMES) * 3])

            # This filtering must be done here, otherwise a memoryleak occurs
            if frame_density is not None:
                camera_coord = camera_coord[::frame_density, :].copy()

            t3d_camera[(subj, action, seqname, name)] = camera_coord

    return t3d_camera


def define_actions(action):
    """
    Given an action string, returns a list of corresponding actions.

    Args
      action: String. either "all" or one of the h36m actions
    Returns
      actions: List of strings. Actions to use.
    Raises
      ValueError: if the action is not a valid action in Human 3.6M
    """
    actions = ["Directions", "Discussion", "Eating", "Greeting",
               "Phoning", "Photo", "Posing", "Purchases",
               "Sitting", "SittingDown", "Smoking", "Waiting",
               "WalkDog", "Walking", "WalkTogether"]

    if action == "All" or action == "all":
        return actions

    if not action in actions:
        raise (ValueError, "Unrecognized action: %s" % action)

    return [action]


def project_to_cameras(poses_set, cams, ncams=4, frame_density=None):
    """
    Project 3d poses using camera parameters

    Args
      poses_set: dictionary with 3d poses
      cams: dictionary with camera parameters
      ncams: number of cameras per subject
    Returns
      t2d: dictionary with 2d poses
    """
    t2d = {}

    for t3dk in sorted(poses_set.keys()):
        subj, a, seqname, camera_name = t3dk
        assert camera_name==''
        t3d = poses_set[t3dk]

        for cam in range(ncams):
            R, T, f, c, k, p, name = cams[(subj, cam + 1)]
            pts2d, _, _, _, _ = cameras.project_point_radial(np.reshape(t3d, [-1, 3]), R, T, f, c, k, p)
            pts2d = np.reshape(pts2d, [-1, len(H36M_NAMES) * 2])

            # This filtering must be done here, otherwise a memory leak occurs
            if frame_density is not None:
                pts2d = pts2d[::frame_density, :].copy()  # copy needed to avoid views and thus holding on to memory

            # sname = seqname[:-3] + "." + name + ".h5"  # e.g.: Waiting 1.58860488.h5
            t2d[(subj, a, seqname, name)] = pts2d

    return t2d


def postprocess_3d(poses_set):
    """
    Center 3d points around root.

    Args
      poses_set: dictionary with 3d data
    Returns
      poses_set: dictionary with 3d data centred around root (center hip) joint
      root_positions: dictionary with the original 3d position of each pose
    """
    root_positions = {}
    for k in poses_set.keys():
        # Keep track of the global position
        root_positions[k] = copy.deepcopy(poses_set[k][:, :3])

        # Remove the root from the 3d position
        poses = poses_set[k]
        poses = poses - np.tile(poses[:, :3], [1, len(H36M_NAMES)])
        poses_set[k] = poses

    return poses_set, root_positions
