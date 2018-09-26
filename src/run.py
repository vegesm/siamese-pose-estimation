import argparse
import os

from keras.models import load_model
from keras.callbacks import LearningRateScheduler
from util.misc import save, ensuredir, load

from data_sources.h36m_interface import H36M, pose_dict_mean_std, normalize_pose_arr, apply_to_pose_dict, create_dataset
from logger import LogAllMillimeterError
from model import siamese_network, HeInitializerClass, cut_main_branch
from util.misc import Params
from generators import SplitBatchGenerator, PosNegBatchGenerator
from util.pose_utils import get_pose_count
from util.training import exp_decay

CAMERAS_FPATH = "../data/h36m/cameras.h5"
DATABASE_FOLDER = "../data/h36m/"
LOG_PATH = '../data/experiments'

TRAIN_SUBJECT_IDS = [1, 5, 6, 7, 8]
TEST_SUBJECT_IDS = [9, 11]


def run_experiment(test_dict_2d, test_dict_3d, train_dict_2d, train_dict_3d, train_dict_3d_absolute,
                   test_dict_actionwise, cam_dict, normalisation_params, out_path, p):
    """
    Runs the experiment specified by the params object `p`. It saves additional data in a
    timestamped folder under LOG_PATH, like network structure, trained model, training loss/errors, etc.
    """
    model = siamese_network(p)

    print "Parameters:"
    print p

    ensuredir(out_path)
    save(os.path.join(out_path, 'params.txt'), str(p))
    save(os.path.join(out_path, 'normalisation.pkl'), normalisation_params)

    train_generator = PosNegBatchGenerator(train_dict_2d, train_dict_3d, p, shuffle=True, cams=cam_dict,
                                           train_dict_3d_absolute=train_dict_3d_absolute)
    valid_generator = SplitBatchGenerator(test_dict_2d, test_dict_3d, p)

    callbacks = []

    if p.weight_decay:
        callbacks.append(LearningRateScheduler(exp_decay(p), verbose=1))

    callbacks.append(LogAllMillimeterError(normalisation_params['std_3d'], test_dict_actionwise, siamese=len(model.inputs) >= 2,
                                           csv=os.path.join(out_path, 'metrics.log')))

    print "Output folder is", out_path
    result = model.fit_generator(train_generator,
                                 steps_per_epoch=p.TRAIN_SIZE / p.batch_size, epochs=p.num_epochs, callbacks=callbacks,
                                 validation_data=valid_generator, validation_steps=p.VALID_SIZE / p.batch_size, verbose=2)

    model.save(os.path.join(out_path, 'model.h5'))

    return result


def siamese_params(test_dict_2d, train_dict_2d):
    """
    Parameters for the siamese network architecture.
    """
    p = Params()
    p.batch_size = 256
    p.learning_rate = 0.001
    p.optimiser = "adam"

    p.normclip_enabled = False
    p.dense_size = 1024
    p.n_layers_in_block = 2
    p.residual_enabled = True
    p.batchnorm_enabled = True

    p.VALID_SIZE = get_pose_count(test_dict_2d)
    p.TRAIN_SIZE = get_pose_count(train_dict_2d) * 21 / 2  # this way the network sees the same amount of poses as without siamese architecture
    p.siamese_loss = 'pose_dist_loss'
    p.pose_dist_weight = 0.01
    p.pos_pairs_ratio = 0.5
    p.weight_decay = True
    p.activation = 'leaky_relu'

    p.geometric_embedding_size = 128

    p.dropout = 0.2
    p.loss_weights = [0.5, 0.5, 0.1]

    return p


def train(use_augmentation, cross_camera, epochs, model_folder):
    data_type_2d = '2dshft'  # Type of 2D input: 2dgt-ground truth, 2dsh-Stacked Hourglass estimation, 2dshft-Finetuned Stacked Hourglass
    frame_density = 5 if use_augmentation else 1  # Sampling of video frames, higher for augmented cameras as there would be too many

    assert not use_augmentation or data_type_2d == '2dgt' or data_type_2d == '2dshft', \
        "USE_AUGMENTED_CAMS only works with GT or SH-FT data"

    print "Start loading data"
    if use_augmentation:
        data_if = H36M(CAMERAS_FPATH, DATABASE_FOLDER, '../data/h36m/aug-cams.pkl', frame_density=frame_density)
    else:
        data_if = H36M(CAMERAS_FPATH, DATABASE_FOLDER, frame_density=frame_density)

    if cross_camera:
        test_camera_names = ['60457274']
        if use_augmentation:
            train_camera_names = [x for x in data_if.all_cam_names if x not in test_camera_names]
        else:
            train_camera_names = ['55011271', '58860488', '54138969']
    else:
        assert not use_augmentation, "Augmented cameras are not supported in non-camera split mode"
        train_camera_names = ['55011271', '58860488', '54138969', '60457274']
        test_camera_names = ['55011271', '58860488', '54138969', '60457274']

    train_dict_2d = data_if.get_data_dict(data_type_2d, camera_names=train_camera_names, subject_ids=TRAIN_SUBJECT_IDS)
    test_dict_2d = data_if.get_data_dict(data_type_2d, camera_names=test_camera_names, subject_ids=TEST_SUBJECT_IDS)
    train_dict_3d = data_if.get_data_dict('3dgt', camera_names=train_camera_names, subject_ids=TRAIN_SUBJECT_IDS)
    test_dict_3d = data_if.get_data_dict('3dgt', camera_names=test_camera_names, subject_ids=TEST_SUBJECT_IDS)
    train_dict_3d_absolute = data_if.get_data_dict('3dorig', camera_names=None, subject_ids=TRAIN_SUBJECT_IDS)
    train_dict_3d_absolute = {(k[0], k[1], k[2]): v for k, v in train_dict_3d_absolute.iteritems()}

    for k in train_dict_2d:
        assert len(train_dict_2d[k]) == len(train_dict_3d[k]), "Error for key " + str(k)

    for k in test_dict_2d:
        assert len(test_dict_2d[k]) == len(test_dict_2d[k]), "Error for key " + str(k)

    # Normalization
    mean_2d, std_2d = pose_dict_mean_std(train_dict_2d)
    mean_3d, std_3d = pose_dict_mean_std(train_dict_3d)
    apply_to_pose_dict(train_dict_2d, normalize_pose_arr, mean_2d, std_2d)
    apply_to_pose_dict(test_dict_2d, normalize_pose_arr, mean_2d, std_2d)
    apply_to_pose_dict(train_dict_3d, normalize_pose_arr, mean_3d, std_3d)
    apply_to_pose_dict(test_dict_3d, normalize_pose_arr, mean_3d, std_3d)

    # Actionwise split of test set for error metric calculation
    test_dict_actionwise = {}
    for action_name in data_if.all_action_names:
        action_dict_2d = data_if.get_data_dict(data_type_2d, action_names=[action_name], camera_names=test_camera_names,
                                               subject_ids=TEST_SUBJECT_IDS)
        action_dict_3d = data_if.get_data_dict('3dgt', action_names=[action_name], camera_names=test_camera_names,
                                               subject_ids=TEST_SUBJECT_IDS)

        apply_to_pose_dict(action_dict_2d, normalize_pose_arr, mean_2d, std_2d)
        apply_to_pose_dict(action_dict_3d, normalize_pose_arr, mean_3d, std_3d)

        action2d, action3d = create_dataset(action_dict_2d, action_dict_3d, data_if.all_cam_names)
        action2d = action2d.reshape((action2d.shape[0], -1))
        action3d = action3d.reshape((action3d.shape[0], -1))

        test_dict_actionwise[action_name] = {'x': action2d, 'y': action3d}

    print "Data loaded"

    p = siamese_params(test_dict_2d, train_dict_2d)
    p.use_augmented_cams = use_augmentation
    p.frame_density = frame_density
    p.data_type_2d = data_type_2d
    p.train_camera_names = train_camera_names
    p.test_camera_names = test_camera_names
    p.num_epochs = epochs

    normalisation_params = {'mean_2d': mean_2d, 'std_2d': std_2d, 'mean_3d': mean_3d, 'std_3d': std_3d}

    run_experiment(test_dict_2d, test_dict_3d, train_dict_2d, train_dict_3d, train_dict_3d_absolute, test_dict_actionwise,
                   data_if.cam_dict, normalisation_params, model_folder, p)


def eval(cross_camera, model_folder):
    """
    Evaluates the given model.

    :param cross_camera: if True, uses Protocol #3 (cross-camera setup) otherwise protocol 1
    :param model_folder: the folder that contains the ``model.h5`` and ``normalisation.pkl`` files.
    """

    data_type_2d = '2dshft'  # Type of 2D input: 2dgt-ground truth, 2dsh-Stacked Hourglass estimation, 2dshft-Finetuned Stacked Hourglass
    frame_density = 5 if cross_camera else 1  # Sampling of video frames

    data_if = H36M(CAMERAS_FPATH, DATABASE_FOLDER, frame_density=frame_density)

    if cross_camera:
        test_camera_names = ['60457274']
    else:
        test_camera_names = ['55011271', '58860488', '54138969', '60457274']

    norm = load(os.path.join(model_folder, 'normalisation.pkl'))

    # Actionwise split of test set for error metric calculation
    test_dict_actionwise = {}
    for action_name in data_if.all_action_names:
        action_dict_2d = data_if.get_data_dict(data_type_2d, action_names=[action_name], camera_names=test_camera_names,
                                               subject_ids=TEST_SUBJECT_IDS)
        action_dict_3d = data_if.get_data_dict('3dgt', action_names=[action_name], camera_names=test_camera_names,
                                               subject_ids=TEST_SUBJECT_IDS)

        apply_to_pose_dict(action_dict_2d, normalize_pose_arr, norm['mean_2d'], norm['std_2d'])
        apply_to_pose_dict(action_dict_3d, normalize_pose_arr, norm['mean_3d'], norm['std_3d'])

        action2d, action3d = create_dataset(action_dict_2d, action_dict_3d, data_if.all_cam_names)
        action2d = action2d.reshape((action2d.shape[0], -1))
        action3d = action3d.reshape((action3d.shape[0], -1))

        test_dict_actionwise[action_name] = {'x': action2d, 'y': action3d}

    m = load_model(os.path.join(model_folder, 'model.h5'), custom_objects={'initializer_he': HeInitializerClass})
    m = cut_main_branch(m)

    evaluator = LogAllMillimeterError(norm['std_3d'], test_dict_actionwise)
    evaluator.model = m
    evaluator.on_epoch_end(None)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", help="evaluate a model instead of training", action="store_true")
    parser.add_argument("--use-augmentation", help="use additional augmented camera views", action="store_true")
    parser.add_argument("--cross-camera", help="use Protocol 3", action="store_true")
    parser.add_argument('--epochs', default=100, type=int, help='number of epochs to train the model')
    parser.add_argument('--model-folder', default='../data/model', type=str, help='folder where the model is saved/loaded from')
    args = parser.parse_args()

    if args.eval:
        eval(args.cross_camera, args.model_folder)
    else:
        train(args.use_augmentation, args.cross_camera, args.epochs, args.model_folder)


if __name__ == "__main__":
    main()
