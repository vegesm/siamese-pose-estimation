# Basic DNN model class for the 3dpose2018 project
#   tested with tensorflow-gpu 1.4, keras 2.1.4
#
#   @author Viktor Varga
#

import keras
import keras.backend as K
import keras.layers as KL
from keras.models import Model

from util.training import get_optimiser


def initializer_he(shape, dtype=None):
    '''
    He et al. initialization from https://arxiv.org/pdf/1502.01852.pdf
    '''
    return K.truncated_normal(shape, dtype=dtype) * K.sqrt(K.constant(2. / float(shape[0])))


class HeInitializerClass:
    """ Wrapper around the function ``initializer_he``, needed for loading models."""

    def __init__(self):
        self.__name__ = 'initializer_he'

    def __call__(self, shape, dtype=None):
        return initializer_he(shape, dtype=dtype)


def cut_main_branch(model):
    """
    Gets the pose estimator part of a siamese model. The returned model
    expects a 2D pose and returns a 3D pose.
    """
    main_branch = model.layers[2]
    assert isinstance(main_branch, keras.engine.training.Model), "Incorrect layer is selected"
    assert main_branch.outputs[-1].shape[1] == 48, "Got layer with shape: " + str(main_branch.outputs[-1].shape)
    main_branch = Model(inputs=main_branch.inputs, outputs=main_branch.outputs[-1])  # Keep only the pose output and throw away the embedding

    return main_branch


def dense_block(input, dense_size, n_layers, activation, dropout, residual_enabled, batchnorm_enabled, normclip_enabled, name=None):
    layer = input
    for i in range(n_layers):
        layer_name = name + "_l%d" % (i + 1) if name is not None else None
        layer = dense_layer(layer, dense_size, activation, dropout, batchnorm_enabled, normclip_enabled, name=layer_name)

    if residual_enabled:
        layer = KL.add([input, layer])
    return layer


def dense_layer(input, dense_size, activation, dropout, batchnorm_enabled, normclip_enabled, name=None):
    assert name is None or isinstance(name, basestring)

    kernel_constraint = None
    if normclip_enabled:
        kernel_constraint = keras.constraints.max_norm(max_value=1.)

    layer = KL.Dense(dense_size, activation='linear', kernel_initializer=initializer_he,
                     bias_initializer=initializer_he, kernel_constraint=kernel_constraint,
                     name=name + "_fc" if name is not None else None)(input)
    if batchnorm_enabled:
        layer = KL.normalization.BatchNormalization(name=name + "_bn" if name is not None else None)(layer)

    if activation == 'leaky_relu':
        layer = KL.LeakyReLU()(layer)
    else:
        layer = KL.Activation(activation)(layer)

    if dropout > 0.:
        layer = KL.Dropout(dropout)(layer)

    return layer


def base_network(params):
    """
    Creates one branch of the siamese network. Returns a Keras Model.

    The model has a single input (the 2D pose) and two outputs:
      1. The geometric embedding, a (3,  ``params.geometric_embedding_size``) shaped tensor
      2. The final 3D pose estimation (48 points)
    """
    input = KL.Input(shape=(2 * 16,))

    # Resize input to params.dense_size
    x = dense_layer(input, params.dense_size, params.activation, params.dropout,
                    params.batchnorm_enabled, params.normclip_enabled)

    outputs = []

    # First block (encoder)
    x = dense_block(x, params.dense_size, params.n_layers_in_block, params.activation,
                    params.dropout, params.residual_enabled, params.batchnorm_enabled,
                    params.normclip_enabled)

    # Geometric embedding
    x = dense_layer(x, 3 * params.geometric_embedding_size, 'linear', 0, False, params.normclip_enabled)
    x = KL.Reshape((3, params.geometric_embedding_size), name='embedding_prenorm')(x)
    hidden = KL.Lambda(lambda y: K.l2_normalize(y, axis=1))(x)
    x = KL.Flatten()(hidden)

    outputs.append(hidden)

    # Resize back to params.dense_size
    x = dense_layer(x, params.dense_size, params.activation, 0, False, params.normclip_enabled)

    # Second block (decoder)
    x = dense_block(x, params.dense_size, params.n_layers_in_block, params.activation,
                    params.dropout, params.residual_enabled, params.batchnorm_enabled,
                    params.normclip_enabled)

    # Final layer
    x = KL.Dense(48, activation='linear', name='y')(x)

    outputs.append(x)
    m = Model(input, outputs)
    return m


def siamese_network(params):
    """
    Creates the full siamese network.
    """
    base_model = base_network(params)

    inp1 = KL.Input(shape=(2 * 16,), name='inp1')
    inp2 = KL.Input(shape=(2 * 16,), name='inp2')
    R = KL.Input(shape=(3, 3), name='inp_R_1')  # Rotation matrices

    outputs1 = base_model(inp1)  # output of block 1
    outputs2 = base_model(inp2)  # output of block 2

    loss = ['mean_squared_error', 'mean_squared_error', 'mean_squared_error']
    inputs = [inp1, inp2, R]

    # Calculate the siamese loss
    def f(a):
        return K.batch_flatten(K.batch_dot(a[2], a[0]) - a[1])

    x = KL.Lambda(f, name='geometric_diff')([outputs1[0], outputs2[0], R])
    l_s = KL.Lambda(lambda x: K.tf.norm(x, keepdims=True, axis=1), name='norm_vecs')(x)

    outputs = [outputs1[-1], outputs2[-1], l_s]

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=get_optimiser(params),
                  loss=loss,
                  loss_weights=params.loss_weights)

    return model
