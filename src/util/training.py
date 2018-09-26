import keras
from keras.optimizers import SGD, Adam


def exp_decay(params):
    def f(epoch):
        return params.learning_rate * (0.96 ** (epoch * 0.243))

    return f


def get_optimiser(params):
    if params.optimiser == "sgd":
        return SGD(lr=params.learning_rate, decay=1e-6, momentum=0.9, nesterov=True)
    elif params.optimiser == "adam":
        return Adam(lr=params.learning_rate, epsilon=1e-08)
    elif params.optimiser == 'rmsprop':
        return keras.optimizers.RMSprop(lr=params.learning_rate)
    else:
        raise Exception()
