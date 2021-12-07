import tensorflow as tf
from tensorflow.keras import layers


class RandomSaturation(layers.Layer):
    """ Random saturation layer """
    def __init__(self, min=5, max=10, **kwargs):
        """ Constructor : """
        super().__init__(**kwargs)
        self.min = min
        self.max = max

    def call(self, x):
        return tf.image.random_saturation(x, self.min, self.max)


class RandomContrast(layers.Layer):
    def __init__(self, min=0.2, max=0.5, **kwargs):
        super().__init__(**kwargs)
        self.min = min
        self.max = max

    def call(self, x):
        return tf.image.random_contrast(x, self.min, self.max)


class RandomBrightness(layers.Layer):
    def __init__(self, max=0.3, **kwargs):
        super().__init__(**kwargs)
        self.max = max

    def call(self, x):
        return tf.image.random_brightness(x, self.max)
