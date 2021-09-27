import tensorflow as tf


def leaky_relu(features, alpha=0.1):
    return tf.where(features<0, alpha * features, features)


def block1():
    
    return tf.keras.Sequential([tf.keras.layers.Conv2D(filters=64, 
                                                       kernel_size=(7,7),
                                                       strides=(2,2),
                                                       padding='SAME',
                                                       activation=leaky_relu),

                                tf.keras.layers.MaxPool2D(pool_size=(2, 2), 
                                                          strides=(2, 2), 
                                                          padding='SAME')])


def block2():
    
    return tf.keras.Sequential([tf.keras.layers.Conv2D(filters=192, 
                                                       kernel_size=(3, 3),
                                                       padding='SAME',
                                                       activation=leaky_relu),

                                tf.keras.layers.MaxPool2D(pool_size=(2, 2), 
                                                          strides=(2, 2), 
                                                          padding='SAME')])


def block3():
    
    return tf.keras.Sequential([tf.keras.layers.Conv2D(filters=128, 
                                                       kernel_size=(1, 1),
                                                       padding='SAME',
                                                       activation=leaky_relu),

                                tf.keras.layers.Conv2D(filters=256, 
                                                        kernel_size=(3, 3),
                                                        padding='SAME',
                                                        activation=leaky_relu),

                                tf.keras.layers.Conv2D(filters=256, 
                                                        kernel_size=(1, 1),
                                                        padding='SAME',
                                                        activation=leaky_relu),
                                    
                                tf.keras.layers.Conv2D(filters=512, 
                                                        kernel_size=(3, 3),
                                                        padding='SAME',
                                                        activation=leaky_relu),

                                tf.keras.layers.MaxPool2D(pool_size=(2, 2), 
                                                            strides=(2, 2), 
                                                            padding='SAME')])

                
def block4(self):

    layers1 = [tf.keras.layers.Conv2D(filters=256, 
                                     kernel_size=(1, 1),
                                     padding='SAME',
                                     activation=leaky_relu),

              tf.keras.layers.Conv2D(filters=512, 
                                     kernel_size=(3, 3),
                                     padding='SAME',
                                     activation=leaky_relu),] * 4


    layers2 = [tf.keras.layers.Conv2D(filters=512, 
                                      kernel_size=(1, 1),
                                      padding='SAME',
                                      activation=leaky_relu),
                                      
               tf.keras.layers.Conv2D(filters=1024, 
                                      kernel_size=(3, 3),
                                      padding='SAME',
                                      activation=leaky_relu),

               tf.keras.layers.MaxPool2D(pool_size=(2, 2), 
                                strides=(2, 2), 
                                padding='SAME')]

    
    return tf.keras.Sequential(layers1 + layers2)


def block5():
    layers1 = [tf.keras.layers.Conv2D(filters=512, 
                                     kernel_size=(1, 1),
                                     padding='SAME',
                                     activation=leaky_relu),

              tf.keras.layers.Conv2D(filters=1024, 
                                     kernel_size=(3, 3),
                                     padding='SAME',
                                     activation=leaky_relu),] * 2


    layers2 = [tf.keras.layers.Conv2D(filters=1024, 
                                      kernel_size=(3, 3),
                                      padding='SAME',
                                      activation=leaky_relu),
                                      
               tf.keras.layers.Conv2D(filters=1024, 
                                      kernel_size=(3, 3),
                                      strides=(2, 2),
                                      padding='SAME',
                                      activation=leaky_relu)]

    
    return tf.keras.Sequential(layers1 + layers2)


def block6():

    return tf.keras.Sequential([tf.keras.layers.Conv2D(filters=1024, 
                                                       kernel_size=(3, 3),
                                                       padding='SAME',
                                                       activation=leaky_relu),

                                tf.keras.layers.Conv2D(filters=1024, 
                                                       kernel_size=(3, 3),
                                                       padding='SAME',
                                                       activation=leaky_relu)])



class YOLO(tf.keras.Model):

    def __init__(self,
                 n_class, 
                 image_shape,
                 n_grids=7,
                 n_bounding_box=2, 
                 *args, 
                 **kwargs):

        super().init(self, *args, **kwargs)

        self._n_class = n_class
        self._image_shape = image_shape
        self._n_grids = n_grids
        self._n_bounding_box = n_bounding_box

        self.blocks = [block1(),
                       block2(),
                       block3(),
                       block4(),
                       block5(),
                       block6()]

        self.flatten = tf.keras.layers.Flatten()
        
        self.dense = tf.keras.layers.Dense(4096)

        self.activation = tf.keras.layers.LeakyRelu(alpha=0.1)

        self.prediction = tf.keras.layers.Dense(n_grids**2 * (5*n_bounding_box + n_class))

        self.reshape = tf.keras.layers.Reshape(shape=(n_grids,
                                                      n_grids,
                                                      1, 
                                                      5*n_bounding_box + n_class))


    def call(self, inputs):
        
        for block in self.blocks:
            inputs = block(inputs)

        outputs = self.flatten(inputs)
        outputs = self.dense(outputs)
        outputs = self.activation(outputs)
        outputs = self.prediction(outputs)
        outputs = self.reshape(outputs)

        return outputs
