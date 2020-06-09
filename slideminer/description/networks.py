# coding: utf8
"""
A module to create network encoder models.

Support only keras applications models for now.
"""
import numpy
from keras.models import Model
import keras.applications.vgg16 as vgg16
import keras.applications.vgg19 as vgg19
import keras.applications.resnet50 as res
import keras.applications.inception_v3 as inc
import keras.applications.xception as xce
from keras.layers import Dense
from keras import initializers
from keras import regularizers
from keras import constraints
from keras.engine.base_layer import InputSpec
from keras.constraints import unit_norm
import keras.backend as K


class CosineDense(Dense):
    """
    My own Dense layer with a custom activation.

    ********************************************
    """

    def __init__(self, units,
                 activation=None,
                 use_bias=False,
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=unit_norm(),
                 bias_constraint=None,
                 **kwargs):
        """
        Initialize like Dense.

        *****************************
        """
        # explicit call to parent constructor
        Dense.__init__(self, units,
                       activation=activation,
                       use_bias=use_bias,
                       kernel_initializer=kernel_initializer,
                       bias_initializer=bias_initializer,
                       kernel_regularizer=kernel_regularizer,
                       bias_regularizer=bias_regularizer,
                       activity_regularizer=activity_regularizer,
                       kernel_constraint=kernel_constraint,
                       bias_constraint=bias_constraint,
                       **kwargs)

    def build(self, input_shape):
        """
        Re-implement for free param kappa.

        For more info, see: https://elib.dlr.de/116408/1/WACV2018.pdf
        """
        assert len(input_shape) >= 2
        input_dim = input_shape[-1]

        self.kernel = self.add_weight(shape=(input_dim, self.units),
                                      initializer=self.kernel_initializer,
                                      name='kernel',
                                      regularizer=self.kernel_regularizer,
                                      constraint=self.kernel_constraint)
        self.kappa = self.add_weight(shape=(1,),
                                     initializer=initializers.Constant(value=1.),
                                     name="kappa",
                                     regularizer=regularizers.l2(1e-1),
                                     constraint=constraints.NonNeg())
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='bias',
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.bias = None
        self.input_spec = InputSpec(min_ndim=2, axes={-1: input_dim})
        self.built = True

    def call(self, inputs):
        """
        Re-implement call to include scale kappa in the output.

        *******************************************************
        """
        output = self.kappa * K.dot(inputs, self.kernel)
        # utput = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias, data_format='channels_last')
        if self.activation is not None:
            output = self.activation(output)
        return output


class Encoder:
    """
    Convenient class to store keras encoder models.

    Currently support only keras applications models.
    """

    def __init__(self, cnn_architecture, layer, patchsize):
        """
        Create a keras model.

        Given architecture name, layer to extract and patch size.
        """
        self.network_name = cnn_architecture
        self.layer_name = layer

        self.get_network_characteristics(patchsize)

        print("Feature extractor: %s // %s" % (self.network_name, self.layer_name))

    def get_network_characteristics(self, patchsize):
        """
        Actually create the keras Model object.

        ***************************************
        """
        if self.network_name == "vgg16":

            base_model = vgg16.VGG16(include_top=False, weights='imagenet', input_shape=(patchsize, patchsize, 3), pooling='avg')
            if self.layer_name == "output":
                self.model = Model(inputs=base_model.input, outputs=base_model.output)
            else:
                self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        elif self.network_name == "vgg19":
            base_model = vgg19.VGG19(include_top=False, weights='imagenet', input_shape=(patchsize, patchsize, 3), pooling='avg')
            if self.layer_name == "output":
                self.model = Model(inputs=base_model.input, outputs=base_model.output)
            else:
                self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        elif self.network_name == "resnet":
            base_model = res.ResNet50(include_top=False, weights='imagenet', input_shape=(patchsize, patchsize, 3), pooling='avg')
            if self.layer_name == "output":
                self.model = Model(inputs=base_model.input, outputs=base_model.output)
            else:
                self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        elif self.network_name == "xception":
            base_model = xce.Xception(include_top=False, weights='imagenet', input_shape=(patchsize, patchsize, 3), pooling='avg')
            if self.layer_name == "output":
                self.model = Model(inputs=base_model.input, outputs=base_model.output)
            else:
                self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        elif self.network_name == "inception":
            base_model = inc.InceptionV3(include_top=False, weights='imagenet', input_shape=(patchsize, patchsize, 3), pooling='avg')
            if self.layer_name == "output":
                self.model = Model(inputs=base_model.input, outputs=base_model.output)
            else:
                self.model = Model(inputs=base_model.input, outputs=base_model.get_layer(self.layer_name).output)

        else:
            print("Error: possible network names:\n-'vgg16'\n-'vgg19'\n-'resnet'\n-'xception'\n-'inception'")

    def predict(self, images):
        """
        Predict image encoding by the network.

        **************************************
        """
        features = self.model.predict(images)
        return [numpy.ndarray.flatten(f) for f in features]

    def load_weights(self, weight_path):
        """
        Load previously-trained weights.

        From a h5 weight file.
        """
        self.model.load_weights(weight_path)
        print("Loaded weights from: {}".format(weight_path))
