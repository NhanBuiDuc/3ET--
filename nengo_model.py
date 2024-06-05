from sklearn.model_selection import learning_curve
import nengo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import nengo_dl

# class SpikingNet:
#     def __init__(self):
#         self.neuron_type = nengo.LIF(amplitude=0.01)
#         self.x = 60
#         self.y = 80
#     def build_model(self):
#         with nengo.Network(seed=0) as net:
#             net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
#             net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
#             net.config[nengo.Connection].synapse = None
#             nengo_dl.configure_settings(stateful=False)
#             inp = nengo.Node(np.zeros(self.x * self.y))
#             x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(inp, shape_in=(self.x, self.y, 1))
#             x = nengo_dl.Layer(self.neuron_type)(x)
#             x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(x, shape_in=(58, 78, 32))
#             x = nengo_dl.Layer(self.neuron_type)(x)
#             x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(x, shape_in=(28, 38, 64))
#             x = nengo_dl.Layer(self.neuron_type)(x)
#             out = nengo_dl.Layer(tf.keras.layers.Dense(units=3))(x)
#             out_p = nengo.Probe(out, label="out_p")
#             out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")
#             return net, inp, out_p, out_p_filt

class SpikingNet:
    def __init__(self):
        self.neuron_type = nengo.LIF(amplitude=0.01)
        self.x = 60
        self.y = 80

    def build_model(self):
        with nengo.Network(seed=0) as net:
            net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
            net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            # net.config[nengo.Connection].synapse = None
            # nengo_dl.configure_settings(stateful=False)
            
            inp = nengo.Node(np.zeros(self.x * self.y))
            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(inp, shape_in=(self.x, self.y, 1))
            x = nengo_dl.Layer(self.neuron_type)(x)
            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(x, shape_in=(58, 78, 32))
            x = nengo_dl.Layer(self.neuron_type)(x)
            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(x, shape_in=(28, 38, 64))
            x = nengo_dl.Layer(self.neuron_type)(x)
            
            x = nengo_dl.Layer(tf.keras.layers.BatchNormalization())(x)
            x = nengo_dl.Layer(tf.keras.layers.Dropout(0.2))(x)
            
            out = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid))(x)
            out_p = nengo.Probe(out, label="out_p")
            out_p_filt = nengo.Probe(out, synapse=0.001, label="out_p_filt")
            
            return net, inp, out_p, out_p_filt
class TestNet:
    def __init__(self):
        self.neuron_type = nengo.LIF(amplitude=0.01)
        self.input_shape = (1, 60, 80)

    def build_model(self):
        with nengo.Network() as model:
            # Define a node to concatenate the outputs of the individual probes
            def concat_probes(x):
                return x
            # Define input node representing the input image
            def inp_cllbck(t, data):
                return data
            inp = nengo.Node(size_in=0, output = np.zeros(self.input_shape[1] * self.input_shape[2]))
            print(np.prod(self.input_shape))
            print("Output: ", inp.size_out)
            
            conv1_transform = nengo.Convolution(

                n_filters=32,

                input_shape=self.input_shape,

                kernel_size=(3, 3),

                padding="same"

            )
            conv2_transform = nengo.Convolution(

                n_filters=64,

                input_shape=conv1_transform.output_shape.shape,

                kernel_size=(3, 3),

                padding="same"

            )
            conv3_transform = nengo.Convolution(

                n_filters=128,

                input_shape=conv2_transform.output_shape.shape,

                kernel_size=(3, 3),

                padding="same"

            )
            print(conv1_transform.output_shape)
            # Define convolutional layer

            conv1_feat = nengo.Ensemble(

                n_neurons = np.prod(conv1_transform.output_shape.shape), dimensions = 100, neuron_type = nengo.LIF(),
            )
            conv2_feat = nengo.Ensemble(

                n_neurons = np.prod(conv2_transform.output_shape.shape), dimensions = 100, neuron_type = nengo.LIF(),
            )
            conv3_feat = nengo.Ensemble(

                n_neurons = np.prod(conv3_transform.output_shape.shape), dimensions = 100, neuron_type = nengo.LIF(),
            )
            ens_1 = nengo.Ensemble(n_neurons = np.prod(conv3_transform.output_shape.shape), dimensions=100, neuron_type=nengo.RectifiedLinear())
            ens_2 = nengo.Ensemble(n_neurons = np.prod(ens_1.n_neurons), dimensions=10, neuron_type=nengo.RectifiedLinear())
            ens_3 = nengo.Ensemble(n_neurons = np.prod(ens_2.n_neurons), dimensions=3, neuron_type=nengo.Sigmoid())

            nengo.Connection(pre = inp, post = conv1_feat.neurons, synapse = 0.01, transform=conv1_transform)
            nengo.Connection(pre = conv1_feat.neurons, post = conv2_feat.neurons, synapse = 0.01, transform=conv2_transform)
            nengo.Connection(pre = conv2_feat.neurons, post = conv3_feat.neurons, synapse = 0.01, transform=conv3_transform)
            nengo.Connection(conv3_feat, ens_1.neurons, synapse=0.01, transform=nengo_dl.dists.Glorot())
            nengo.Connection(ens_1.neurons, ens_2.neurons, synapse=0.01, transform=nengo_dl.dists.Glorot())
            nengo.Connection(ens_2.neurons, ens_3.neurons, synapse=0.01, transform=nengo_dl.dists.Glorot())
            # out = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid))(conv3_feat)
            out_p = nengo.Probe(ens_3, label="out_p")
            out_p_filt = nengo.Probe(ens_3, synapse=0.01, label="out_p_filt")
            return model, inp, out_p, out_p_filt
# import nengo

# import numpy as np

# from nengo.utils.numpy import rmse



# # Define input signal

# input_shape = (60, 80, 1)  # Assuming grayscale images with size 28x28

# n_filters = 32

# kernel_size = (3, 3)

# # Input callback function
# def inp_cllbck(t, data):
#     return np.zeros(np.prod(input_shape))
    
# # Create a function to define a convolutional block
# def conv_block(network, n_filters, input_shape, n_layers):
#     for _ in range(n_layers):
#         conv_transform = nengo.Convolution(
#             n_filters=n_filters,
#             input_shape=input_shape,
#             kernel_size=(3, 3),
#             padding=conv_params["padding"],
#             strides=conv_params["strides"]
#         )
#         input_shape = conv_transform.output_shape.shape
#         network.append((conv_transform, input_shape))
#     return input_shape
    
# with nengo.Network() as model:

#     # Define input node representing the input image

#     inp = nengo.Node(size_in=1, output = inp_cllbck)
#     print(np.prod(input_shape))
#     print("Output: ", inp.size_out)
    
#     conv_transform = nengo.Convolution(

#         n_filters=n_filters,

#         input_shape=input_shape,

#         kernel_size=kernel_size,

#         padding="same"

#     )
#     print(conv_transform.output_shape)
#     # Define convolutional layer

#     conv1_feat = nengo.Ensemble(

#         n_neurons = 1000, dimensions = np.prod(conv_transform.output_shape.shape), neuron_type = nengo.SpikingRectifiedLinear()
#     )
#     nengo.Connection(pre = inp, post = conv1_feat, synapse = 0.001, transform=conv_transform)

  