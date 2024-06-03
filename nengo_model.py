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


# class SpikingNet:
#     def __init__(self):
#         self.neuron_type = nengo.LIF(amplitude=0.01)
#         self.x = 60
#         self.y = 80
    
#     # def custom_activation(self, x):
#     #     # Apply ReLU to the first two dimensions
#     #     x_first_two_relu = tf.nn.relu(x[:, :2])
#     #     # Apply sigmoid to the last dimension
#     #     x_last_sigmoid = tf.nn.sigmoid(x[:, 2])
#     #     # Round the values of the last dimension to either 0 or 1
#     #     x_last_rounded = tf.round(x_last_sigmoid)
#     #     # Concatenate the modified dimensions
#     #     modified_x = tf.concat([x_first_two_relu, tf.expand_dims(x_last_rounded, axis=1)], axis=1)
#     #     return modified_x
    
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
            
#             # # Last dense layer with custom activation
#             # x = nengo_dl.Layer(tf.keras.layers.Dense(units=100, activation=tf.nn.relu))(x)
#             out = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.relu))(x)
#             out_p = nengo.Probe(out, label="out_p")
#             out_p_filt = nengo.Probe(out, synapse=0.001, label="out_p_filt")
            
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
            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(inp, shape_in=(self.x, self.y, 1))
            x = nengo_dl.Layer(self.neuron_type)(x)
            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(x, shape_in=(58, 78, 32))
            x = nengo_dl.Layer(self.neuron_type)(x)
            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(x, shape_in=(28, 38, 64))
            x = nengo_dl.Layer(self.neuron_type)(x)
            
            x = nengo_dl.Layer(tf.keras.layers.BatchNormalization())(x)
            x = nengo_dl.Layer(tf.keras.layers.Dropout(0.2))(x)
            
            out = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01)))(x)
            out_p = nengo.Probe(out, label="out_p")
            out_p_filt = nengo.Probe(out, synapse=0.001, label="out_p_filt")
            
            return net, inp, out_p, out_p_filt