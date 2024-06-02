import nengo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

import nengo_dl

class SpikingNet:
    def __init__(self, n_time_bins):
        self.neuron_type = nengo.LIF(amplitude=0.01)
        self.x = 60
        self.y = 80
        self.n_time_bins = n_time_bins
    def build_model(self):
        with nengo.Network(seed=0) as net:
            net.config[nengo.Ensemble].max_rates = nengo.dists.Choice([100])
            net.config[nengo.Ensemble].intercepts = nengo.dists.Choice([0])
            net.config[nengo.Connection].synapse = None
            nengo_dl.configure_settings(stateful=False)
            inp = nengo.Node(np.zeros(self.x * self.y * self.n_time_bins))
            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=32, kernel_size=3))(inp, shape_in=(self.x, self.y, 1))
            x = nengo_dl.Layer(self.neuron_type)(x)
            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=64, strides=2, kernel_size=3))(x, shape_in=(58, 78, 32))
            x = nengo_dl.Layer(self.neuron_type)(x)
            x = nengo_dl.Layer(tf.keras.layers.Conv2D(filters=128, strides=2, kernel_size=3))(x, shape_in=(28, 38, 64))
            x = nengo_dl.Layer(self.neuron_type)(x)
            out = nengo_dl.Layer(tf.keras.layers.Dense(units=3))(x)
            out_p = nengo.Probe(out, label="out_p")
            out_p_filt = nengo.Probe(out, synapse=0.1, label="out_p_filt")
            return net, inp, out_p, out_p_filt