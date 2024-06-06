from sklearn.model_selection import learning_curve
import nengo
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from nengo.utils.filter_design import cont2discrete
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
class SigmoidNeuronType(nengo.neurons.NeuronType):
    """Custom neuron type with sigmoid activation function."""

    def __init__(self, tau_ref=0.0025):
        self.tau_ref = tau_ref

    def gain_bias(self, max_rates, intercepts):
        """Compute gain and bias."""
        return np.ones_like(max_rates), -intercepts

    def step(self, dt, J, output):
        """Implement the sigmoid nonlinearity."""
        output[...] = 1 / (1 + np.exp(-J / self.tau_ref))

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

def sigmoid_activation(time, data):
    return 1 / (1 + np.exp(-data))
        
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

                n_filters=128,

                input_shape=self.input_shape,

                kernel_size=(3, 3),

                padding="same"

            )
            conv2_transform = nengo.Convolution(

                n_filters=128,

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

                n_neurons = np.prod(conv1_transform.output_shape.shape), dimensions = 1000, neuron_type = nengo.LIF(),
            )
            conv2_feat = nengo.Ensemble(

                n_neurons = np.prod(conv2_transform.output_shape.shape), dimensions = 1000, neuron_type = nengo.LIF(),
            )
            conv3_feat = nengo.Ensemble(

                n_neurons = np.prod(conv3_transform.output_shape.shape), dimensions = 1000, neuron_type = nengo.LIF(),
            )

            nengo.Connection(pre = inp, post = conv1_feat.neurons, synapse = 0.01, transform=conv1_transform)
            nengo.Connection(pre = conv1_feat.neurons, post = conv2_feat.neurons, synapse = 0.01, transform=conv2_transform)
            nengo.Connection(pre = conv2_feat.neurons, post = conv3_feat.neurons, synapse = 0.01, transform=conv3_transform)
            # nengo.Connection(conv3_feat, ens_1.neurons, synapse=0.01, transform=nengo_dl.dists.Glorot())
            # nengo.Connection(ens_1.neurons, ens_2.neurons, synapse=0.01, transform=nengo_dl.dists.Glorot())
            # nengo.Connection(ens_2.neurons, ens_3.neurons, synapse=0.01, transform=nengo_dl.dists.Glorot())
            out = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid))(conv3_feat)
            # out = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid))(conv3_feat)
            out_p = nengo.Probe(out, label="out_p")
            out_p_filt = nengo.Probe(out, synapse=0.01, label="out_p_filt")
            return model, inp, out_p, out_p_filt
        
class LMUCell(nengo.Network):
    def __init__(self, units, order, theta, input_d, **kwargs):
        super().__init__(**kwargs)
        # compute the A and B matrices according to the LMU's mathematical derivation
        # (see the paper for details)
        Q = np.arange(order, dtype=np.float64)
        R = (2 * Q + 1)[:, None] / theta
        j, i = np.meshgrid(Q, Q)

        A = np.where(i < j, -1, (-1.0) ** (i - j + 1)) * R
        B = (-1.0) ** Q[:, None] * R
        C = np.ones((1, order))
        D = np.zeros((1,))

        A, B, _, _, _ = cont2discrete((A, B, C, D), dt=1.0, method="zoh")

        with self:
            nengo_dl.configure_settings(trainable=None)

            # create objects corresponding to the x/u/m/h variables in the above diagram
            self.x = nengo.Node(size_in=input_d)
            self.u = nengo.Node(size_in=1)
            self.m = nengo.Node(size_in=order)
            self.h = nengo_dl.TensorNode(tf.nn.tanh, shape_in=(units,), pass_time=False)

            # compute u_t from the above diagram. we have removed e_h and e_m as they
            # are not needed in this task.
            nengo.Connection(
                self.x, self.u, transform=np.ones((1, input_d)), synapse=None
            )

            # compute m_t
            # in this implementation we'll make A and B non-trainable, but they
            # could also be optimized in the same way as the other parameters.
            # note that setting synapse=0 (versus synapse=None) adds a one-timestep
            # delay, so we can think of any connections with synapse=0 as representing
            # value_{t-1}.
            conn_A = nengo.Connection(self.m, self.m, transform=A, synapse=0)
            self.config[conn_A].trainable = False
            conn_B = nengo.Connection(self.u, self.m, transform=B, synapse=None)
            self.config[conn_B].trainable = False

            # compute h_t
            nengo.Connection(
                self.x, self.h, transform=nengo_dl.dists.Glorot(), synapse=None
            )
            nengo.Connection(
                self.h, self.h, transform=nengo_dl.dists.Glorot(), synapse=0
            )
            nengo.Connection(
                self.m,
                self.h,
                transform=nengo_dl.dists.Glorot(),
                synapse=None,
            )
class LMU():
    def __init__(self):
        self.neuron_type = nengo.LIF(amplitude=0.01)
        self.input_shape = (60, 80)
    def build_model(self):
        with nengo.Network(seed=42) as model:
            # remove some unnecessary features to speed up the training
            nengo_dl.configure_settings(
                trainable=True,
                stateful=True,
                keep_history=True,
            )

            # input node
            inp = nengo.Node(np.zeros(self.input_shape[0] * self.input_shape[1]))

            # lmu cell
            lmu = LMUCell(
                units=212,
                order=256,
                theta=self.input_shape[0],
                input_d=(self.input_shape[0] * self.input_shape[1]),
            )
            conn = nengo.Connection(inp, lmu.x, synapse=None)
            model.config[conn].trainable = True

            out = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid))(lmu.x)

            # record output. note that we set keep_history=False above, so this will
            # only record the output on the last timestep (which is all we need
            # on this task)
            out_p = nengo.Probe(out)
            out_p_filt = nengo.Probe(out, synapse = 0.001)
            self.out = out_p
            self.out_p_filt = out_p_filt
            return model, inp, out_p, out_p_filt
class LMUConv():
    def __init__(self):
        self.neuron_type = nengo.LIF(amplitude=0.01)
        self.input_shape = (1, 60, 80)
    def build_model(self):
        with nengo.Network(seed=42) as model:
            # remove some unnecessary features to speed up the training
            nengo_dl.configure_settings(
                trainable=True,
                stateful=True,
                keep_history=True,
            )
            inp = nengo.Node(size_in=0, output = np.zeros(self.input_shape[1] * self.input_shape[2]))
            input_ens = nengo.Ensemble(

                n_neurons = np.prod(self.input_shape[1] * self.input_shape[2]), dimensions = 1, neuron_type = nengo.LIF(),
            )

            nengo.Connection(pre = inp, post = input_ens.neurons, synapse = 0.001)
            print(np.prod(self.input_shape))
            print("Output: ", inp.size_out)
            
            conv1_transform = nengo.Convolution(

                n_filters=256,

                input_shape=self.input_shape,

                kernel_size=(3, 3),

                padding="same"

            )
            conv2_transform = nengo.Convolution(

                n_filters=32,

                input_shape=self.input_shape,

                kernel_size=(4, 4),

                padding="same"

            )
            conv3_transform = nengo.Convolution(

                n_filters=8,

                input_shape=self.input_shape,

                kernel_size=(7, 7),

                padding="same"

            )
            print(conv1_transform.output_shape)
            # Define convolutional layer

            conv1_feat = nengo.Ensemble(

                n_neurons = np.prod(conv1_transform.output_shape.shape), dimensions = 1000, neuron_type = nengo.LIF(),
            )
            conv2_feat = nengo.Ensemble(

                n_neurons = np.prod(conv2_transform.output_shape.shape), dimensions = 1000, neuron_type = nengo.LIF(),
            )
            conv3_feat = nengo.Ensemble(

                n_neurons = np.prod(conv3_transform.output_shape.shape), dimensions = 1000, neuron_type = nengo.LIF(),
            )

            nengo.Connection(pre = input_ens.neurons, post = conv1_feat.neurons, synapse = 0.01, transform=conv1_transform)
            nengo.Connection(pre = input_ens.neurons, post = conv2_feat.neurons, synapse = 0.01, transform=conv2_transform)
            nengo.Connection(pre = input_ens.neurons, post = conv3_feat.neurons, synapse = 0.01, transform=conv3_transform)

            out_1 = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid))(conv1_feat)
            out_2 = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid))(conv2_feat)
            out_3 = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid))(conv3_feat)
            # Concatenate the outputs along the last dimension (axis=-1)
            concatenated_out = tf.keras.layers.concatenate([out_1, out_2, out_3], axis=-1)
            # Define the output layer
            out = nengo_dl.Layer(tf.keras.layers.Dense(units=3, activation=tf.nn.sigmoid))(concatenated_out)
            # record output. note that we set keep_history=False above, so this will
            # only record the output on the last timestep (which is all we need
            # on this task)
            out_p = nengo.Probe(out)
            out_p_filt = nengo.Probe(out, synapse = 0.001)
            self.out = out_p
            self.out_p_filt = out_p_filt
            return model, inp, out_p, out_p_filt