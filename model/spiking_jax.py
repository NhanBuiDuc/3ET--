import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import spyx
import spyx.nn as snn
class MyLinear1(hk.Module):

  def __init__(self, output_size, name=None):
    super().__init__(name=name)
    self.output_size = output_size

  def __call__(self, x):
    j, k = x.shape[-1], self.output_size
    w_init = hk.initializers.TruncatedNormal(1. / np.sqrt(j))
    w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
    b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.ones)
    return jnp.dot(x, w) + b

class ConvolutionHead(hk.Module):
    def __init__(self, name = "Conv"):
        super().__init__(name=name)
        self.conv1 = hk.Conv2D(
            output_channels = 8, 
            kernel_shape = 3, 
            stride=1, 
            rate=1, 
            padding='VALID', 
            with_bias=True, 
            w_init=hk.initializers.RandomNormal(stddev=1.0, mean=0.0), 
            b_init=hk.initializers.RandomNormal(stddev=1.0, mean=0.0), 
            mask=None, 
            name=None
        )
        self.conv2 = hk.Conv2D(
            output_channels = 16, 
            kernel_shape = 3, 
            stride=1, 
            rate=1, 
            padding='VALID', 
            with_bias=True, 
            w_init=hk.initializers.RandomNormal(stddev=1.0, mean=0.0), 
            b_init=hk.initializers.RandomNormal(stddev=1.0, mean=0.0), 
            mask=None, 
            name=None
        )
        self.conv3 = hk.Conv2D(
            output_channels = 32, 
            kernel_shape = 3, 
            stride=1, 
            rate=1, 
            padding='VALID', 
            with_bias=True, 
            w_init=hk.initializers.RandomNormal(stddev=1.0, mean=0.0), 
            b_init=hk.initializers.RandomNormal(stddev=1.0, mean=0.0), 
            mask=None, 
            name=None
        )

    def __call__(self, x):
        x = jnp.reshape(x, (x.shape[0], x.shape[2], x.shape[3], x.shape[1]))
        feat_1 = self.conv1(x)
        feat_2 = self.conv2(feat_1)
        feat_3 = self.conv3(feat_2)
        return feat_3

class Model(hk.Module):
  def __init__(self, output_size=2, name='JaxModel'):
    super().__init__(name=name)

    self._conv_head_1 = ConvolutionHead(name = "Conv")
    self.avg_pooling = hk.AvgPool(window_shape=8, strides=2, padding="VALID", channel_axis=-1, name=None)
    self._internal_linear_1 = hk.Linear(
      output_size=32,
      with_bias = True,
      w_init = hk.initializers.RandomNormal(stddev=1.0, mean=0.0),
      b_init = hk.initializers.RandomNormal(stddev=1.0, mean=0.0),
      name='in_linear')
    self.RNNcore_1 = hk.DeepRNN([
        snn.LIF((32,), activation=spyx.axn.triangular()), #LIF neuron layer with triangular activation
        hk.Linear(32, with_bias=True),
        snn.LIF((32,), activation=spyx.axn.triangular()), #LIF neuron layer with triangular activation
        hk.Linear(32, with_bias=True),
        snn.LIF((32,), activation=spyx.axn.triangular()), #LIF neuron layer with triangular activation
        hk.Linear(32, with_bias=True),
        snn.LI((32,)) # Non-spiking final layer
    ])
    self._internal_linear_2 = hk.Linear(
      output_size=output_size,
      w_init = hk.initializers.RandomNormal(stddev=1.0, mean=0.0),
      b_init = hk.initializers.RandomNormal(stddev=1.0, mean=0.0),
      name='out_linear')
      
  def __call__(self, x):
    timestep = x.shape[0]
    channel = x.shape[3]
    x = self._conv_head_1(x)
    x = self.avg_pooling(x)
    x = jnp.reshape(x, (timestep, -1))
    x = self._internal_linear_1(x)
    x = jnp.reshape(x, (x.shape[1], x.shape[0]))
    # This takes our SNN core and computes it across the input data.
    spikes, V = hk.static_unroll(self.RNNcore_1, x, self.RNNcore_1.initial_state(x.shape[0]), time_major=False) # unroll our model.
    spikes = jnp.reshape(spikes, (timestep, -1))
    return self._internal_linear_2(spikes), V

def forward(x):
  module = Model()
  return module(x)
