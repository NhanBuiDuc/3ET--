import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

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
    def __init__(self, name: "Conv"):
        super().__init__(name=name)
        self.conv1 = hk.Conv2d(
            output_channels = 8, 
            kernel_shape = (3,3), 
            stride=1, 
            rate=1, 
            padding='SAME', 
            with_bias=True, 
            w_init=None, 
            b_init=None, 
            mask=None, 
            name=None
        )
        self.conv2 = hk.Conv2d(
            output_channels = 16, 
            kernel_shape = (3,3), 
            stride=1, 
            rate=1, 
            padding='SAME', 
            with_bias=True, 
            w_init=None, 
            b_init=None, 
            mask=None, 
            name=None
        )
        self.conv3 = hk.Conv2d(
            output_channels = 32, 
            kernel_shape = (3,3), 
            stride=1, 
            rate=1, 
            padding='SAME', 
            with_bias=True, 
            w_init=None, 
            b_init=None, 
            mask=None, 
            name=None
        )
    def __call__(self, x):
        feat_1 = self.conv1(x)
        feat_2 = self.conv1(feat_1)
        feat_3 = self.conv1(feat_2)
        return feat_3

class Model(hk.Module):
  def __init__(self, output_size=2, name='JaxModel'):
    super().__init__(name=name)

    self._conv_head_1 = ConvolutionHead()
    self._internal_linear_2 = MyLinear1(output_size=output_size, name='old_linear')
  def __call__(self, x):
    return self._internal_linear_2(self._conv_head_1(x))

def forward(x):
  module = Model()
  return module(x)
