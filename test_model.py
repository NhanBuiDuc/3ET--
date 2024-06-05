import nengo

import numpy as np

from nengo.utils.numpy import rmse



# Define input signal

input_shape = (60, 80, 1)  # Assuming grayscale images with size 28x28

n_filters = 32

kernel_size = (3, 3)

# Input callback function
def inp_cllbck(t, data):
    return np.zeros(np.prod(input_shape))
    
    
with nengo.Network() as model:

    # Define input node representing the input image

    inp = nengo.Node(size_in=1, output = inp_cllbck)
    print(np.prod(input_shape))
    print("Output: ", inp.size_out)
    
    conv_transform = nengo.Convolution(

        n_filters=n_filters,

        input_shape=input_shape,

        kernel_size=kernel_size,

        padding="same"

    )
    print(conv_transform.output_shape)
    # Define convolutional layer

    conv1_feat = nengo.Ensemble(

        n_neurons = 1000, dimensions = np.prod(conv_transform.output_shape.shape), neuron_type = nengo.SpikingRectifiedLinear()
    )
    nengo.Connection(pre = inp, post = conv1_feat, synapse = 0.001, transform=conv_transform, learning_rule_type = nengo.Pes)

  