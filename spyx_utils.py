# implement our SNN in DeepMind's Haiku
import spyx
import haiku as hk
import spyx.nn as snn
# for surrogate loss training.
import optax
from jax_tqdm import scan_tqdm
from tqdm import tqdm
import os
import jax
from jax import numpy as jnp
import jmp # jax mixed-precision
import numpy as np

def mse_spikerate(sparsity=0.25, smoothing=0.0, time_axis=1):
    """Calculate the mean squared error of the mean spike rate. Allows for label smoothing to discourage silencing the other neurons in the readout layer.

    :param sparsity: the percentage of the time you want the neurons to spike
    :param smoothing: [optional] rate at which to smooth labels.
    :return: Mean-Squared-Error loss function on the spike rate that takes SNN output traces and integer index labels.
    """
    def _mse_spikerate(traces, targets):

        t = traces.shape[time_axis]
        logits = jnp.mean(traces, axis=time_axis) # time axis.
        labels = optax.smooth_labels(jax.nn.one_hot(targets, logits.shape[-1]), smoothing)
        return jnp.mean(optax.squared_error(logits, labels * sparsity * t))

    return jax.jit(_mse_spikerate)

def euclidean_distance_loss(axis = 1):

    """Calculate the mean squared Euclidean distance between predictions and targets.

    :param predictions: Predicted values from the model, shape (batch_size, 2).
    :param targets: Actual target values, shape (batch_size, 2).
    :return: Mean squared Euclidean distance.
    """

    def _euclidean_distance_loss(predictions, targets):
        # Calculate the Euclidean distance for each prediction-target pair
        distances = jnp.sqrt(jnp.sum(jnp.square(predictions - targets), axis=axis))
        # Return the mean of these distances
        return jnp.mean(distances)
    return jax.jit(_euclidean_distance_loss)

# def mse_spikerate(sparsity=0.25, smoothing=0.0):
#     """Calculate the mean squared error of the mean spike rate. Allows for label smoothing to discourage silencing the other neurons in the readout layer.

#     :param sparsity: the percentage of the time you want the neurons to spike
#     :param smoothing: [optional] rate at which to smooth labels.
#     :return: Mean-Squared-Error loss function on the spike rate that takes SNN output traces and integer index labels.
#     """
#     def _mse_spikerate(traces, targets):
#         # Here we assume traces is already of shape (batch_size, num_classes)
#         logits = traces
#         # One-hot encode targets and apply smoothing
#         labels = optax.smooth_labels(jax.nn.one_hot(targets, logits.shape[-1]), smoothing)
#         # Scale labels with sparsity and number of time steps
#         return jnp.mean(optax.squared_error(logits, labels * sparsity * traces.shape[0]))

    return jax.jit(_mse_spikerate)

def sigmoid(x):
    """
    Computes the sigmoid activation function element-wise.

    :param x: Input array.
    :return: Element-wise sigmoid of x.
    """
    return 1 / (1 + jnp.exp(-x))

def shd_snn(x):
    batch = x.shape[0]
    timestep = x.shape[1]
    x = np.reshape(x, (batch, timestep, -1))
    # We can use batch apply to apply the first linear layer across all of the input data
    x = hk.BatchApply(hk.Linear(64, with_bias=False))(x)

    # Haiku has the ability to stack multiple layers/recurrent modules as one entity
    core = hk.DeepRNN([
        snn.LIF((64,), activation=spyx.axn.triangular()), #LIF neuron layer with triangular activation
        hk.Linear(64, with_bias=False),
        snn.LIF((64,), activation=spyx.axn.triangular()),
        hk.Linear(2, with_bias=False),
        snn.LI((2,)) # Non-spiking final layer
    ])

    # This takes our SNN core and computes it across the input data.
    spikes, V = hk.dynamic_unroll(core, x, core.initial_state(x.shape[0]), time_major=False, unroll=32) # unroll our model.
    # # Combine the time dimension to match the target shape (e.g., using mean)

    return spikes, V

def gd(SNN, params, dl, epochs=300, schedule=3e-4):

    # We use optax for our optimizer.
    opt = optax.lion(learning_rate=schedule)

    # Loss = spyx.fn.integral_crossentropy()
    Loss = euclidean_distance_loss(axis=1)
    # Acc = spyx.fn.integral_accuracy()

    # create and initialize the optimizer
    opt_state = opt.init(params)
    grad_params = params

    # define and compile our eval function that computes the loss for our SNN
    @jax.jit
    def net_eval(weights, events, targets):
        readout = SNN.apply(weights, events)
        traces, V_f = readout
        # spikes = jnp.mean(spikes, axis=1)
        traces = np.reshape(traces, (traces.shape[0], -1))
        return Loss(traces, targets)

    # Use JAX to create a function that calculates the loss and the gradient!
    surrogate_grad = jax.value_and_grad(net_eval)

    rng = jax.random.PRNGKey(0)

    # compile the meat of our training loop for speed
    @jax.jit
    def train_step(state, data):
        # unpack the parameters and optimizer state
        grad_params, opt_state = state
        # unpack the data into x, y
        events, targets = data
        # events = jnp.unpackbits(events, axis=1) # decompress temporal axis
        # compute loss and gradient
        loss, grads = surrogate_grad(grad_params, events, targets)
        # generate updates based on the gradients and optimizer
        updates, opt_state = opt.update(grads, opt_state, grad_params)
        # return the updated parameters
        new_state = [optax.apply_updates(grad_params, updates), opt_state]
        return new_state, loss

    # For validation epochs, do the same as before but compute the
    # accuracy, predictions and losses (no gradients needed)
    # @jax.jit
    # def eval_step(grad_params, data):
    #     # unpack our data
    #     events, targets = data
    #     # # decompress information along temporal axis
    #     # events = jnp.unpackbits(events, axis=1)
    #     # apply the network to the data
    #     readout = SNN.apply(grad_params, events)
    #     # unpack the final layer outputs and end state of each SNN layer
    #     traces, V_f = readout
    #     # compute accuracy, predictions, and loss
    #     # acc, pred = Acc(traces, targets)
    #     loss = Loss(traces, targets)
    #     # we return the parameters here because of how jax.scan is structured.
    #     # return grad_params, jnp.array([acc, loss])
    #     return grad_params
 
    @jax.jit
    def eval_step(grad_params, data):
        events, targets = data

        # Apply the SNN to compute outputs
        readout = SNN.apply(grad_params, events)
        traces, V_f = readout

        # Calculate mean squared error (MSE)
        mse = jnp.mean((traces - targets) ** 2)

        # Alternatively, calculate root mean squared error (RMSE)
        rmse = jnp.sqrt(jnp.mean((traces - targets) ** 2))

        # Calculate R² score (Coefficient of Determination)
        mean_targets = jnp.mean(targets, axis=0)
        ss_tot = jnp.sum((targets - mean_targets) ** 2)
        ss_res = jnp.sum((traces - targets) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        # Return as a tuple (grad_params, metrics)
        return grad_params, jnp.array([mse, rmse, r2_score])


    val_data = dl.val_epoch()

    # Here's the start of our training loop!
    @scan_tqdm(epochs)
    def epoch(epoch_state, epoch_num):
        curr_params, curr_opt_state = epoch_state

        shuffle_rng = jax.random.fold_in(rng, epoch_num)
        train_data = dl.train_epoch(shuffle_rng)

        # train epoch
        end_state, train_loss = jax.lax.scan(
            train_step,# our function which computes and updates gradients for one batch
            [curr_params, curr_opt_state], # initialize with parameters and optimizer state of current epoch
            train_data,# pass the newly shuffled training data
            train_data.obs.shape[0]# this corresponds to the number of training batches
        )

        new_params, _ = end_state

        # val epoch
        _, val_metrics = jax.lax.scan(
            eval_step,# func
            new_params,# init
            val_data,# xs
            val_data.obs.shape[0]# len
        )


        return end_state, jnp.concatenate([jnp.expand_dims(jnp.mean(train_loss),0), jnp.mean(val_metrics, axis=0)])
    # end epoch

    # epoch loop
    final_state, metrics = jax.lax.scan(
        epoch,
        [grad_params, opt_state], # metric arrays
        jnp.arange(epochs), #
        epochs # len of loop
    )

    final_params, final_optimizer_state = final_state


    # return our final, optimized network.
    return final_params, metrics

def test_gd(SNN, params, dl):

    # Define the Mean Squared Error (MSE) loss function
    def mse_loss(predictions, targets):
        return jnp.mean((predictions - targets) ** 2)

    @jax.jit
    def test_step(params, data):
        events, targets = data
        readout = SNN.apply(params, events)
        traces, V_f = readout

        # Calculate MSE between predictions (traces) and targets
        mse = mse_loss(traces, targets)

        # Calculate R² score (Coefficient of Determination)
        mean_targets = jnp.mean(targets, axis=0)
        ss_tot = jnp.sum((targets - mean_targets) ** 2)
        ss_res = jnp.sum((traces - targets) ** 2)
        r2_score = 1 - (ss_res / ss_tot)

        return params, [mse, r2_score, traces, targets]

    test_data = dl.test_epoch()

    _, test_metrics = jax.lax.scan(
            test_step,   # Function to apply
            params,      # Initial parameters
            test_data,   # Input data (events, targets)
            test_data.obs.shape[0]  # Number of batches
    )

    # Compute mean metrics over the test set
    mse = jnp.mean(test_metrics[0])
    r2_score = jnp.mean(test_metrics[1])

    # Flatten predictions and targets for further analysis if needed
    preds = jnp.concatenate(test_metrics[2], axis=0)
    tgts = jnp.concatenate(test_metrics[3], axis=0)

    return mse, r2_score, preds, tgts
