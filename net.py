import torch
import snntorch as snn
import torch.nn as nn
from snntorch import surrogate
import numpy as np
def create_temporal_embedding(length, device='cuda'):
    center = length // 2
    sigma = length / 10  # Standard deviation, controls the width of the bell curve
    positions = np.arange(length)
    embedding = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
    embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min())
    embedding = torch.tensor(embedding, dtype=torch.float32).to(device)
    embedding = embedding.view(1, length, 1, 1, 1)  # Reshape to allow broadcasting
    return embedding

def apply_temporal_embedding(data, embedding):
    data = data.to(embedding.device)
    weighted_data = data * embedding  # Broadcasting happens here
    return weighted_data

def create_temporal_embedding(length, device='cuda'):
    center = length // 2
    sigma = length / 10  # Standard deviation, controls the width of the bell curve
    positions = np.arange(length)
    embedding = np.exp(-0.5 * ((positions - center) / sigma) ** 2)
    embedding = (embedding - embedding.min()) / (embedding.max() - embedding.min())
    embedding = torch.tensor(embedding, dtype=torch.float32).to(device)
    embedding = embedding.view(1, length, 1, 1, 1)  # Reshape to allow broadcasting
    return embedding

def apply_temporal_embedding(data, embedding):
    data = data.to(embedding.device)
    weighted_data = data * embedding  # Broadcasting happens here
    return weighted_data

class TonicSSNNet(nn.Module):
    def __init__(self, timesteps, hidden, output_size, device):
        super().__init__()

        self.timesteps = timesteps
        self.hidden = hidden
        self.output_size = output_size
        self.device = device
        spike_grad = surrogate.fast_sigmoid()

        # Randomly initialize decay rate and threshold for layer 1
        beta_in = torch.rand(self.hidden)
        thr_in = torch.rand(self.hidden)

        # Layer 1
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.fc_in = nn.Linear(in_features=32 * 60 * 80, out_features=self.hidden)
        self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)

        # Randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(self.hidden)
        thr_hidden = torch.rand(self.hidden)

        # Layer 2
        self.fc_hidden = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)

        # Additional Layer 3
        beta_hidden2 = torch.rand(self.hidden)
        thr_hidden2 = torch.rand(self.hidden)

        self.fc_hidden2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
        self.lif_hidden2 = snn.Leaky(beta=beta_hidden2, threshold=thr_hidden2, learn_beta=True, spike_grad=spike_grad)

        # Randomly initialize decay rate and threshold for output layer
        beta_out = torch.rand(self.output_size)
        thr_out = torch.rand(self.output_size)

        # Output layer: leaky integrator neuron with ReLU activation
        self.fc_out = nn.Linear(in_features=self.hidden, out_features=self.output_size)
        self.li_out = snn.Leaky(beta=beta_out, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")

        # Temporal embedding
        self.temporal_embedding = create_temporal_embedding(timesteps, device=device)

    def forward(self, x):
        # Apply temporal embedding
        x = apply_temporal_embedding(x, self.temporal_embedding)

        # Initialize membrane potentials
        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_3 = self.li_out.init_leaky()
        mem_4 = self.lif_hidden2.init_leaky()

        # Empty list to record outputs
        spk_out_rec = []
        mem_3_rec = []

        # Forward pass over time steps
        for step in range(self.timesteps):
            x_timestep = x[:, step]  # Extract the timestep data from the batch

            # Apply convolutional layers
            x_timestep = torch.relu(self.conv1(x_timestep))
            x_timestep = torch.relu(self.conv2(x_timestep))

            # Flatten the spatial dimensions
            x_timestep = x_timestep.view(x_timestep.size(0), -1)

            cur_in = self.fc_in(x_timestep)
            spk_in, mem_1 = self.lif_in(cur_in, mem_1)

            cur_hidden = self.fc_hidden(spk_in)
            spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)

            cur_hidden2 = self.fc_hidden2(spk_hidden)
            spk_hidden2, mem_4 = self.lif_hidden2(cur_hidden2, mem_4)

            cur_out = self.fc_out(spk_hidden2)
            spk_out, mem_3 = self.li_out(cur_out, mem_3)

            spk_out_rec.append(spk_out)

        # Stack the recorded outputs
        spk_out_rec = torch.stack(spk_out_rec, dim=1)  # Shape: (batch_size, timesteps, output_size)

        return spk_out_rec

class Net(torch.nn.Module):
    """Simple spiking neural network in snntorch."""

    def __init__(self, timesteps, hidden, beta):
        super().__init__()

        self.timesteps = timesteps
        self.hidden = hidden
        self.beta = beta

        # layer 1
        self.fc1 = torch.nn.Linear(in_features=14400, out_features=7700)
        self.rlif1 = snn.RLeaky(beta=self.beta, linear_features=7700)

        # layer 2
        self.fc2 = torch.nn.Linear(in_features=7700, out_features=500)
        self.rlif2 = snn.RLeaky(beta=self.beta, linear_features=500)
        self.fc3 =  torch.nn.Linear(in_features=15000, out_features=5000)
        self.fc4 =  torch.nn.Linear(in_features=5000, out_features=1000)
        self.fc5 =  torch.nn.Linear(in_features=1000, out_features=500)
        self.fc6 =  torch.nn.Linear(in_features=500, out_features=100)
        self.fc7 =  torch.nn.Linear(in_features=100, out_features=10)
        self.fc8 =  torch.nn.Linear(in_features=10, out_features=2)
        self.relu = torch.nn.ReLU()
    def forward(self, x):
        """Forward pass for several time steps."""
        x = x.view(x.shape[0], x.shape[1], -1)
        # Initalize membrane potential
        spk1, mem1 = self.rlif1.init_rleaky()
        spk2, mem2 = self.rlif2.init_rleaky()

        # Empty lists to record outputs
        spk_recording = []

        for step in range(self.timesteps):
            spk1, mem1 = self.rlif1(self.fc1(x), spk1, mem1)
            spk2, mem2 = self.rlif2(self.fc2(spk1), spk2, mem2)
            spk_recording.append(spk2)

        final_output = torch.stack(spk_recording)
        final_output = final_output.view(final_output.shape[0], final_output.shape[1], -1)
        final_output = self.fc3(final_output)
        final_output = self.relu(final_output)
        final_output = self.fc4(final_output)
        final_output = self.relu(final_output)
        final_output = self.fc5(final_output)
        final_output = self.relu(final_output)
        final_output = self.fc6(final_output)
        final_output = self.relu(final_output)
        final_output = self.fc7(final_output)
        final_output = self.relu(final_output)
        final_output = self.fc8(final_output)
        final_output = final_output.view(final_output.shape[1], final_output.shape[0], -1)
        return final_output
class SpikeRegressionNet(torch.nn.Module):
    """Simple spiking neural network for regression using spike counts."""

    def __init__(self, timesteps, hidden, output_size, device):
        super().__init__()

        self.timesteps = timesteps  # Number of time steps to simulate the network
        self.hidden = hidden  # Number of hidden neurons
        self.output_size = output_size  # Size of the output
        self.device = device
        spike_grad = surrogate.fast_sigmoid()  # Surrogate gradient function

        # Randomly initialize decay rate and threshold for layer 1
        beta_in = torch.rand(self.hidden)
        thr_in = torch.rand(self.hidden)

        # Layer 1
        self.fc_in = torch.nn.Linear(in_features=3, out_features=self.hidden)
        self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)

        # Randomly initialize decay rate and threshold for layer 2
        beta_hidden = torch.rand(int(self.hidden))
        thr_hidden = torch.rand(int(self.hidden))

        # Layer 2
        self.fc_hidden = torch.nn.Linear(in_features=self.hidden, out_features=int(self.hidden))
        self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)

        # Additional Layer 3
        beta_hidden2 = torch.rand(int(self.hidden))
        thr_hidden2 = torch.rand(int(self.hidden))

        self.fc_hidden2 = torch.nn.Linear(in_features=self.hidden, out_features=int(self.hidden))
        self.lif_hidden2 = snn.Leaky(beta=beta_hidden2, threshold=thr_hidden2, learn_beta=True, spike_grad=spike_grad)

        # Randomly initialize decay rate and threshold for output layer
        beta_out = torch.rand(int(self.output_size))
        thr_out = torch.rand(int(self.output_size))

        # Output layer: leaky integrator neuron with ReLU activation
        self.fc_hidden_2 = torch.nn.Linear(in_features=int(self.hidden), out_features=int(self.hidden))
        self.fc_out = torch.nn.Linear(in_features=int(self.hidden), out_features=int(output_size))
        self.fc_pred = torch.nn.Linear(in_features=int(self.timesteps) * 4, out_features=int(output_size))
        self.li_out = snn.Leaky(beta=beta_out, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")
        self.relu = nn.ReLU()

        # Temporal embedding
        self.temporal_embedding = create_temporal_embedding(timesteps, device=device)

    def forward(self, x):
        """Forward pass for several time steps."""

        # Apply temporal embedding
        x = apply_temporal_embedding(x, self.temporal_embedding)

        # Initialize membrane potential
        mem_1 = self.lif_in.init_leaky()
        mem_2 = self.lif_hidden.init_leaky()
        mem_3 = self.li_out.init_leaky()
        mem_4 = self.lif_hidden2.init_leaky()

        # Empty list to record outputs
        mem_3_rec = []
        spk_out_rec = []
        if self.timesteps > 1:
            # Forward pass over time steps
            for step in range(int(self.timesteps)):
                x_timestep = x[step]

                cur_in = self.fc_in(x_timestep)
                spk_in, mem_1 = self.lif_in(cur_in, mem_1)

                cur_hidden = self.fc_hidden(spk_in)
                spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)

                cur_hidden2 = self.fc_hidden2(spk_hidden)
                spk_hidden2, mem_4 = self.lif_hidden2(cur_hidden2, mem_4)

                cur_out = self.fc_out(spk_hidden2)
                spk_out, mem_3 = self.li_out(cur_out, mem_3)

                spk_out_rec.append(spk_out)
                mem_3_rec.append(mem_3)

            # Convert lists to tensors
            spk_out_rec = torch.stack(spk_out_rec, dim=0)
            mem_3_rec = torch.stack(mem_3_rec, dim=0)

            # Concatenate the spikes and membrane potentials along the feature dimension
            out = torch.cat([spk_out_rec, mem_3_rec], dim=-1)
            out = out.view(out.shape[1], out.shape[0] * out.shape[2])
            # Apply the final fully connected layer
            pred = self.fc_pred(out)

            return pred
        else:
                cur_in = self.fc_in(x)
                spk_in, mem_1 = self.lif_in(cur_in, mem_1)

                cur_hidden = self.fc_hidden(spk_in)
                spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)

                cur_hidden2 = self.fc_hidden2(spk_hidden)
                spk_hidden2, mem_4 = self.lif_hidden2(cur_hidden2, mem_4)

                cur_out = self.fc_hidden_2(spk_hidden2)
                spk_out, mem_3 = self.li_out(cur_out, mem_3)
                pred = self.fc_out(spk_out)
                pred = pred.squeeze(0)
                return pred
# class TonicSSNNet(nn.Module):
#     def __init__(self, timesteps, hidden, output_size, device):
#         super().__init__()

#         self.timesteps = timesteps
#         self.hidden = hidden
#         self.output_size = output_size
#         self.device = device
#         spike_grad = surrogate.fast_sigmoid()

#         # Randomly initialize decay rate and threshold for layer 1
#         beta_in = torch.rand(self.hidden)
#         thr_in = torch.rand(self.hidden)

#         # Layer 1
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
#         self.fc_in = nn.Linear(in_features=32 * 60 * 80, out_features=self.hidden)
#         self.lif_in = snn.Leaky(beta=beta_in, threshold=thr_in, learn_beta=True, spike_grad=spike_grad)

#         # Randomly initialize decay rate and threshold for layer 2
#         beta_hidden = torch.rand(self.hidden)
#         thr_hidden = torch.rand(self.hidden)

#         # Layer 2
#         self.fc_hidden = nn.Linear(in_features=self.hidden, out_features=self.hidden)
#         self.lif_hidden = snn.Leaky(beta=beta_hidden, threshold=thr_hidden, learn_beta=True, spike_grad=spike_grad)

#         # Additional Layer 3
#         beta_hidden2 = torch.rand(self.hidden)
#         thr_hidden2 = torch.rand(self.hidden)

#         self.fc_hidden2 = nn.Linear(in_features=self.hidden, out_features=self.hidden)
#         self.lif_hidden2 = snn.Leaky(beta=beta_hidden2, threshold=thr_hidden2, learn_beta=True, spike_grad=spike_grad)

#         # Randomly initialize decay rate and threshold for output layer
#         beta_out = torch.rand(self.output_size)
#         thr_out = torch.rand(self.output_size)

#         # Output layer: leaky integrator neuron with ReLU activation
#         self.fc_out = nn.Linear(in_features=self.hidden, out_features=self.output_size)
#         self.li_out = snn.Leaky(beta=beta_out, learn_beta=True, spike_grad=spike_grad, reset_mechanism="none")

#         # Temporal embedding
#         self.temporal_embedding = create_temporal_embedding(timesteps, device=device)

#     def forward(self, x):
#         # Apply temporal embedding
#         x = apply_temporal_embedding(x, self.temporal_embedding)

#         # Initialize membrane potentials
#         mem_1 = self.lif_in.init_leaky()
#         mem_2 = self.lif_hidden.init_leaky()
#         mem_3 = self.li_out.init_leaky()
#         mem_4 = self.lif_hidden2.init_leaky()

#         # Empty list to record outputs
#         spk_out_rec = []
#         mem_3_rec = []

#         # Forward pass over time steps
#         for step in range(self.timesteps):
#             x_timestep = x[:, step]  # Extract the timestep data from the batch

#             # Apply convolutional layers
#             x_timestep = torch.relu(self.conv1(x_timestep))
#             x_timestep = torch.relu(self.conv2(x_timestep))

#             # Flatten the spatial dimensions
#             x_timestep = x_timestep.view(x_timestep.size(0), -1)

#             cur_in = self.fc_in(x_timestep)
#             spk_in, mem_1 = self.lif_in(cur_in, mem_1)

#             cur_hidden = self.fc_hidden(spk_in)
#             spk_hidden, mem_2 = self.lif_hidden(cur_hidden, mem_2)

#             cur_hidden2 = self.fc_hidden2(spk_hidden)
#             spk_hidden2, mem_4 = self.lif_hidden2(cur_hidden2, mem_4)

#             cur_out = self.fc_out(spk_hidden2)
#             spk_out, mem_3 = self.li_out(cur_out, mem_3)

#             spk_out_rec.append(spk_out)
#             mem_3_rec.append(mem_3)

#         # Stack and process recorded outputs
#         spk_out_rec = torch.stack(spk_out_rec, dim=0)
#         mem_3_rec = torch.stack(mem_3_rec, dim=0)

#         out = torch.cat([spk_out_rec, mem_3_rec], dim=-1)
#         out = out.view(out.shape[1], out.shape[0] * out.shape[2])
#         pred = self.fc_pred(out)

#         return pred