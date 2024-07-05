import argparse
import json
import os

from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["CUDA_VISIBLE_DEVICES"] = "1, 3"
# import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import train_nengo_epoch, train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from dataset import ThreeETplus_EyetrackingNumpyDataset, ThreeETplus_EyetrackingJaxNumpyDataset, ThreeETplus_Eyetracking,  ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets, EventSlicesToMap
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
from nengo_model import SpikingNet, TestNet, LMU, LMUConv
import nengo_dl
import tensorflow as tf
import numpy as np
import spyx
import spyx.nn as snn

# JAX imports
import os
import jax
from jax import numpy as jnp
import jmp # jax mixed-precision
import numpy as np

from jax_tqdm import scan_tqdm
from tqdm import tqdm

# implement our SNN in DeepMind's Haiku
import haiku as hk
from spyx_utils import shd_snn, gd, test_gd, full_gd
from model.spiking_jax import forward
# for surrogate loss training.
import optax
from dataloader import DataLoader, SplitDataLoader

def print_jax(x):
    jax.debug.print("{x}", x=x)
    return x


if __name__ == "__main__":
    config_file = 'sliced_baseline.json'
    with open(os.path.join('./configs', config_file), 'r') as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    device = "/gpu"
    policy = jmp.get_policy('half')


    hk.mixed_precision.set_policy(hk.Linear, policy)
    hk.mixed_precision.set_policy(snn.LIF, policy)
    hk.mixed_precision.set_policy(snn.LI, policy)

    lr = args.lr
    factor = args.spatial_factor  # spatial downsample factor
    temp_subsample_factor = args.temporal_subsample_factor  # downsampling original 100Hz label to 20Hz

    # The original labels are spatially downsampled with 'factor', downsampled to 20Hz, and normalized w.r.t width and height to [0,1]
    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])
    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000 / temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2), n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
    ])

    slicing_time_window = args.train_length * int(10000/temp_subsample_factor)  # microseconds
    train_stride_time = int(10000 / temp_subsample_factor * args.train_stride)  # microseconds

    train_slicer = SliceByTimeEventsTargets(
        slicing_time_window, 
        overlap=slicing_time_window - train_stride_time, 
        seq_length=args.train_length, 
        seq_stride=args.train_stride, 
        include_incomplete=False
    )
    val_slicer = SliceByTimeEventsTargets(
        slicing_time_window, 
        overlap=0, 
        seq_length=args.val_length, 
        seq_stride=args.val_stride, 
        include_incomplete=False
    )
    train_data_orig = ThreeETplus_EyetrackingJaxNumpyDataset(
        data_dir=args.data_dir, 
        split="train", 
        transform=transforms.Downsample(spatial_factor=factor), 
        target_transform=label_transform, 
        slicer=train_slicer, 
        post_slicer_transform = post_slicer_transform, 
        device = device,
        cache = True,
        cache_dir = "./cache")
    # val_data_orig = ThreeETplus_EyetrackingDataset(data_dir=args.data_dir, split="test", transform=transforms.Downsample(spatial_factor=factor), target_transform=label_transform, slicer=val_slicer, post_slicer_transform = post_slicer_transform)
    
    train_x = train_data_orig.train_x
    train_y = train_data_orig.train_y
    val_x = train_data_orig.val_x
    val_y = train_data_orig.val_y
    test_x = train_data_orig.test_x
    test_y = train_data_orig.test_y

    isTrain = False
    
    # Create a random key
    key = jax.random.PRNGKey(0)
    # Since there's nothing stochastic about the network, we can avoid using an RNG as a param!
    SNN = hk.without_apply_rng(hk.transform(forward))
    params = SNN.init(rng=key, x=train_x[0])
    split = False
    if split:
        dl = SplitDataLoader(train_x, train_y, val_x, val_y, test_x, test_y, batch_size=64)
        grad_params, metrics = gd(SNN, params, dl, epochs=300) # this takes a minute or two to compile on Colab because of weak CPU compute.
        print("grad_params")
        print_jax(grad_params)
        print("metrics")
        print_jax(metrics)
        mse, r2_score, preds, tgts = test_gd(SNN, grad_params, dl)
        print("mse: ")
        print_jax(mse)
        print("pred: ")
        print_jax(preds)
        print("targets: ")
        print_jax(tgts)
    else:
        # Concatenate the arrays to form full_x and full_y
        full_x = jnp.concatenate([train_x, val_x, test_x], axis=0)
        full_y = jnp.concatenate([train_y, val_y, test_y], axis=0)
        dl = DataLoader(full_x, full_y, batch_size = 64)
        grad_params, metrics = full_gd(SNN, params, dl, epochs=50) # this takes a minute or two to compile on Colab because of weak CPU compute.
        print("grad_params")
        print_jax(grad_params)

        print("train loss")
        print_jax(metrics)