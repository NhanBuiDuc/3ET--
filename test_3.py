import argparse
import json
import os
# import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import train_nengo_epoch, train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
from nengo_model import SpikingNet
import nengo_dl
import tensorflow as tf
def train(model, sim, out_p, train_loader, val_loader, args):
    best_val_loss = float("inf")
    patience_counter = 0  # Counter to keep track of patient epochs

    # Training loop
    for epoch in range(args.num_epochs):
        model, train_loss, metrics = train_nengo_epoch(model, sim, out_p, train_loader, args)
        print("train_loss: ", train_loss)
        print("train_p_acc_all: ", metrics['tr_p_acc_all'])
        print("train_p_error_all: ", metrics['tr_p_error_all'])

        if args.val_interval > 0 and (epoch + 1) % args.val_interval == 0:
            val_loss, val_metrics = validate_epoch(model, val_loader, args)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0  # Reset patience counter
                # Save the new best model
                # torch.save(model.state_dict(), os.path.join("checkpoints", f"model_best_ep{epoch}_val_loss_{val_loss:.4f}.pth"))
            else:
                patience_counter += 1

            # # Step the scheduler
            # scheduler.step(val_loss)

            print(f"[Validation] at Epoch {epoch+1}/{args.num_epochs}: Val Loss: {val_loss:.4f}")
            print("val_p_acc_all: ", val_metrics['val_p_acc_all'])
            print("val_p_error_all: ", val_metrics['val_p_error_all'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.num_epochs}: Train Loss: {train_loss:.4f}")

        # Check if training should be stopped due to lack of improvement
        if patience_counter >= 10:
            print("Early stopping due to no improvement in validation loss for 10 consecutive epochs.")
            break

    return model

if __name__ == "__main__":
    config_file = 'sliced_baseline.json'
    with open(os.path.join('./configs', config_file), 'r') as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    device = "cuda"
    
    # Define your model, optimizer, and criterion
    model, inp, out_p, out_p_filt = SpikingNet().build_model()
    minibatch_size = 64
    sim = nengo_dl.Simulator(model, minibatch_size=minibatch_size)
    sim.compile(
        optimizer=tf.optimizers.RMSprop(0.001),
        loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},
    )
    factor = args.spatial_factor  # spatial downsample factor
    temp_subsample_factor = args.temporal_subsample_factor  # downsampling original 100Hz label to 20Hz

    # The original labels are spatially downsampled with 'factor', downsampled to 20Hz, and normalized w.r.t width and height to [0,1]
    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    train_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="train", transform=transforms.Downsample(spatial_factor=factor), target_transform=label_transform)
    val_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="val", transform=transforms.Downsample(spatial_factor=factor), target_transform=label_transform)

    slicing_time_window = args.train_length * int(10000/temp_subsample_factor)  # microseconds
    train_stride_time = int(10000 / temp_subsample_factor * args.train_stride)  # microseconds

    train_slicer = SliceByTimeEventsTargets(slicing_time_window, overlap=slicing_time_window - train_stride_time, seq_length=args.train_length, seq_stride=args.train_stride, include_incomplete=False)
    val_slicer = SliceByTimeEventsTargets(slicing_time_window, overlap=0, seq_length=args.val_length, seq_stride=args.val_stride, include_incomplete=False)

    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000 / temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2), n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
    ])

    train_data = SlicedDataset(train_data_orig, train_slicer, transform=post_slicer_transform, metadata_path=f"./metadata/3et_train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}")
    val_data = SlicedDataset(val_data_orig, val_slicer, transform=post_slicer_transform, metadata_path=f"./metadata/3et_val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}")

    train_data = DiskCachedDataset(train_data, cache_path=f'./cached_dataset/train_tl_{args.train_length}_ts{args.train_stride}_ch{args.n_time_bins}')
    val_data = DiskCachedDataset(val_data, cache_path=f'./cached_dataset/val_vl_{args.val_length}_vs{args.val_stride}_ch{args.n_time_bins}')
    
    train_loader = DataLoader(train_data, batch_size=minibatch_size, shuffle=True, num_workers=int(os.cpu_count()-2), pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=minibatch_size, shuffle=False, num_workers=int(os.cpu_count()-2))
    model = train(model, sim, out_p, train_loader, val_loader, args)