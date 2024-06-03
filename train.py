import argparse
import json
import os

from sympy import false

# import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import train_nengo_epoch, train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from dataset import ThreeETplus_EyetrackingDataset,ThreeETplus_Eyetracking,  ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
from nengo_model import SpikingNet
import nengo_dl
import tensorflow as tf


def p_tolerance_accuracy(y_true, y_pred, tolerance, width_scale, height_scale):
    """
    Custom metric to calculate the pixel-tolerance accuracy.
    
    Args:
    y_true: True labels.
    y_pred: Predicted labels.
    tolerance: Pixel tolerance value.
    width_scale: Width scaling factor.
    height_scale: Height scaling factor.
    
    Returns:
    Pixel tolerance accuracy.
    """
    y_true = y_true[:, -1, :2]  # Take only the last frame's coordinates
    y_pred = y_pred[:, -1, :2]  # Take only the last frame's coordinates

    diff = tf.abs(y_true - y_pred)
    diff = diff * tf.constant([width_scale, height_scale], dtype=tf.float32)
    within_tolerance = tf.reduce_sum(tf.cast(diff <= tolerance, tf.float32), axis=-1)
    
    accuracy = tf.reduce_mean(tf.cast(within_tolerance == 2, tf.float32))
    return accuracy

def p1_accuracy(y_true, y_pred):
    return p_tolerance_accuracy(y_true, y_pred, tolerance=1, width_scale=args.sensor_width * args.spatial_factor, height_scale=args.sensor_height * args.spatial_factor)

def p3_accuracy(y_true, y_pred):
    return p_tolerance_accuracy(y_true, y_pred, tolerance=3, width_scale=args.sensor_width * args.spatial_factor, height_scale=args.sensor_height * args.spatial_factor)

def p5_accuracy(y_true, y_pred):
    return p_tolerance_accuracy(y_true, y_pred, tolerance=5, width_scale=args.sensor_width * args.spatial_factor, height_scale=args.sensor_height * args.spatial_factor)

def p10_accuracy(y_true, y_pred):
    return p_tolerance_accuracy(y_true, y_pred, tolerance=10, width_scale=args.sensor_width * args.spatial_factor, height_scale=args.sensor_height * args.spatial_factor)

def p15_accuracy(y_true, y_pred):
    return p_tolerance_accuracy(y_true, y_pred, tolerance=15, width_scale=args.sensor_width * args.spatial_factor, height_scale=args.sensor_height * args.spatial_factor)

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
    
# if __name__ == "__main__":
#     config_file = 'sliced_baseline.json'
#     with open(os.path.join('./configs', config_file), 'r') as f:
#         config = json.load(f)
#     args = argparse.Namespace(**config)
#     device = "cuda"
#     minibatch_size = 64
#     # Define your model, optimizer, and criterion
#     model, inp, out_p, out_p_filt = SpikingNet().build_model()
#     sim = nengo_dl.Simulator(model, minibatch_size=minibatch_size)

#     factor = args.spatial_factor  # spatial downsample factor
#     temp_subsample_factor = args.temporal_subsample_factor  # downsampling original 100Hz label to 20Hz
#     sim.compile(
#         optimizer=tf.optimizers.RMSprop(0.001),
#         loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},

#     )
#     label_transform = transforms.Compose([
#         ScaleLabel(factor),
#         TemporalSubsample(temp_subsample_factor),
#         NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
#     ])
    # train_data_orig = ThreeETplus_EyetrackingDataset(data_dir=args.data_dir, split="train")
    # val_data_orig = ThreeETplus_EyetrackingDataset(data_dir=args.data_dir, split="test")
if __name__ == "__main__":
    config_file = 'sliced_baseline.json'
    with open(os.path.join('./configs', config_file), 'r') as f:
        config = json.load(f)
    args = argparse.Namespace(**config)
    device = "cuda"
    
    # Define your model, optimizer, and criterion
    model, inp, out_p, out_p_filt = SpikingNet().build_model()
    minibatch_size = 1
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
    train_data_orig = ThreeETplus_EyetrackingDataset(data_dir=args.data_dir, split="train", transform=transforms.Downsample(spatial_factor=factor), target_transform=label_transform, slicer=train_slicer, post_slicer_transform = post_slicer_transform)
    # val_data_orig = ThreeETplus_EyetrackingDataset(data_dir=args.data_dir, split="test", transform=transforms.Downsample(spatial_factor=factor), target_transform=label_transform, slicer=val_slicer, post_slicer_transform = post_slicer_transform)
    train_x = train_data_orig.inputs
    train_y = train_data_orig.targets
    print(len(train_x))
    print(len(train_y))
    # val_x = val_data_orig.inputs
    # val_y = val_data_orig.targets
    # model = train(model, sim, out_p, train_loader, val_loader, args)
    train = True
    if train:
        sim.compile(
            optimizer=tf.optimizers.RMSprop(0.001),
            loss={
                out_p: tf.losses.MeanSquaredError(),
                # out_p_filt: [                    
                #     p1_accuracy,
                #     p3_accuracy,
                #     p5_accuracy,
                #     p10_accuracy,
                #     p15_accuracy,
                #     tf.keras.losses.MeanAbsoluteError(),
                #     tf.keras.losses.MeanSquaredError()
                # ]
                },
            metrics={
                out_p: [
                    p1_accuracy,
                    p3_accuracy,
                    p5_accuracy,
                    p10_accuracy,
                    p15_accuracy,
                    tf.keras.losses.MeanAbsoluteError(),
                    tf.keras.losses.MeanSquaredError()
                ],
                # out_p_filt: [                    
                #     p1_accuracy,
                #     p3_accuracy,
                #     p5_accuracy,
                #     p10_accuracy,
                #     p15_accuracy,
                #     tf.keras.losses.MeanAbsoluteError(),
                #     tf.keras.losses.MeanSquaredError()
                # ]
            }
        )

        sim.fit(train_x, {out_p: train_y}, epochs=200)
        # save the parameters to file
        sim.save_params("./best")
    else:
        # load parameters
        sim.load_params("./best")  
        #Assuming test_images and test_labels are your test data
        # evaluation_results = sim.evaluate(val_x, {out_p_filt: val_y}, verbose=1)
        # print("Val loss:", evaluation_results['loss'])
        # print("P loss:", evaluation_results['out_p_loss'])
        # print("P1 Accuracy:", evaluation_results['out_p_p1_accuracy'])
        # print("P3 Accuracy:", evaluation_results['out_p_p3_accuracy'])
        # print("P5 Accuracy:", evaluation_results['out_p_p5_accuracy'])
        # print("P10 Accuracy:", evaluation_results['out_p_p10_accuracy'])
        # print("P15 Accuracy:", evaluation_results['out_p_p15_accuracy'])
        # print("Mean Absolute Error:", evaluation_results['out_p_mean_absolute_error'])
        # print("Mean Squared Error:", evaluation_results['out_p_mean_squared_error'])
        num_samples = len(train_x)

        # total_loss = 0.0
        # total_p_loss = 0.0
        # total_p1_accuracy = 0.0
        # total_p3_accuracy = 0.0
        # total_p5_accuracy = 0.0
        # total_p10_accuracy = 0.0
        # total_p15_accuracy = 0.0
        
        # total_p1_accuracy_filt = 0.0
        # total_p3_accuracy_filt = 0.0
        # total_p5_accuracy_filt = 0.0
        # total_p10_accuracy_filt = 0.0
        # total_p15_accuracy_filt = 0.0
        # total_mean_absolute_error = 0.0
        # total_mean_squared_error = 0.0

        for i in range(0, num_samples, minibatch_size):

            batch_x = train_x[i:i+minibatch_size]
            batch_y = train_y[i:i+minibatch_size]
            if len(batch_x) >= minibatch_size:
                pred = sim.predict(batch_x)
                out_probe = pred[out_p]
                out_probe_filt = pred[out_p_filt]

                # out_probe[..., 0] *= 640 * factor
                # out_probe[..., 1] *= 480 * factor

                # out_probe_filt[..., 0] *= 640 * factor
                # out_probe_filt[..., 1] *= 480 * factor
                # total_p1_accuracy += p1_accuracy(batch_y, out_p_tensor).numpy()
                # total_p3_accuracy += p3_accuracy(batch_y, out_p_tensor).numpy()
                # total_p5_accuracy += p5_accuracy(batch_y, out_p_tensor).numpy()
                # total_p10_accuracy += p10_accuracy(batch_y, out_p_tensor).numpy()
                # total_p15_accuracy += p15_accuracy(batch_y, out_p_tensor).numpy()

                # total_p1_accuracy_filt += p1_accuracy(batch_y, out_p_filt_tensor).numpy()
                # total_p3_accuracy_filt += p3_accuracy(batch_y, out_p_filt_tensor).numpy()
                # total_p5_accuracy_filt += p5_accuracy(batch_y, out_p_filt_tensor).numpy()
                # total_p10_accuracy_filt += p10_accuracy(batch_y, out_p_filt_tensor).numpy()
                # total_p15_accuracy_filt += p15_accuracy(batch_y, out_p_filt_tensor).numpy()
        # # Calculate averages
        # avg_loss = total_loss / num_samples
        # avg_p_loss = total_p_loss / num_samples
        # avg_p1_accuracy = total_p1_accuracy / num_samples
        # avg_p3_accuracy = total_p3_accuracy / num_samples
        # avg_p5_accuracy = total_p5_accuracy / num_samples
        # avg_p10_accuracy = total_p10_accuracy / num_samples
        # avg_p15_accuracy = total_p15_accuracy / num_samples
        # avg_mean_absolute_error = total_mean_absolute_error / num_samples
        # avg_mean_squared_error = total_mean_squared_error / num_samples

        # # print("Average Val loss:", avg_loss)
        # # print("Average P loss:", avg_p_loss)
        # print("Average P1 Accuracy:", avg_p1_accuracy)
        # print("Average P3 Accuracy:", avg_p3_accuracy)
        # print("Average P5 Accuracy:", avg_p5_accuracy)
        # print("Average P10 Accuracy:", avg_p10_accuracy)
        # print("Average P15 Accuracy:", avg_p15_accuracy)
        # # print("Average Mean Absolute Error:", avg_mean_absolute_error)
        # # print("Average Mean Squared Error:", avg_mean_squared_error)