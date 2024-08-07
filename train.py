import argparse
import json
import os

from sklearn.model_selection import train_test_split
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 2, 4, 6"
# import torch
# import torch.nn as nn
# import torch.optim as optim
from torch.utils.data import DataLoader
from utils.training_utils import train_nengo_epoch, train_epoch, validate_epoch, top_k_checkpoints
from utils.metrics import weighted_MSELoss
from dataset import ThreeETplus_EyetrackingDataset, ThreeETplus_Eyetracking,  ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets, EventSlicesToMap
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
from nengo_model import SpikingNet, TestNet, LMU, LMUConv
import nengo_dl
import tensorflow as tf
import numpy as np
def combined_p_tolerance_accuracy(y_true_x, y_true_y, y_pred_x, y_pred_y, tolerance, width_scale, height_scale):
    y_true = tf.stack([y_true_x, y_true_y], axis=-1)
    y_pred = tf.stack([y_pred_x, y_pred_y], axis=-1)

    diff = tf.abs(y_true - y_pred)
    diff = diff * tf.constant([width_scale, height_scale], dtype=tf.float32)
    within_tolerance = tf.reduce_sum(tf.cast(diff <= tolerance, tf.float32), axis=-1)
    
    accuracy = tf.reduce_mean(tf.cast(within_tolerance == 2, tf.float32))
    return accuracy
def create_combined_metric(tolerance):
    def combined_metric(y_true, y_pred):
        y_true_x, y_true_y = y_true[:, :, 0:1], y_true[:, :, 1:2]
        y_pred_x, y_pred_y = y_pred[:, :, 0:1], y_pred[:, :, 1:2]
        return combined_p_tolerance_accuracy(y_true_x, y_true_y, y_pred_x, y_pred_y, tolerance, args.sensor_width * args.spatial_factor, args.sensor_height * args.spatial_factor)
    return combined_metric

# Define combined metrics for different tolerances
combined_p1_accuracy = create_combined_metric(tolerance=1)
combined_p3_accuracy = create_combined_metric(tolerance=3)
combined_p5_accuracy = create_combined_metric(tolerance=5)
combined_p10_accuracy = create_combined_metric(tolerance=10)
combined_p15_accuracy = create_combined_metric(tolerance=15)
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

def train(model, sim, p_x, train_loader, val_loader, args):
    best_val_loss = float("inf")
    patience_counter = 0  # Counter to keep track of patient epochs

    # Training loop
    for epoch in range(args.num_epochs):
        model, train_loss, metrics = train_nengo_epoch(model, sim, p_x, train_loader, args)
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
    # Configure TensorFlow for multi-GPU support
    strategy = tf.distribute.MirroredStrategy()
    print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
    device = "/gpu"
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # for gpu in gpus:
    #     tf.config.experimental.set_memory_growth(gpu, True)
    # Open a strategy scope.
    if True:
        lr = args.lr
        # Define your model, optimizer, and criterion
        # model, inp, out_p, out_p_filt = TestNet().build_model()
        # model, inp, p_x, p_y, p_b, p_x_filt, p_y_filt, p_b_filt = TestNet(lr=lr).build_model()
        minibatch_size = 16
        model, inp, p_x, p_y, p_x_filt, p_y_filt = TestNet(batch_size = minibatch_size, lr=lr).build_model()

        sim = nengo_dl.Simulator(model, minibatch_size=minibatch_size, device=device)
        with strategy.scope():
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
            # post_slicer_transform = transforms.Compose([
            #     SliceLongEventsToShort(time_window=int(10000 / temp_subsample_factor), overlap=0, include_incomplete=True),
            #     EventSlicesToMap(sensor_size=(int(640*factor), int(480*factor), 2), \
            #                             n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization,
            #                             map_type=args.map_type)
            # ])
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

            train_data_orig = ThreeETplus_EyetrackingDataset(data_dir=args.data_dir, split="train", transform=transforms.Downsample(spatial_factor=factor), target_transform=label_transform, slicer=train_slicer, post_slicer_transform = post_slicer_transform, device = device)
            # val_data_orig = ThreeETplus_EyetrackingDataset(data_dir=args.data_dir, split="test", transform=transforms.Downsample(spatial_factor=factor), target_transform=label_transform, slicer=val_slicer, post_slicer_transform = post_slicer_transform)
            train_x = train_data_orig.train_x
            train_y = train_data_orig.train_y
            val_x = train_data_orig.val_x
            val_y = train_data_orig.val_y
            test_x = train_data_orig.test_x
            test_y = train_data_orig.test_y

            isTrain = False
            sim.compile(
                optimizer=tf.optimizers.Adam(),
                loss={
                    p_x: tf.losses.MeanSquaredError(),
                    p_y: tf.losses.MeanSquaredError(),
                    # p_b: tf.losses.MeanSquaredError(),
                },
                metrics={
                }
            )
        # with tf.device(device):
        # with strategy.scope():
        # if True:
            if isTrain:
            
                # sim.fit(x=train_x, y=train_y)
                # val_loss = sim.evaluate(x=val_x, y=val_y)
                best_val_loss = float("inf")
                patience_counter = 0
                for epoch in range(args.num_epochs):
                    print(f"epoch {epoch}")
                    sim.fit(
                        # x={inp: train_x},  y={p_x: train_y[:, :, 0:1], p_y: train_y[:, :, 1:2]})
                        x={inp: train_x},  y={p_x: train_y[:, :, 0:1], p_y: train_y[:, :, 0:1],
                                            p_x_filt: train_y[:, :, 0:1], p_y_filt: train_y[:, :, 0:1]})
                        # x={inp: train_x},  y={p_x: train_y, p_x_filt: train_y})
                    losses = sim.evaluate(x={inp: val_x}, y={p_x: val_y[:, :, 0:1], p_y: val_y[:, :, 1:2],
                                                            p_x_filt: val_y[:, :, 0:1], p_y_filt: val_y[:, :, 1:2]})
                    # val_loss = sim.evaluate(x={inp: val_x}, y={p_x: val_y, p_x_filt: val_y})['loss']
                    val_loss = losses['loss']
                    print(f"val_loss: {losses}")
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        sim.save_params("./best_model")
                    else:
                        patience_counter += 1
                    print(f"patient {patience_counter}")
                    if patience_counter >= 50:
                        print("Early stopping due to no improvement in validation loss for 10 consecutive epochs.")
                        break
                test = sim.evaluate(x={inp: test_x}, y={p_x: test_y[:, :, 0:1], p_y: test_y[:, :, 1:2],
                                                        p_x_filt: test_y[:, :, 0:1], p_y_filt: test_y[:, :, 1:2]})
                # Merge train, validation, and test sets
                combined_x = np.concatenate([train_x, val_x, test_x], axis=0)
                combined_y = np.concatenate([train_y, val_y, test_y], axis=0)

                # Load the best model
                sim.load_params("./best_model")
                # Train the best model for 50 more epochs on the combined dataset
                for epoch in range(50):
                    print(f"Additional training epoch {epoch}")

                    sim.fit(x={inp: combined_x}, y={p_x: combined_y[:, :, 0:1], p_y: combined_y[:, :, 1:2],
                                                    p_x_filt: combined_y[:, :, 0:1], p_y_filt: combined_y[:, :, 1:2]})

                # Evaluate the final model
                final_loss = sim.evaluate(x={inp: combined_x}, y={p_x: combined_y[:, :, 0:1], p_y: combined_y[:, :, 1:2],
                                                                p_x_filt: combined_y[:, :, 0:1], p_y_filt: combined_y[:, :, 1:2]})
                print(f"Final loss after additional training: {final_loss}")
            else:
                combined_x = np.concatenate([train_x, val_x, test_x], axis=0)
                combined_y = np.concatenate([train_y, val_y, test_y], axis=0)
                best_train_loss = float("inf")
                # Load the best model
                # sim.load_params("./best_model")
                for epoch in range(50):
                    print(f"Additional training epoch {epoch}")
                    sim.fit(x={inp: combined_x}, y={p_x: combined_y[:, :, 0:1], p_y: combined_y[:, :, 1:2],
                                                    p_x_filt: combined_y[:, :, 0:1], p_y_filt: combined_y[:, :, 1:2]})
                    # losses = sim.evaluate(x={inp: combined_x}, y={p_x: combined_y[:, :, 0:1], p_y: combined_y[:, :, 1:2],
                    #                                            p_x_filt: combined_y[:, :, 0:1], p_y_filt: combined_y[:, :, 1:2]})
                    # loss = losses['loss']
                    # if loss < best_train_loss:
                    #     best_train_loss = loss
                    sim.save_params("./best_model")
    # with tf.device(device):
    #     if isTrain:
            
    #         # sim.fit(x=train_x, y=train_y)
    #         # val_loss = sim.evaluate(x=val_x, y=val_y)
    #         best_val_loss = float("inf")
    #         patience_counter = 0
    #         for epoch in range(args.num_epochs):
    #             print(f"epoch {epoch}")
    #             sim.fit(
    #                 # x={inp: train_x},  y={p_x: train_y[:, :, 0:1], p_y: train_y[:, :, 1:2]})
    #                 x={inp: train_x},  y={p_x: train_y[:, :, 0:1], p_y: train_y[:, :, 0:1], p_b: train_y[:, :, 1:2], 
    #                                     p_x_filt: train_y[:, :, 0:1], p_y_filt: train_y[:, :, 0:1], p_b_filt: train_y[:, :, 1:2]})
    #                 # x={inp: train_x},  y={p_x: train_y, p_x_filt: train_y})
    #             losses = sim.evaluate(x={inp: val_x}, y={p_x: val_y[:, :, 0:1], p_y: val_y[:, :, 1:2], p_b: val_y[:, :, 2:3],
    #                                                        p_x_filt: val_y[:, :, 0:1], p_y_filt: val_y[:, :, 1:2], p_b_filt: val_y[:, :, 2:3]})
    #             # val_loss = sim.evaluate(x={inp: val_x}, y={p_x: val_y, p_x_filt: val_y})['loss']
    #             val_loss = losses['loss']
    #             print(f"val_loss: {losses}")
    #             if val_loss < best_val_loss:
    #                 best_val_loss = val_loss
    #                 patience_counter = 0
    #                 sim.save_params("./best_model")
    #             else:
    #                 patience_counter += 1
    #             print(f"patient {patience_counter}")
    #             if patience_counter >= 50:
    #                 print("Early stopping due to no improvement in validation loss for 10 consecutive epochs.")
    #                 break
    #         test = sim.evaluate(x={inp: test_x}, y={p_x: test_y[:, :, 0:1], p_y: test_y[:, :, 1:2], p_b: test_y[:, :, 2:3],
    #                                                 p_x_filt: test_y[:, :, 0:1], p_y_filt: test_y[:, :, 1:2], p_b_filt: test_y[:, :, 2:3]})
    #         # Merge train, validation, and test sets
    #         combined_x = np.concatenate([train_x, val_x, test_x], axis=0)
    #         combined_y = np.concatenate([train_y, val_y, test_y], axis=0)

    #         # Load the best model
    #         sim.load_params("./best_model")
    #         # Train the best model for 50 more epochs on the combined dataset
    #         for epoch in range(50):
    #             print(f"Additional training epoch {epoch}")

    #             sim.fit(x={inp: combined_x}, y={p_x: combined_y[:, :, 0:1], p_y: combined_y[:, :, 1:2], p_b: combined_y[:, :, 2:3],
    #                                             p_x_filt: combined_y[:, :, 0:1], p_y_filt: combined_y[:, :, 1:2], p_b_filt: combined_y[:, :, 2:3]})

    #         # Evaluate the final model
    #         final_loss = sim.evaluate(x={inp: combined_x}, y={p_x: combined_y[:, :, 0:1], p_y: combined_y[:, :, 1:2], p_b: combined_y[:, :, 2:3],
    #                                                           p_x_filt: combined_y[:, :, 0:1], p_y_filt: combined_y[:, :, 1:2], p_b_filt: combined_y[:, :, 2:3]})
    #         print(f"Final loss after additional training: {final_loss}")
    #     else:
    #         combined_x = np.concatenate([train_x, val_x, test_x], axis=0)
    #         combined_y = np.concatenate([train_y, val_y, test_y], axis=0)
    #         # Load the best model
    #         # sim.load_params("./best_model")
    #         for epoch in range(50):
    #             print(f"Additional training epoch {epoch}")
    #             sim.fit(x={inp: combined_x}, y={p_x: combined_y[:, :, 0:1], p_y: combined_y[:, :, 1:2], p_b: combined_y[:, :, 2:3],
    #                                             p_x_filt: combined_y[:, :, 0:1], p_y_filt: combined_y[:, :, 1:2], p_b_filt: combined_y[:, :, 2:3]})
    #             sim.save_params("./best_model")