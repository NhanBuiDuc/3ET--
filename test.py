"""
Author: Zuowen Wang
Affiliation: Insitute of Neuroinformatics, University of Zurich and ETH Zurich
Email: wangzu@ethz.ch
"""

import argparse, json, os, csv
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model.BaselineEyeTrackingModel import CNN_GRU
from dataset import ThreeETplus_Eyetracking, ScaleLabel, NormalizeLabel, \
    TemporalSubsample, NormalizeLabel, SliceLongEventsToShort, \
    EventSlicesToVoxelGrid, SliceByTimeEventsTargets
import tonic.transforms as transforms
from tonic import SlicedDataset, DiskCachedDataset
from nengo_model import SpikingNet, TestNet
import pandas as pd
import numpy as np
import nengo_dl
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1' 
os.environ["CUDA_VISIBLE_DEVICES"] = '3'
def p_acc(target, prediction, pixel_tolerances=[1,3,5,10]):
    """
    Calculate the accuracy of prediction
    :param target: (N, seq_len, 2) tensor, seq_len could be 1
    :param prediction: (N, seq_len, 2) tensor
    :return: a dictionary of p-total correct and batch size of this batch
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dist = torch.norm(dis, dim=-1)

    total_correct = {}
    for p_tolerance in pixel_tolerances:
        total_correct[f'p{p_tolerance}'] = torch.sum(dist < p_tolerance)

    bs_times_seqlen = target.shape[0]
    return total_correct, bs_times_seqlen

def px_euclidean_dist(target, prediction):
    """
    Calculate the total pixel euclidean distance between target and prediction
    in a batch over the sequence length
    :param target: (N, seqlen, 2) tensor
    :param prediction: (N, seqlen, 2) tensor
    :return: total pixel euclidean distance and sample numbers
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)[:, :2]
    prediction = prediction.reshape(-1, 2)

    dis = target - prediction
    dist = torch.norm(dis, dim=-1)
    total_px_euclidean_dist = torch.sum(dist)
    sample_numbers = target.shape[0]
    return total_px_euclidean_dist, sample_numbers

def px_manhaten_dist(target, prediction):
    """
    Calculate the total pixel manhaten distance between target and prediction
    in a batch over the sequence length
    :param target: (N, seqlen, 2) tensor
    :param prediction: (N, seqlen, 2) tensor
    :return: total pixel manhaten distance and sample numbers
    """
    # flatten the N and seqlen dimension of target and prediction
    target = target.reshape(-1, 2)[:, :2]
    prediction = prediction.reshape(-1, 2)
    dis = target - prediction
    dist = torch.sum(torch.abs(dis), dim=-1)
    total_px_manhaten_dist = torch.sum(dist)
    sample_numbers = target.shape[0]
    return total_px_manhaten_dist, sample_numbers

def calculate_metrics(gt_df, predictions_df):
    # Convert the dataframes into PyTorch tensors
    gt_tensor = torch.tensor(gt_df[['x', 'y']].values, dtype=torch.float32)
    predictions_tensor = torch.tensor(predictions_df[['x', 'y']].values, dtype=torch.float32)

    # Calculate pixel tolerated accuracy
    total_correct, sample_size = p_acc(gt_tensor, predictions_tensor, pixel_tolerances=[1,3,5,10,15])
    # enumerate the tolerances and print the accuracy
    print("p_acc:", end=" ")
    for p_tolerance, correct in total_correct.items():
        print(f'{p_tolerance}={correct.item() / sample_size:.2%}', end=", ")
    print()
    # calculate the px_euclidean_dist
    total_px_euclidean_dist, sample_size = px_euclidean_dist(gt_tensor, predictions_tensor)
    print(f'px_euclidean_dist: {total_px_euclidean_dist.item() / sample_size:.2f}')

    # calculate the px_manhaten_dist
    total_px_manhaten_dist, sample_size = px_manhaten_dist(gt_tensor, predictions_tensor)
    print(f'px_manhaten_dist: {total_px_manhaten_dist.item() / sample_size:.2f}')
    

def main(args):
    # Load hyperparameters from JSON configuration file
    if args.config_file:
        with open(os.path.join('./configs', args.config_file) + ".json", 'r') as f:
            config = json.load(f)
        # Overwrite hyperparameters with command-line arguments
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
        args = argparse.Namespace(**config)
    else:
        raise ValueError("Please provide a JSON configuration file.")

    # # also dump the args to a JSON file in MLflow artifact
    # with open(os.path.join(mlflow.get_artifact_uri(), "args.json"), 'w') as f:
    #     json.dump(vars(args), f)
    device = "/gpu:3"
    # Define your model, optimizer, and criterion
    model, inp, out_p, out_p_filt = TestNet().build_model()
    minibatch_size = 1
    sim = nengo_dl.Simulator(model, minibatch_size=minibatch_size)
    sim.compile(
        optimizer=tf.optimizers.RMSprop(0.001),
        loss={out_p: tf.losses.SparseCategoricalCrossentropy(from_logits=True)},
    )
    # load parameters
    sim.load_params("./best_model")
    # test data loader always cuts the event stream with the labeling frequency
    factor = args.spatial_factor
    temp_subsample_factor = args.temporal_subsample_factor

    label_transform = transforms.Compose([
        ScaleLabel(factor),
        TemporalSubsample(temp_subsample_factor),
        NormalizeLabel(pseudo_width=640*factor, pseudo_height=480*factor)
    ])

    test_data_orig = ThreeETplus_Eyetracking(save_to=args.data_dir, split="test", \
                    transform=transforms.Downsample(spatial_factor=factor),
                    target_transform=label_transform)

    slicing_time_window = args.test_length*int(10000/temp_subsample_factor) #microseconds

    test_slicer=SliceByTimeEventsTargets(slicing_time_window, overlap=0, \
                    seq_length=args.test_length, seq_stride=args.test_stride, include_incomplete=True)

    post_slicer_transform = transforms.Compose([
        SliceLongEventsToShort(time_window=int(10000/temp_subsample_factor), overlap=0, include_incomplete=True),
        EventSlicesToVoxelGrid(sensor_size=(int(640*factor), int(480*factor), 2), \
                                n_time_bins=args.n_time_bins, per_channel_normalize=args.voxel_grid_ch_normaization)
    ])

    test_data = SlicedDataset(test_data_orig, test_slicer, transform=post_slicer_transform)

    # Uncomment the following lines to use the cached dataset
    # Use with caution! Don't forget to update the cache path if you change the dataset or the slicing parameters

    # test_data = SlicedDataset(test_data_orig, test_slicer, transform=post_slicer_transform, \
    #     metadata_path=f"./metadata/3et_test_l{args.test_length}s{args.test_stride}_ch{args.n_time_bins}")

    # cache the dataset to disk to speed up training. The first epoch will be slow, but the following epochs will be fast.
    # test_data = DiskCachedDataset(test_data, \
    #                               cache_path=f'./cached_dataset/test_l{args.test_length}s{args.test_stride}_ch{args.n_time_bins}')

    assert args.batch_size == 1 
    # otherwise the collate function will through an error. 
    # This is only used in combination of include_incomplete=True during testing
    test_loader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, \
                            num_workers=int(os.cpu_count()-2))
    accumulated_batches = []
    accumulated_targets = []
    
    with open("my_submissions/submission_probe.csv", 'w', newline='') as csvfile_probe, open("my_submissions/submission_probe_filt.csv", 'w', newline='') as csvfile_probe_filt:
        csv_writer_probe = csv.writer(csvfile_probe, delimiter=',')
        csv_writer_probe_filt = csv.writer(csvfile_probe_filt, delimiter=',')
        
        csv_writer_probe.writerow(['row_id', 'x', 'y'])
        csv_writer_probe_filt.writerow(['row_id', 'x', 'y'])
        
        row_id = 0
        with tf.device(device):
            for batch_idx, (data, target_placeholder) in enumerate(test_loader):
                data = data.to(args.device)
                accumulated_batches.append(data)
                accumulated_targets.append(target_placeholder)
                
                if len(accumulated_batches) == minibatch_size:
                    stacked_data = torch.cat(accumulated_batches)
                    tf_data = tf.constant(stacked_data.cpu().detach().numpy())
                    tf_data = tf.reshape(tf_data, [tf_data.shape[0], tf_data.shape[1] * tf_data.shape[2], -1])
                    
                    output = sim.predict(tf_data)
                    out_probe = output[out_p]
                    out_probe_filt = output[out_p_filt]

                    out_probe[..., 0] *= 640 * factor
                    out_probe[..., 1] *= 480 * factor

                    out_probe_filt[..., 0] *= 640 * factor
                    out_probe_filt[..., 1] *= 480 * factor

                    for sample in range(minibatch_size):
                        for frame_id in range(target_placeholder.shape[1]):
                            row_to_write_probe = out_probe[sample][frame_id]
                            row_to_write_probe = np.insert(row_to_write_probe, 0, row_id)
                            csv_writer_probe.writerow(row_to_write_probe)
                            
                            row_to_write_probe_filt = out_probe_filt[sample][frame_id]
                            row_to_write_probe_filt = np.insert(row_to_write_probe_filt, 0, row_id)
                            csv_writer_probe_filt.writerow(row_to_write_probe_filt)
                            
                            row_id += 1

                    # Clear the accumulated batches
                    accumulated_batches = []
                    accumulated_targets = []

            # If there are remaining batches that didn't reach the minibatch size
            if len(accumulated_batches) > 0:
                stacked_data = torch.cat(accumulated_batches)
                tf_data = tf.constant(stacked_data.cpu().detach().numpy())
                tf_data = tf.reshape(tf_data, [tf_data.shape[0], tf_data.shape[1] * tf_data.shape[2], -1])
                
                output = sim.predict(tf_data)
                out_probe = output[out_p]
                out_probe_filt = output[out_p_filt]

                out_probe[..., 0] *= 640 * factor
                out_probe[..., 1] *= 480 * factor

                out_probe_filt[..., 0] *= 640 * factor
                out_probe_filt[..., 1] *= 480 * factor

                for sample in range(len(accumulated_batches)):
                    for frame_id in range(target_placeholder.shape[1]):
                        row_to_write_probe = out_probe[sample][frame_id]
                        row_to_write_probe = np.insert(row_to_write_probe, 0, row_id)
                        csv_writer_probe.writerow(row_to_write_probe)
                        
                        row_to_write_probe_filt = out_probe_filt[sample][frame_id]
                        row_to_write_probe_filt = np.insert(row_to_write_probe_filt, 0, row_id)
                        csv_writer_probe_filt.writerow(row_to_write_probe_filt)
                        
                        row_id += 1
    # load weights from a checkpoint
    # if args.checkpoint:
    #     model.load_state_dict(torch.load(args.checkpoint))
    # else:
    #     raise ValueError("Please provide a checkpoint file.")
    # scaling_factor_torch = torch.tensor((640, 480))
    # scaling_factor_tf = tf.constant((640, 480))
    # evaluate on the validation set and save the predictions into a csv file.
    # with open(args.output_path, 'w', newline='') as csvfile:
    #     csv_writer = csv.writer(csvfile, delimiter=',')
    #     # add column names 'row_id', 'x', 'y'
    #     csv_writer.writerow(['row_id', 'x', 'y'])
    #     row_id = 0
    #     for batch_idx, (data, target_placeholder) in enumerate(test_loader):
    #         data = data.to(args.device)
    #         tf_data = tf.constant(data.cpu().detach().numpy())
    #         tf_data = tf.reshape(tf_data, [tf_data.shape[0] , tf_data.shape[1] * tf_data.shape[2], -1])
    #         output = sim.predict(tf_data)
    #         out_probe = output[out_p]
    #         out_probe_filt = output[out_p_filt]
    #         # Multiplying the first two elements in the last dimension for out_probe
    #         out_probe[..., 0] *= 640 * factor
    #         out_probe[..., 1] *= 480 * factor

    #         # Multiplying the first two elements in the last dimension for out_probe_filt
    #         out_probe_filt[..., 0] *= 640 * factor
    #         out_probe_filt[..., 1] *= 480 * factor
    #         # out_p_tftensor = tf.constant(out_probe)
    #         # out_p_tftensor = tf.multiply(out_p_tftensor, scaling_factor_tf).numpy()

    #         # out_p_filt_tftensor = tf.constant(out_probe_filt)
    #         # out_p_filt_tftensor = tf.multiply(out_p_filt_tftensor, scaling_factor_tf).numpy()

    #         # out_p_torchtensor = torch.tensor(out_probe)
    #         # out_p_torchtensor = torch.mul(out_p_torchtensor, scaling_factor_torch)

    #         # out_p_filt_torchtensor = torch.tensor(out_probe)
    #         # out_p_filt_torchtensor = torch.mul(out_p_filt_torchtensor, scaling_factor_torch)      
    #         # Important! 
    #         # cast the output back to the downsampled sensor space (80x60)
    #         # output = output * torch.tensor((640*factor, 480*factor)).to(args.device)

    #         for sample in range(target_placeholder.shape[0]):
    #             for frame_id in range(target_placeholder.shape[1]):
    #                 row_to_write = out_probe[sample][frame_id]
    #                 # prepend the row_id
    #                 row_to_write = np.insert(row_to_write, 0, row_id)
    #                 csv_writer.writerow(row_to_write)
    #                 row_id += 1

    gt_df = pd.read_csv('./gt_orig_merged.csv')

    # iterate over csv files in a directory
    folder = './my_submissions/'
    files = os.listdir(folder)
    # remove the files that contains null
    files = [file for file in files if 'null' not in file]
    # sort files by the Priv_ score
    # sort_files = sorted(files, key=lambda x: float(x.split('_')[-1].split('.')[0]), reverse=True)
    for filename in files:
        if filename.endswith(".csv"):
            print(filename)
            predictions_df = pd.read_csv(os.path.join(folder, filename))

            calculate_metrics(gt_df, predictions_df)
        else:
            continue
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # a config file 
    parser.add_argument("--config_file", type=str, default='test_config', \
                        help="path to JSON configuration file")
    # load weights from a checkpoint
    parser.add_argument("--checkpoint", type=str, help="path to checkpoint")
    parser.add_argument("--output_path", type=str, default='./submission.csv')

    args = parser.parse_args()

    main(args)
