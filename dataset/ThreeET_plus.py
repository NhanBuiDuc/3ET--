import enum
import os
from typing import Any, Callable, Optional, Tuple
import h5py
import numpy as np
import tensorflow as tf
from tonic.dataset import Dataset
import tonic.transforms as transforms
from sklearn.model_selection import train_test_split

class ThreeETplus_Eyetracking(Dataset):
    """3ET DVS eye tracking `3ET <https://github.com/qinche106/cb-convlstm-eyetracking>`_
    ::

        @article{chen20233et,
            title={3ET: Efficient Event-based Eye Tracking using a Change-Based ConvLSTM Network},
            author={Chen, Qinyu and Wang, Zuowen and Liu, Shih-Chii and Gao, Chang},
            journal={arXiv preprint arXiv:2308.11771},
            year={2023}
        }

        authors: Qinyu Chen^{1,2}, Zuowen Wang^{1}
        affiliations: 1. Institute of Neuroinformatics, University of Zurich and ETH Zurich, Switzerland
                      2. Univeristy of Leiden, Netherlands

    Parameters:
        save_to (string): Location to save files to on disk.
        transform (callable, optional): A callable of transforms to apply to the data.
        split (string, optional): The dataset split to use, ``train`` or ``val``.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and
                                         labels at the same time.

    Returns:
         A dataset object that can be indexed or iterated over.
         One sample returns a tuple of (events, targets).
    """

    sensor_size = (640, 480, 2)
    dtype = np.dtype([("t", int), ("x", int), ("y", int), ("p", int)])
    ordering = dtype.names

    def __init__(
        self,
        save_to: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ):
        super().__init__(
            save_to,
            transform=transform,
            target_transform=target_transform,
            transforms=transforms,
        )

        data_dir = save_to
        data_list_dir = './dataset'
        # Load filenames from the provided lists
        if split == "train":
            filenames = self.load_filenames(os.path.join(data_list_dir, "train_files.txt"))
        elif split == "val":
            filenames = self.load_filenames(os.path.join(data_list_dir, "val_files.txt"))
        elif split == "test":
            filenames = self.load_filenames(os.path.join(data_list_dir, "test_files.txt"))
        else:
            raise ValueError("Invalid split name")

        # Get the data file paths and target file paths
        if split == "train" or split == "val":
            self.data = [os.path.join(data_dir, "train", f, f + ".h5") for f in filenames]
            self.targets = [os.path.join(data_dir, "train", f, "label.txt") for f in filenames]
        elif split == "test":
            self.data = [os.path.join(data_dir, "test", f, f + ".h5") for f in filenames]
            # for test set, we load the placeholder labels with all zeros
            self.targets = [os.path.join(data_dir, "test", f, "label_zeros.txt") for f in filenames]
    # def __getitem__(self, index: int) -> Tuple[Any, Any]:
    #     """
    #     Returns:
    #         (events, target) where target is index of the target class.
    #     """
    #     # get events from .h5 file
    #     with h5py.File(self.data[index], "r") as f:
    #         # original events.dtype is dtype([('t', '<u8'), ('x', '<u8'), ('y', '<u8'), ('p', '<u8')])
    #         # t is in us
    #         events = f["events"][:].astype(self.dtype)
    #         events['p'] = events['p']*2 -1  # convert polarity to -1 and 1
            
    #     # load the sparse labels
    #     with open(self.targets[index], "r") as f:
    #         # target is at the frequency of 100 Hz. It will be downsampled to 20 Hz in the target transformation
    #         target = np.array(
    #             [list(map(float, line.strip('()\n').split(', '))) for line in f.readlines()], np.float32)
        
    #     # Convert NumPy arrays to TensorFlow tensors
    #     events_tf = tf.constant(events)
    #     target_tf = tf.constant(target)

    #     if self.transform is not None:
    #         events_tf = self.transform(events_tf)
    #     if self.target_transform is not None:
    #         target_tf = self.target_transform(target_tf)
    #     if self.transforms is not None:
    #         events_tf, target_tf = self.transforms(events_tf, target_tf)
    #     return events_tf, target_tf
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Returns:
            (events, target) where target is index of the target class.
        """
        # get events from .h5 file
        with h5py.File(self.data[index], "r") as f:
            # original events.dtype is dtype([('t', '<u8'), ('x', '<u8'), ('y', '<u8'), ('p', '<u8')])
            # t is in us
            events = f["events"][:].astype(self.dtype)
            events['p'] = events['p']*2 -1  # convert polarity to -1 and 1
            
        # load the sparse labels
        with open(self.targets[index], "r") as f:
            # target is at the frequency of 100 Hz. It will be downsampled to 20 Hz in the target transformation
            target = np.array(
                [list(map(float, line.strip('()\n').split(', '))) for line in f.readlines()], np.float32)

        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.transforms is not None:
            events, target = self.transforms(events, target)
        return events, target

    def __len__(self):
        return len(self.data)

    def _check_exists(self):
        return self._is_file_present()

    def load_filenames(self, path):
        with open(path, "r") as f:
            return [line.strip() for line in f.readlines()]



class ThreeETplus_EyetrackingDataset:
    """
    Raw Eyetracking Dataset

    Parameters:
        data_dir (str): Directory containing data files.
        label_dir (str): Directory containing label files.
        transform (callable, optional): A callable of transforms to apply to the data.
        target_transform (callable, optional): A callable of transforms to apply to the targets/labels.
        transforms (callable, optional): A callable of transforms that is applied to both data and labels at the same time.
    """

    def __init__(
        self,
        data_dir: str,
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        slicer: Optional[Callable] = None,
        post_slicer_transform = None
    ):
        self.split = split
        self.data_dir = data_dir
        self.data_files = os.listdir(os.path.join(data_dir, split))
        # Load all data and labels into memory
        self.transform = transform
        self.target_transform = target_transform
        self.slicer = slicer
        self.post_slicer_transform = post_slicer_transform
        # Load all data and labels into memory
        self.train_x, self.train_y, self.val_x, self.val_y, self.test_x, self.test_y = self.load_data()
    def __len__(self):
        return len(self.data_files)

    def load_data(self):
        all_inputs = []
        all_targets = []
        for index, dir in enumerate(self.data_files):
            data_file_path = os.path.join(os.path.join(self.data_dir, self.split, dir), dir + ".h5")
            if self.split == "test":
                label_file_path = os.path.join(os.path.join(self.data_dir, self.split, dir), "label_zeros.txt")               
            else:
                label_file_path = os.path.join(os.path.join(self.data_dir, self.split, dir), "label.txt")

            with h5py.File(data_file_path, "r") as f:
                events = f["events"][:].astype(np.dtype([("t", int), ("x", int), ("y", int), ("p", int)]))
                events['p'] = events['p'] * 2 - 1  # convert polarity to -1 and 1

            with open(label_file_path, "r") as f:
                target = np.array(
                    [list(map(float, line.strip('()\n').split(', '))) for line in f.readlines()], dtype=np.float32)
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                target = self.target_transform(target)
            if self.slicer is not None:
                sliced_events, sliced_targets = self.slicer.slice(events, target)
                if self.post_slicer_transform is not None:
                    sliced_events = [self.post_slicer_transform(ev) for ev in sliced_events]
                    # sliced_targets = [self.post_slicer_transform(tg) for tg in sliced_targets]
                all_inputs.extend(sliced_events)
                all_targets.extend(sliced_targets)
            else:
                all_inputs.append(events)
                all_targets.append(target)
        train_x, val_x, train_y, val_y = train_test_split(all_inputs, all_targets, test_size=0.3, random_state=42)
        val_x, test_x, val_y, test_y = train_test_split(val_x, val_y, test_size=0.3, random_state=42)
            # print(events.shape)
            # print(target.shape)
        train_x =  tf.constant(train_x)
        train_x = tf.reshape(train_x, [train_x.shape[0] , train_x.shape[1] * train_x.shape[2], -1])

        val_x =  tf.constant(val_x)
        val_x = tf.reshape(val_x, [val_x.shape[0] , val_x.shape[1] * val_x.shape[2], -1])

        train_y =  tf.constant(train_y)

        val_y =  tf.constant(val_y)

        test_x =  tf.constant(test_x)
        test_x = tf.reshape(test_x, [test_x.shape[0] , test_x.shape[1] * test_x.shape[2], -1])

        test_y =  tf.constant(test_y)
        return train_x, train_y, val_x, val_y, test_x, test_y