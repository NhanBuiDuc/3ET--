import jax
import jax.numpy as jnp
from collections import namedtuple
import numpy as np
from sklearn.model_selection import train_test_split

State = namedtuple("State", ["obs", "labels"])

class SplitDataLoader:
    """
    Dataloading wrapper for a custom dataset. The entire dataset is loaded to vRAM in a temporally compressed format.

    :batch_size: Number of samples per batch.
    """

    def __init__(self, train_x, train_y, val_x, val_y, test_x, test_y, batch_size=256):
        self.batch_size = batch_size

        # Convert the provided data to jax.numpy arrays
        self.x_train = train_x
        self.y_train = train_y
        self.train_len = len(train_x)

        self.x_val = val_x
        self.y_val = val_y
        self.val_len = len(val_x)

        self.x_test = test_x
        self.y_test = test_y
        self.test_len = len(test_x)

        @jax.jit
        def _train_epoch(shuffle_key):

            obs = jax.random.permutation(shuffle_key, self.x_train, axis=0)
            labels = jax.random.permutation(shuffle_key, self.y_train, axis=0)

            return State(obs=obs, labels=labels)

        self.train_epoch = _train_epoch

        @jax.jit
        def _val_epoch():

            x_val = self.x_val
            y_val = self.y_val

            return State(obs=x_val, labels=y_val)

        self.val_epoch = _val_epoch

        @jax.jit
        def _test_epoch():

            x_test = self.x_test
            y_test = self.y_test

            return State(obs=x_test, labels=y_test)

        self.test_epoch = _test_epoch

class DataLoader:
    """
    Dataloading wrapper for a custom dataset. The entire dataset is loaded to vRAM in a temporally compressed format.

    :batch_size: Number of samples per batch.
    """

    def __init__(self, data, targets, batch_size=64):
        self.batch_size = batch_size

        # Convert the provided data to jax.numpy arrays
        self.data = data
        self.targets = targets
        self.len = len(data)


        @jax.jit
        def _train_epoch(shuffle_key):

            obs = jax.random.permutation(shuffle_key, self.data, axis=0)
            labels = jax.random.permutation(shuffle_key, self.targets, axis=0)

            return State(obs=obs, labels=labels)

        self.train_epoch = _train_epoch

       
class SHD_loader():
    """
    Dataloading wrapper for the Spiking Heidelberg Dataset. The entire dataset is loaded to vRAM in a temporally compressed format. The user must
    apply jnp.unpackbits(events, axis=<time axis>) prior to feeding to an SNN. 

    https://zenkelab.org/resources/spiking-heidelberg-datasets-shd/

    
    :batch_size: Number of samples per batch.
    :sample_T: Number of time steps per sample.
    :channels: Number of frequency channels used.
    :val_size: Fraction of the training dataset to set aside for validation.
    """


    # Change this to allow a config dictionary of 
    def __init__(self, batch_size=256, sample_T = 128, channels=128, val_size=0.2):
        #####################################
        # Load datasets and process them using tonic.
        #####################################
        if not optional_dependencies_installed:
            raise ImportError("Please install the optional dependencies by running 'pip install spyx[loaders]' to use this feature.")

        shd_timestep = 1e-6
        shd_channels = 700
        net_channels = channels
        net_dt = 1/sample_T
           
        self.batch_size = batch_size
        self.val_size = val_size
        self.obs_shape = tuple([net_channels,])
        self.act_shape = tuple([20,])
        
        transform = transforms.Compose([
        transforms.Downsample(
            time_factor=shd_timestep / net_dt,
            spatial_factor=net_channels / shd_channels
            ),
            _SHD2Raster(net_channels, sample_T = sample_T)
        ])
        
        train_val_dataset = datasets.SHD("./data", train=True, transform=transform)
        test_dataset = datasets.SHD("./data", train=False, transform=transform)
        
        
        #########################################################################
        # load entire dataset to GPU as JNP Array, create methods for splits
        #########################################################################

    
        # create train/validation split here...
        # generate indices: instead of the actual data we pass in integers instead
        train_indices, val_indices = train_test_split(
            range(len(train_val_dataset)),
            test_size=self.val_size,
            random_state=0,
            shuffle=True # This really should be set externally!!!!!
        )


        train_split = Subset(train_val_dataset, train_indices)
        self.train_len = len(train_indices)

        train_dl = iter(DataLoader(train_split, batch_size=self.train_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
        x_train, y_train = next(train_dl)
        self.x_train = jnp.array(x_train, dtype=jnp.uint8)
        self.y_train = jnp.array(y_train, dtype=jnp.uint8)
        ############################
        
        val_split = Subset(train_val_dataset, val_indices)
        self.val_len = len(val_indices)

        val_dl = iter(DataLoader(val_split, batch_size=self.val_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=False))
        
        x_val, y_val = next(val_dl)
        self.x_val = jnp.array(x_val, dtype=jnp.uint8)
        self.y_val = jnp.array(y_val, dtype=jnp.uint8)
        ##########################
        # Test set setup
        ##########################
        self.test_len = len(test_dataset)
        test_dl = iter(DataLoader(test_dataset, batch_size=self.test_len,
                          collate_fn=tonic.collation.PadTensors(batch_first=True), drop_last=True, shuffle=True))
                
        x_test, y_test = next(test_dl)
        self.x_test = jnp.array(x_test, dtype=jnp.uint8)
        self.y_test = jnp.array(y_test, dtype=jnp.uint8)


        
        @jax.jit
        def _train_epoch(shuffle_key):
            cutoff = self.train_len % self.batch_size
            
            obs = jax.random.permutation(shuffle_key, self.x_train, axis=0)[:-cutoff] # self.x_train[:-cutoff]
            labels = jax.random.permutation(shuffle_key, self.y_train, axis=0)[:-cutoff] # self.y_train[:-cutoff]
            
            obs = jnp.reshape(obs, (-1, self.batch_size) + obs.shape[1:])
            labels = jnp.reshape(labels, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
            
        self.train_epoch = _train_epoch
            
        @jax.jit
        def _val_epoch():
            cutoff = self.val_len % self.batch_size
            
            x_val = self.x_val[:-cutoff]
            y_val = self.y_val[:-cutoff]
            
            obs = jnp.reshape(x_val, (-1, self.batch_size) + x_val.shape[1:])
            labels = jnp.reshape(y_val, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
        
        self.val_epoch = _val_epoch
        
        
        @jax.jit
        def _test_epoch():
            cutoff = self.test_len % self.batch_size
            
            x_test = self.x_test[:-cutoff]
            y_test = self.y_test[:-cutoff]
            
            obs = jnp.reshape(x_test, (-1, self.batch_size) + x_test.shape[1:])
            labels = jnp.reshape(y_test, (-1, self.batch_size))
            
            return State(obs=obs, labels=labels)
        
        self.test_epoch = _test_epoch