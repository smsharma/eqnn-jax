from functools import partial
import tensorflow as tf
import numpy as np
from pycorr import (
    TwoPointCorrelationFunction,
)  # Can install it here https://github.com/cosmodesi/pycorr


# Mean and std for halos and cosmological parameters
MEAN_HALOS_DICT = {'x': 499.91877075908684, 'y': 500.0947802559321,  'z': 499.964508664328,'Jx': 212560050888254.06,'Jy': 349712732356652.25, 'Jz': -100259775332585.12, 'vx': -0.0512854365234889, 'vy': -0.01263126442198149, 'vz': -0.06458034372345466, 'M200c': 321308383763206.9, 'Rvir': 1424.4071655758826}
STD_HALOS_DICT = {'x': 288.71092533309235, 'y': 288.7525818573022, 'z': 288.70234893905575, 'Jx': 2.4294356933448945e+18, 'Jy': 2.3490019110577966e+18, 'Jz': 2.406422979830857e+18, 'vx': 344.0231468131901, 'vy': 343.9333673335964, 'vz': 344.071876710777, 'M200c': 405180433634974.75, 'Rvir': 298.14502916425675}
MEAN_PARAMS_DICT = {'Omega_m': 0.29994175, 'Omega_b': 0.049990308, 'h': 0.69996387, 'n_s': 0.9999161, 'sigma_8': 0.7999111}
STD_PARAMS_DICT = {'Omega_m': 0.11547888, 'Omega_b': 0.017312417, 'h': 0.11543678, 'n_s': 0.115482554, 'sigma_8': 0.11545073}
MEAN_TPCF_VEC = [1.47385902e+01, 4.52754450e+00, 1.89688166e+00, 1.00795493e+00,
                6.09400184e-01, 3.98518764e-01, 2.79545049e-01, 2.01358601e-01,
                1.53487009e-01, 1.18745081e-01, 9.51346027e-02, 7.83494908e-02,
                6.92183650e-02, 6.41181254e-02, 6.05992822e-02, 5.77399258e-02,
                5.27855615e-02, 4.64777462e-02, 3.97492901e-02, 3.17941626e-02,
                2.49663476e-02, 1.92553030e-02, 1.28971533e-02, 9.48586955e-03]
STD_TPCF_VEC = [8.37356624, 2.36190046, 1.15493691, 0.73567994, 0.52609708, 0.40239359,
                0.32893873, 0.27772011, 0.24173466, 0.21431925, 0.19276616, 0.17816693,
                0.16773013, 0.15968612, 0.15186733, 0.14234885, 0.13153203, 0.11954234,
                0.10549666, 0.09024256, 0.07655078, 0.06350282, 0.05210615, 0.0426435]

def _parse_function(proto, features=['x', 'y', 'z', 'Jx', 'Jy', 'Jz', 'vx', 'vy', 'vz', 'M200c'], 
                    params=['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8'],
                    include_tpcf=False):
    
    # Define your tfrecord again. It must match with how you wrote it
    keys_to_features = {k: tf.io.FixedLenFeature([], tf.string) for k in features}
    keys_to_params = {k: tf.io.FixedLenFeature([], tf.string) for k in params}

    # Load one example
    parsed_features = tf.io.parse_single_example(proto, keys_to_features)
    parsed_params = tf.io.parse_single_example(proto, keys_to_params)

    # Convert each feature from a serialized string to a tensor and store in a list
    feature_tensors = [tf.io.parse_tensor(parsed_features[k], out_type=tf.float32) for k in features]
    param_tensors = [tf.io.parse_tensor(parsed_params[k], out_type=tf.float32) for k in params]

    # Stack the feature tensors to create a single tensor
    # Each tensor must have the same shape
    stacked_features = tf.stack(feature_tensors, axis=1)  # Creates a [num_points, num_features] tensor
    stacked_params = tf.stack(param_tensors, axis=0)  # Creates a [num_params] tensor

    if include_tpcf:
        key_to_tpcf = {'tpcf': tf.io.FixedLenFeature([], tf.string)}
        parsed_tpcf = tf.io.parse_single_example(proto, key_to_tpcf)
        tpcf_tensor = tf.io.parse_tensor(parsed_tpcf['tpcf'], out_type=tf.float32)

        return stacked_features, stacked_params, tpcf_tensor

    return stacked_features, stacked_params
    

def get_halo_dataset(batch_size=64, 
                     num_samples=None,  # If not None, only return this many samples
                     split='train',
                     features=['x', 'y', 'z', 'Jx', 'Jy', 'Jz', 'vx', 'vy', 'vz', 'M200c', 'Rvir'],
                     params=['Omega_m', 'sigma_8'],
                     return_mean_std=False,
                     standardize=True,
                     seed=42,
                     tfrecords_path= '/quijote_tfrecords_consistent_trees',
                     include_tpcf=False
                     ):

    files = tf.io.gfile.glob(f"{tfrecords_path}/halos*{split}*.tfrecord")
    dataset = tf.data.TFRecordDataset(files)

    if num_samples is not None:
        dataset = dataset.take(num_samples)
        num_total = num_samples  # Adjust num_total if num_samples is specified
    else:
        num_total = sum(1 for _ in tf.data.TFRecordDataset(files))  # This could be inefficient for large datasets
        num_total = num_total - (num_total % batch_size) # Make sure divisible into even batches
        dataset = dataset.take(num_total)

    dataset = dataset.map(partial(_parse_function, features=features, params=params, include_tpcf=include_tpcf))

    # Get mean and std as tf arrays
    mean = tf.constant([MEAN_HALOS_DICT[f] for f in features], dtype=tf.float32)
    std = tf.constant([STD_HALOS_DICT[f] for f in features], dtype=tf.float32)
    mean_params = tf.constant([MEAN_PARAMS_DICT[f] for f in params], dtype=tf.float32)
    std_params = tf.constant([STD_PARAMS_DICT[f] for f in params], dtype=tf.float32)
    mean_tpcf = tf.constant(MEAN_TPCF_VEC)
    std_tpcf = tf.constant(STD_TPCF_VEC)

    if standardize:
        if include_tpcf:
            dataset = dataset.map(lambda x, p, t: ((x - mean) / std, (p - mean_params) / std_params, (t - mean_tpcf) / std_tpcf))
        else:
            dataset = dataset.map(lambda x, p: ((x - mean) / std, (p - mean_params) / std_params))

    if batch_size is None:
        batch_size = num_total

    dataset = dataset.batch(batch_size)
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=num_total, seed=seed)

    if split == 'train':
        dataset = dataset.repeat()
    
    if return_mean_std:
        return dataset, num_total, mean, std, mean_params, std_params
    else:
        return dataset, num_total
    

def generate_tpcfs(batch_size, num_samples, split, save=False, seed=42, standardize=True):
    features = ['x', 'y', 'z']  # ['x', 'y', 'z', 'Jx', 'Jy', 'Jz', 'vx', 'vy', 'vz', 'M200c']
    params = ['Omega_m', 'sigma_8']  # ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']

    dataset, num_total = get_halo_dataset(batch_size=batch_size,  # Batch size
                               num_samples=num_samples,  # If not None, will only take a subset of the dataset
                               split=split,  # 'train', 'val', 'test
                               standardize=False,  # If True, will standardize the features
                               return_mean_std=False,  # If True, will return (dataset, num_total, mean, std, mean_params, std_params), else (dataset, num_total)
                               seed=seed,  # Random seed
                               features=features,  # Features to include
                               params=params  # Parameters to include
                            )
    iterator = iter(dataset)

    halos = []
    for _ in range(num_total // batch_size + num_total % batch_size):
        x, params = next(iterator)
        halos.append(np.array(x))
    halos = np.concatenate(halos, axis=0)
    
    r_bins = np.linspace(0.5, 150.0, 25)
    r_c = 0.5 * (r_bins[1:] + r_bins[:-1])
    mu_bins = np.linspace(-1, 1, 201)
    box_size = 1000.0

    tpcfs = []
    for halo in halos:
        tpcfs.append(
            TwoPointCorrelationFunction(
                "smu",
                edges=(np.array(r_bins), np.array(mu_bins)),
                data_positions1=np.array(halo[:, :3]).T,
                engine="corrfunc",
                boxsize=box_size,
                los="z",
            )(ells=[0])[0]
        )
    tpcfs = np.stack(tpcfs)

    if standardize:
        mean_tpcfs = np.mean(tpcfs, axis=0)
        std_tpcfs = np.std(tpcfs, axis=0)
        tpcfs = (tpcfs - mean_tpcfs) / (std_tpcfs + 1e-10)
    
    if save:
        np.save(f"tpcfs/tpcfs_{split}_large.npy", tpcfs)

    return tpcfs