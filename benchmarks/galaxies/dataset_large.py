from functools import partial
import tensorflow as tf

# Mean and std for halos and cosmological parameters
MEAN_HALOS_DICT = {'x': 499.91877075908684, 'y': 500.0947802559321,  'z': 499.964508664328,'Jx': 212560050888254.06,'Jy': 349712732356652.25, 'Jz': -100259775332585.12, 'vx': -0.0512854365234889, 'vy': -0.01263126442198149, 'vz': -0.06458034372345466, 'M200c': 321308383763206.9, 'Rvir': 1424.4071655758826}
STD_HALOS_DICT = {'x': 288.71092533309235, 'y': 288.7525818573022, 'z': 288.70234893905575, 'Jx': 2.4294356933448945e+18, 'Jy': 2.3490019110577966e+18, 'Jz': 2.406422979830857e+18, 'vx': 344.0231468131901, 'vy': 343.9333673335964, 'vz': 344.071876710777, 'M200c': 405180433634974.75, 'Rvir': 298.14502916425675}
MEAN_PARAMS_DICT = {'Omega_m': 0.29994175, 'Omega_b': 0.049990308, 'h': 0.69996387, 'n_s': 0.9999161, 'sigma_8': 0.7999111}
STD_PARAMS_DICT = {'Omega_m': 0.11547888, 'Omega_b': 0.017312417, 'h': 0.11543678, 'n_s': 0.115482554, 'sigma_8': 0.11545073}

def _parse_function(proto, features=['x', 'y', 'z', 'Jx', 'Jy', 'Jz', 'vx', 'vy', 'vz', 'M200c'], 
                    params=['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']):
    
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

    return stacked_features, stacked_params

def get_halo_dataset(batch_size=64, 
                     num_samples=None,  # If not None, only return this many samples
                     split='train',
                     features=['x', 'y', 'z', 'Jx', 'Jy', 'Jz', 'vx', 'vy', 'vz', 'M200c', 'Rvir'],
                     params=['Omega_m', 'sigma_8'],
                     return_mean_std=False,
                     standardize=True,
                     seed=42,
                     tfrecords_path='/n/holystore01/LABS/iaifi_lab/Lab/quijote_bsq_tfrecords'):

    files = tf.io.gfile.glob(f"{tfrecords_path}/halos*{split}*.tfrecord")
    dataset = tf.data.TFRecordDataset(files)

    if num_samples is not None:
        dataset = dataset.take(num_samples)
        num_total = num_samples  # Adjust num_total if num_samples is specified
    else:
        num_total = sum(1 for _ in tf.data.TFRecordDataset(files))  # This could be inefficient for large datasets

    dataset = dataset.map(partial(_parse_function, features=features, params=params))

    # Get mean and std as tf arrays
    mean = tf.constant([MEAN_HALOS_DICT[f] for f in features], dtype=tf.float32)
    std = tf.constant([STD_HALOS_DICT[f] for f in features], dtype=tf.float32)
    mean_params = tf.constant([MEAN_PARAMS_DICT[f] for f in params], dtype=tf.float32)
    std_params = tf.constant([STD_PARAMS_DICT[f] for f in params], dtype=tf.float32)

    if standardize:
        dataset = dataset.map(lambda x, p: ((x - mean) / std, (p - mean_params) / std_params))

    dataset = dataset.batch(batch_size)
    if split == 'train':
        dataset = dataset.shuffle(buffer_size=num_total, seed=seed)

    if split == 'train':
        dataset = dataset.repeat()
    
    if return_mean_std:
        return dataset, num_total, mean, std, mean_params, std_params
    else:
        return dataset, num_total