import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt

import os
import tensorflow as tf

from tqdm import tqdm
from pycorr import TwoPointCorrelationFunction # Can install it here https://github.com/cosmodesi/pycorr

def compute_tpcf(positions, box_size=1000.,):
    r_bins = np.linspace(0.5, 150.0, 25)
    mu_bins = np.linspace(-1, 1, 201)
    return TwoPointCorrelationFunction(
                "smu",
                edges=(np.array(r_bins), np.array(mu_bins)),
                data_positions1=np.array(positions[:,:3]).T,
                engine="corrfunc",
                boxsize=box_size,
                los="z",
            )(ells=[0])[0]

def read_halos(data_dir, snapshot=10, n_halos=5000,):
    data = np.loadtxt(data_dir / f'out_{snapshot}.list')
    # get the n_halos heaviest ones
    data = data[np.argsort(data[:,21])[-n_halos:]]
    return pd.DataFrame(
        {
            'x': data[:,8],
            'y': data[:,9],
            'z': data[:,10],
            'v_x': data[:,11],
            'v_y': data[:,12],
            'v_z': data[:,13],
            'J_x': data[:,14],
            'J_y': data[:,15],
            'J_z': data[:,16],
            'M200c': data[:,21],
            'Rvir': data[:,5],
        },
    )

def read_cosmologies():
    cosmo_url = "https://raw.githubusercontent.com/franciscovillaescusa/Quijote-simulations/master/BSQ/BSQ_params.txt"
    return pd.read_csv(
                cosmo_url, 
                sep=" ", 
                names=['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma8'], 
                skiprows=1, 
                header=None,
    )

def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def serialize_example(features, tpcf_features, params, params_names):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    feature = {k: _bytes_feature(tf.io.serialize_tensor(tf.cast(v, tf.float32)).numpy()) for k, v in features.items()}

    # Params contains 5 values, pick individual values and save using params_names as the key, similar to features above
    for i, param in enumerate(params):
        feature[params_names[i]] = _bytes_feature(tf.io.serialize_tensor(tf.cast(param, tf.float32)).numpy()) 
    feature['tpcf'] = _bytes_feature(tf.io.serialize_tensor(tf.cast(tpcf_features, tf.float32)).numpy())
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




def write_tfrecords(tfrecord_file, indices, data_dir, snapshot, n_halos):
    print(f"Starting to write TFRecord: {tfrecord_file}")
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for idx in tqdm(indices, desc=f'Writing {tfrecord_file}'):
            try:
                features = read_halos(data_dir / f'{idx}', snapshot, n_halos)
                tpcf_features = compute_tpcf(features[['x', 'y', 'z']].values)
                params = params_df.iloc[idx].values
                example = serialize_example(features, tpcf_features, params, params_names)
                writer.write(example)
            except Exception as e:
                print(f"Error writing index {idx} to {tfrecord_file}: {e}")

def process_data_to_tfrecords(params_df, data_dir, tfrecords_path, num_tfrecords=20, num_tfrecords_val=1, num_tfrecords_test=1, snapshot=10, n_halos=5000):
    """
    Converts data from the Quijote simulations to a specified number of TFRecord files.
    """
    # Make tfrecords_path if it doesn't exist
    os.makedirs(tfrecords_path, exist_ok=True)
    print(f"TFRecords path: {tfrecords_path}")

    num_files = len(params_df)
    files_per_tfrecord = int(np.ceil(num_files / (num_tfrecords + num_tfrecords_val + num_tfrecords_test)))
    print(f'{files_per_tfrecord} files per tfrecord')

    for i in range(num_tfrecords):
        tfrecord_file = os.path.join(tfrecords_path, f'halos_train_{i + 1}.tfrecord')
        indices = range(i * files_per_tfrecord, min((i + 1) * files_per_tfrecord, num_files))
        write_tfrecords(tfrecord_file, indices, data_dir, snapshot, n_halos)

    for i in range(num_tfrecords_val):
        tfrecord_file = os.path.join(tfrecords_path, f'halos_val_{i + 1}.tfrecord')
        indices = range((num_tfrecords + i) * files_per_tfrecord, min((num_tfrecords + i + 1) * files_per_tfrecord, num_files))
        write_tfrecords(tfrecord_file, indices, data_dir, snapshot, n_halos)

    for i in range(num_tfrecords_test):
        tfrecord_file = os.path.join(tfrecords_path, f'halos_test_{i + 1}.tfrecord')
        indices = range((num_tfrecords + num_tfrecords_val + i) * files_per_tfrecord, min((num_tfrecords + num_tfrecords_val + i + 1) * files_per_tfrecord, num_files))
        write_tfrecords(tfrecord_file, indices, data_dir, snapshot, n_halos)


if __name__ == '__main__':
    data_dir = Path('/pscratch/sd/c/cuesta/quijote_bsq/')
    params_names = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
    params_df = read_cosmologies()

    process_data_to_tfrecords(
        params_df,
        num_tfrecords=50,
        data_dir = Path('/pscratch/sd/c/cuesta/quijote_bsq/'),
        tfrecords_path = Path('/pscratch/sd/c/cuesta/quijote_tfrecords/'),
    )