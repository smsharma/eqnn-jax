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

def read_halos(data_dir, snapshot=10, num_halos=5000,):
    data = np.loadtxt(data_dir / f'out_{snapshot}.list')
    # get the n_halos heaviest ones
    data = data[np.argsort(data[:,21])[-num_halos:]]
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

def read_halos_consistent_trees(data_dir, scale_factor=1., num_halos=5000, host_halos=True, keep_columns = ["x", "y", "z", "v_x", "v_y", "v_z", "J_x", "J_y", "J_z", "M200c", "Rvir"]):
    column_names = [
        "scale",
        "id",
        "desc_scale",
        "desc_id",
        "num_prog",
        "pid",
        "upid",
        "desc_pid",
        "phantom",
        "sam_Mvir",
        "Mvir",
        "Rvir",
        "rs",
        "vrms",
        "mmp?",
        "scale_of_last_MM",
        "vmax",
        "x",
        "y",
        "z",
        "v_x",
        "v_y",
        "v_z",
        "J_x",
        "J_y",
        "J_z",
        "Spin",
        "Breadth_first_ID",
        "Depth_first_ID",
        "Tree_root_ID",
        "Orig_halo_ID",
        "Snap_idx",
        "Next_coprogenitor_depthfirst_ID",
        "Last_progenitor_depthfirst_ID",
        "Last_mainleaf_depthfirst_ID",
        "Tidal_Force",
        "Tidal_ID",
        "Rs_Klypin",
        "Mvir_all",
        "M200b",
        "M200c",
        "M500c",
        "M2500c",
        "Xoff",
        "Voff",
        "Spin_Bullock",
        "b_to_a",
        "c_to_a",
        "Ax",
        "Ay",
        "Az",
        "b_to_a_500c",
        "c_to_a_500c",
        "Ax_500c",
        "Ay_500c",
        "Az_500c",
        "T_U",
        "M_pe_Behroozi",
        "M_pe_Diemer",
        "Halfmass_Radius",
        "Macc",
        "Mpeak",
        "Vacc",
        "Vpeak",
        "Halfmass_Scale",
        "Acc_Rate_Inst",
        "Acc_Rate_100Myr",
        "Acc_Rate_1Tdyn",
        "Acc_Rate_2Tdyn",
        "Acc_Rate_Mpeak",
        "Acc_Log_Vmax_Inst",
        "Acc_Log_Vmax_1Tdyn",
        "Mpeak_Scale",
        "Acc_Scale",
        "First_Acc_Scale",
        "First_Acc_Mvir",
        "First_Acc_Vmax",
        "Vmax_Mpeak",
        "Tidal_Force_Tdyn",
        "Log_Vmax_Vmax_max_Tdyn_TMpeak",
        "Time_to_future_merger",
        "Future_merger_MMP_ID",
    ]
    filename = data_dir / f"hlists/hlist_{scale_factor:.5f}.list"
    df = pd.read_csv(
        filename, delim_whitespace=True, comment="#", names=column_names, header=None
    )
    if host_halos:
        df = df[df["upid"] == -1]
    df= df.sort_values(by='M200c', ascending=False)
    df = df.iloc[:num_halos]
    if keep_columns is not None:
        df = df[keep_columns]
    return df

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


def write_tfrecords(tfrecord_file, indices, data_dir, snapshot, n_halos, use_consistent_trees):
    print(f"Starting to write TFRecord: {tfrecord_file}")
    with tf.io.TFRecordWriter(tfrecord_file) as writer:
        for idx in tqdm(indices, desc=f'Writing {tfrecord_file}'):
            try:
                if use_consistent_trees:
                    features = read_halos_consistent_trees(data_dir / f'{idx}', snapshot, num_halos=n_halos)
                else:
                    features = read_halos(data_dir / f'{idx}', snapshot, num_halos=n_halos)
                tpcf_features = compute_tpcf(features[['x', 'y', 'z']].values)
                params = params_df.iloc[idx].values
                example = serialize_example(features, tpcf_features, params, params_names)
                writer.write(example)
            except Exception as e:
                print(f"Error writing index {idx} to {tfrecord_file}: {e}")

def process_data_to_tfrecords(params_df, data_dir, tfrecords_path, num_tfrecords=20, use_consistent_trees=False,num_tfrecords_val=1, num_tfrecords_test=1, snapshot=10, n_halos=5000):
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
        write_tfrecords(tfrecord_file, indices, data_dir, snapshot, n_halos, use_consistent_trees=use_consistent_trees,)

    for i in range(num_tfrecords_val):
        tfrecord_file = os.path.join(tfrecords_path, f'halos_val_{i + 1}.tfrecord')
        indices = range((num_tfrecords + i) * files_per_tfrecord, min((num_tfrecords + i + 1) * files_per_tfrecord, num_files))
        write_tfrecords(tfrecord_file, indices, data_dir, snapshot, n_halos, use_consistent_trees=use_consistent_trees,)

    for i in range(num_tfrecords_test):
        tfrecord_file = os.path.join(tfrecords_path, f'halos_test_{i + 1}.tfrecord')
        indices = range((num_tfrecords + num_tfrecords_val + i) * files_per_tfrecord, min((num_tfrecords + num_tfrecords_val + i + 1) * files_per_tfrecord, num_files))
        write_tfrecords(tfrecord_file, indices, data_dir, snapshot, n_halos, use_consistent_trees=use_consistent_trees,)


if __name__ == '__main__':
    use_consistent_trees = True
    if use_consistent_trees:
        data_dir = Path('/quijote_bsq_consistent_trees/')
        tfrecords_path = Path('/quijote_tfrecords/')
        snapshot = 1.
    else:
        data_dir = Path('/quijote_bsq/')
        tfrecords_path = Path('/quijote_tfrecords/')
        snapshot = 10
    params_names = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
    params_df = read_cosmologies()

    process_data_to_tfrecords(
        params_df,
        num_tfrecords=50,
        use_consistent_trees=use_consistent_trees,
        data_dir = data_dir,
        tfrecords_path = tfrecords_path,
        snapshot=snapshot,
    )
