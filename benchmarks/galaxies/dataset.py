from pathlib import Path
import jax.numpy as np
import pandas as pd
import jax
import jraph
from typing import Callable, List, Optional, Tuple, Sequence, Union, NamedTuple
from tqdm.notebook import tqdm
from pycorr import (
    TwoPointCorrelationFunction,
)  # Can install it here https://github.com/cosmodesi/pycorr


class GalaxyDataset:
    """Dark matter halo dataset class."""

    def __init__(
        self,
        data_dir,
        use_pos=True,
        use_vel=False,
        use_mass=False,
        use_tpcf="none",
    ):

        self.data_dir = data_dir
        halos_train, halos_val, halos_test = self.load_node_feats(
            data_dir, use_pos, use_vel, use_mass
        )
        self.halos_train, self.halos_val, self.halos_test = self.normalize(
            halos_train, halos_val, halos_test
        )

        if use_tpcf != "none":
            tpcf_dir = Path(__file__).parent / "tpcfs/"
            if not tpcf_dir.is_dir():
                tpcfs_train = self.generate_tpcfs(self, halos_train, data_str="train")
                tpcfs_val = self.generate_tpcfs(self, halos_val, data_str="val")
                tpcfs_test = self.generate_tpcfs(self, halos_test, data_str="test")
            else:
                tpcfs_train = np.load(tpcf_dir / "tpcfs_train.npy")
                tpcfs_val = np.load(tpcf_dir / "tpcfs_val.npy")
                tpcfs_test = np.load(tpcf_dir / "tpcfs_test.npy")

            if use_tpcf == "small":
                tpcf_idx = list(range(8))
            elif use_tpcf == "large":
                tpcf_idx = list(range(15, 24))
            else:
                tpcf_idx = list(range(24))

            tpcfs_train = tpcfs_train[:, tpcf_idx]
            tpcfs_val = tpcfs_val[:, tpcf_idx]
            tpcfs_test = tpcfs_test[:, tpcf_idx]

            self.tpcfs_train, self.tpcfs_val, self.tpcfs_test = tpcfs_train, tpcfs_val, tpcfs_test
            self.tpcfs_train, self.tpcfs_val, self.tpcfs_test = self.normalize(tpcfs_train, tpcfs_val, tpcfs_test)

        self.targets_train, self.targets_val, self.targets_test = self.load_targets(data_dir)
        self.targets_train, self.targets_val, self.targets_test = self.normalize(
            self.targets_train, self.targets_val, self.targets_test
        )

  
    def load_node_feats(self, data_dir, use_pos, use_vel, use_mass, n_nodes=5000) -> Tuple[np.ndarray, ...]:
        halos_train = np.load(data_dir / 'train_halos.npy')
        halos_val = np.load(data_dir / 'val_halos.npy')
        halos_test = np.load(data_dir / 'test_halos.npy')
        
        feat_idx = list(range(3))*use_pos + list(range(3, 6))*use_vel + [7]*use_mass
        
        halos_train =  halos_train[:, :n_nodes, feat_idx] / 1000.
        halos_val =  halos_val[:, :n_nodes, feat_idx] / 1000.
        halos_test =  halos_test[:, :n_nodes, feat_idx] / 1000.
        
        self.halos_mean = halos_train.mean((0,1))
        self.halos_std = halos_train.std((0,1))
        
        return halos_train, halos_val, halos_test

    def load_targets(self, data_dir) -> Tuple[np.ndarray, ...]:
        cosmology_train = pd.read_csv(data_dir / f"train_cosmology.csv")
        cosmology_val = pd.read_csv(data_dir / f"val_cosmology.csv")
        cosmology_test = pd.read_csv(data_dir / f"test_cosmology.csv")

        omega_m_train = np.array(cosmology_train["Omega_m"].values)[:, None]
        omega_m_val = np.array(cosmology_val["Omega_m"].values)[:, None]
        omega_m_test = np.array(cosmology_test["Omega_m"].values)[:, None]

        sigma_8_train = np.array(cosmology_train["sigma_8"].values)[:, None]
        sigma_8_val = np.array(cosmology_val["sigma_8"].values)[:, None]
        sigma_8_test = np.array(cosmology_test["sigma_8"].values)[:, None]

        targets_train = np.concatenate([omega_m_train, sigma_8_train], axis=-1)
        targets_val = np.concatenate([omega_m_val, sigma_8_val], axis=-1)
        targets_test = np.concatenate([omega_m_test, sigma_8_test], axis=-1)

        return targets_train, targets_val, targets_test
        
    
    def normalize(self, feats_train, feats_val, feats_test, eps=1e-8) -> Tuple[np.ndarray, ...]:
        axes_except_last = tuple(range(feats_train.ndim - 1))
        feats_mean = feats_train.mean(axes_except_last, keepdims=True)
        feats_std = feats_train.std(axes_except_last, keepdims=True)

        feats_train = (feats_train - feats_mean) / (feats_std + eps)
        feats_val = (feats_val - feats_mean) / (feats_std + eps)
        feats_test = (feats_test - feats_mean) / (feats_std + eps)

        return feats_train, feats_val, feats_test

    def generate_tpcfs(self, halos, data_str="train"):
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
                    n_threads=2,
                    boxsize=box_size,
                    los="z",
                )(ells=[0])[0]
            )
        tpcfs = np.stack(tpcfs)
        np.save(f"tpcfs_{data_str}.npy", tpcfs)
