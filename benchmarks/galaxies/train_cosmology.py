import flax
from flax.training.train_state import TrainState
from functools import partial
import flax.linen as nn
from flax.training.early_stopping import EarlyStopping
from flax.training import checkpoints
import optax

import jax
from jax.experimental.sparse import BCOO

from tqdm import trange, tqdm
from time import sleep
from datetime import datetime
from typing import Dict, Callable
import dataclasses

import argparse
from pathlib import Path
import jax.numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import json
import os

import jraph
import e3nn_jax as e3nn

import sys
sys.path.append("../../")

from models.utils.graph_utils import build_graph, get_apply_pbc
from models.utils.irreps_utils import balanced_irreps
from models.utils.equivariant_graph_utils import get_equivariant_graph

from models.mlp import MLP
from models.gnn import GNN
from models.egnn import EGNN
from models.segnn import SEGNN
from models.nequip import NequIP
from models.diffpool import DiffPool

from benchmarks.galaxies.dataset import get_halo_dataset

MLP_PARAMS = {
    "feature_sizes": [128, 128, 128, 2],
}

GNN_PARAMS = {
    "d_hidden": 128,
    "n_layers": 3,
    "message_passing_steps": 3,
    "message_passing_agg": "mean",
    "activation": "gelu",
    "norm": "none",
    "task": "graph",
    "n_outputs": 2,
    "readout_agg": "mean",
    "mlp_readout_widths": (4, 2, 2),
    "position_features": True,
    "residual": True,
}

EGNN_PARAMS = {
    "message_passing_steps": 3,
    "d_hidden": 128,
    "n_layers": 3,
    "activation": "gelu",
    "soft_edges": False,
    "positions_only": True,
    "tanh_out": False,
    "decouple_pos_vel_updates": True,
    "message_passing_agg": "mean",
    "readout_agg": "mean",
    "mlp_readout_widths": [4, 2, 2],
    "task": "graph",
    "n_outputs": 2,
}

DIFFPOOL_PARAMS = {
    "n_downsamples": 2,
    "d_downsampling_factor": 5,
    "k": 10,
    "gnn_kwargs": GNN_PARAMS,
    "combine_hierarchies_method": "mean",
    "use_edge_features": True,
    "task": "graph",
    "mlp_readout_widths": [8, 2, 2],
}

POINTNET_PARAMS = {
    "n_downsamples": 2,
    "d_downsampling_factor": 5,
    "radius": 0.2,
    "gnn_kwargs": GNN_PARAMS,
    "combine_hierarchies_method": "mean",
    "use_edge_features": True,
    "task": "graph",
    "mlp_readout_widths": [8, 2, 2],
}

SEGNN_PARAMS = {
    "d_hidden": 128,
    "n_layers": 3,
    "message_passing_steps": 3,
    "message_passing_agg": "mean",
    "scalar_activation": "gelu",
    "gate_activation": "sigmoid",
    "task": "graph",
    "n_outputs": 2,
    "output_irreps": e3nn.Irreps("1x0e"),
    "readout_agg": "mean",
    "mlp_readout_widths": (4, 2, 2),
    "l_max_hidden": 2,
    "hidden_irreps": None,
    "residual": True,
}

NEQUIP_PARAMS = {
    "d_hidden": 128,
    "l_max":1,
    "sphharm_norm": 'integral',
    "irreps_out": e3nn.Irreps("1x0e"),
    "message_passing_steps": 3,
    "n_layers": 3,
    "message_passing_agg": "mean",
    "readout_agg": "mean",
    "mlp_readout_widths": [4, 2, 2],
    "task": "graph",
    "n_outputs": 2,
}

class GraphWrapper(nn.Module):
    model_name: str
    param_dict: Dict
    apply_pbc: Callable = None
    n_radial_basis: int = 64
    r_max: float = 0.6

    @nn.compact
    def __call__(self, x):
        if self.model_name == "DeepSets":
            raise NotImplementedError
        elif self.model_name == "MLP":
            return jax.vmap(MLP(**self.param_dict))(x.globals)
        elif self.model_name == "GNN":
            return jax.vmap(GNN(**self.param_dict))(x)
        elif self.model_name == "EGNN":
            return jax.vmap(EGNN(**self.param_dict))(x)
        elif self.model_name == "DiffPool":
            return jax.vmap(DiffPool(**self.param_dict))(x)
        elif self.model_name == "PointNet":
            return jax.vmap(PointNet(**self.param_dict))(x)
        elif self.model_name == "EquivariantTransformer":
            #             pos = e3nn.IrrepsArray("1o", x.nodes[..., :3])
            #             feat = e3nn.IrrepsArray("1o", x.nodes[..., 3:])

            #             return jax.vmap(EquivariantTransformer(**self.param_dict))(pos, feat, x.senders, x.receivers,)
            raise NotImplementedError
        elif self.model_name == "SEGNN":
            positions = e3nn.IrrepsArray("1o", x.nodes[..., :3])
        
            if x.nodes.shape[-1] == 3:
                nodes = e3nn.IrrepsArray("1o", x.nodes[..., :])
                velocities = None
            else:
                nodes = e3nn.IrrepsArray("1o + 1o", x.nodes[..., :])
                velocities = e3nn.IrrepsArray("1o", x.nodes[..., 3:6])

            
            st_graph = get_equivariant_graph(
                node_features=nodes,
                positions=positions,
                velocities=velocities,
                steerable_velocities=False,
                senders=x.senders,
                receivers=x.receivers,
                n_node=x.n_node,
                n_edge=x.n_edge,
                globals=x.globals,
                edges=None,
                lmax_attributes=self.param_dict['l_max_hidden'],
                apply_pbc=self.apply_pbc,
                n_radial_basis=self.n_radial_basis,
                r_max=self.r_max,
            )
            
            return jax.vmap(SEGNN(**self.param_dict))(st_graph)

        elif self.model_name == "NequIP":
            if x.nodes.shape[-1] == 3:
                ones = np.ones(x.nodes[..., :].shape[:2] + (1,))
                nodes = np.concatenate([x.nodes[..., :], x.nodes[..., :], ones], axis=-1)
                nodes = e3nn.IrrepsArray("1o + 1o + 1x0e", nodes)
            else:
                nodes = e3nn.IrrepsArray("1o + 1o + 1x0e", x.nodes[..., :])
            
            graph = jraph.GraphsTuple(
                n_node=x.n_node,
                n_edge=x.n_edge,
                edges=None,
                globals=x.globals,
                nodes=nodes, 
                senders=x.senders,
                receivers=x.receivers)
            
            return jax.vmap(NequIP(**self.param_dict))(graph)
        else:
            raise ValueError("Please specify a valid model name.")


def loss_mse(pred_batch, cosmo_batch):
    # return np.mean((pred_batch - cosmo_batch) ** 2)
    return np.mean((pred_batch - cosmo_batch) ** 2, axis=0)


@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(4,5,6,7)
)
def train_step(state, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, r_max, use_3d_distances):
    
    halo_graph = build_graph(
        halo_batch, 
        tpcfs_batch, 
        k=K, 
        apply_pbc=apply_pbc, 
        use_edges=True, 
        n_radial_basis=n_radial_basis,
        r_max=r_max,
        use_3d_distances=use_3d_distances
    )

    def loss_fn(params):
        outputs = state.apply_fn(params, halo_graph)
        if len(outputs.shape) > 2:
            outputs = np.squeeze(outputs, axis=-1)
        loss = loss_mse(outputs, y_batch)
        return loss.mean()

    # Get loss, grads, and update state
    avg_loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)

    outputs = state.apply_fn(state.params, halo_graph)
    if len(outputs.shape) > 2:
            outputs = np.squeeze(outputs, axis=-1)
    individual_losses = jax.lax.stop_gradient(loss_mse(outputs, y_batch))
    metrics = {"loss": jax.lax.pmean(individual_losses, "batch")}

    return new_state, metrics

@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(4,5,6,7)
)
def eval_step(state, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, r_max, use_3d_distances):
    # Build graph
    halo_graph = build_graph(
        halo_batch, 
        tpcfs_batch, 
        k=K, 
        apply_pbc=apply_pbc, 
        use_edges=True, 
        n_radial_basis=n_radial_basis,
        r_max=r_max,
        use_3d_distances=use_3d_distances
    )

    outputs = state.apply_fn(state.params, halo_graph)
    if len(outputs.shape) > 2:
        outputs = np.squeeze(outputs, axis=-1)
    individual_losses = jax.lax.stop_gradient(loss_mse(outputs, y_batch))

    return outputs, {"loss": jax.lax.pmean(individual_losses, "batch")} 


def split_batches(num_local_devices, halo_batch, y_batch, tpcfs_batch):
    halo_batch = jax.tree.map(
        lambda x: np.split(x, num_local_devices, axis=0), halo_batch
    )
    y_batch = jax.tree.map(
        lambda x: np.split(x, num_local_devices, axis=0), y_batch
    )
    halo_batch, y_batch = np.array(halo_batch), np.array(y_batch)

    if tpcfs_batch is not None:
        tpcfs_batch = jax.tree.map(
            lambda x: np.split(x, num_local_devices, axis=0), tpcfs_batch
        )
        tpcfs_batch = np.array(tpcfs_batch)

    return halo_batch, y_batch, tpcfs_batch


def run_expt(
    model_name,
    feats,
    param_dict,
    data_dir,
    target = ["Omega_m", "sigma_8"],
    use_pbcs=True,
    boxsize=1000.,
    use_edges=True,
    n_radial_basis=64,
    r_max=0.6,
    use_3d_distances=False,
    use_tpcf="none",
    n_steps=1000,
    batch_size=32,
    n_train=1248, 
    n_val=512, 
    n_test=512, 
    learning_rate=3e-4,
    weight_decay=1e-5,
    eval_every=500,
    get_node_reps=False,
    plotting=True,
    save_model=True
):
    num_local_devices = jax.local_device_count()
    
    # Create experiment directory
    experiments_base_dir = Path(__file__).parent / "experiments/"
    d_hidden = param_dict["d_hidden"]
    experiment_id = (
        f"{model_name}_N={n_train}_tpcf=" + use_tpcf
    )
    current_experiment_dir = experiments_base_dir / experiment_id
    current_experiment_dir.mkdir(parents=True, exist_ok=True)
    
    print('Loading dataset...')
    
    if feats == 'pos':
        features = ['x', 'y', 'z']
    elif feats == 'all':
        features = ['x', 'y', 'z', 'v_x', 'v_y', 'v_z']
    else:
        raise NotImplementedError

    train_dataset, n_train, _, std, _, _ = get_halo_dataset(batch_size=batch_size,  # Batch size
                                                            num_samples=n_train,  # If not None, will only take a subset of the dataset
                                                            split='train',  # 'train', 'val', 'test'
                                                            standardize=True,  # If True, will standardize the features
                                                            return_mean_std=True,  # If True, will return (dataset, num_total, mean, std, mean_params, std_params), else (dataset, num_total)
                                                            seed=42,  # Random seed
                                                            features=features,  # Features to include
                                                            params=target,  # Parameters to include
                                                            include_tpcf=True,
                                                            tfrecords_path=data_dir
                                                            )
    std = std.numpy()
    train_iter = iter(train_dataset)
    halo_train, y_train, tpcfs_train = next(train_iter)
    halo_train, y_train, tpcfs_train = halo_train.numpy(), y_train.numpy(), tpcfs_train.numpy()

    val_dataset, n_val = get_halo_dataset(batch_size=batch_size,  
                                           num_samples=n_val, 
                                           split='val',
                                           standardize=True, 
                                           return_mean_std=False,  
                                           seed=42,
                                           features=features, 
                                           params=target,
                                           include_tpcf=True,
                                           tfrecords_path=data_dir
                                        )

    test_dataset, n_test = get_halo_dataset(batch_size=batch_size,  
                                           num_samples=n_test, 
                                           split='test',
                                           standardize=True, 
                                           return_mean_std=False,  
                                           seed=42,
                                           features=features, 
                                           params=target,
                                           include_tpcf=True,
                                           tfrecords_path=data_dir
                                        )

    print('Train-Val-Test split:', n_train, n_val, n_test)

    if use_tpcf == "small":
        tpcf_idx = list(range(6))
    elif use_tpcf == "large":
        tpcf_idx = list(range(13, 24))
    else:
        tpcf_idx = list(range(24))

    tpcfs_train = tpcfs_train[:, tpcf_idx]
    if use_tpcf == 'none':
        init_tpcfs = None
    else:
        init_tpcfs = tpcfs_train[:2]

    apply_pbc = get_apply_pbc(std=std / boxsize) if use_pbcs else None

    if model_name in ['EGNN', 'PointNet']:
        param_dict['apply_pbc'] = apply_pbc

    if model_name == 'EGNN':
        param_dict['n_radial_basis'] = n_radial_basis
        param_dict['r_max'] = r_max
    if model_name == 'NequIP':
        param_dict['n_radial_basis'] = n_radial_basis
        param_dict['r_cutoff'] = r_max


    graph = build_graph(
        halo_train[:2],
        init_tpcfs,
        k=K,
        apply_pbc=apply_pbc,
        use_edges=use_edges,
        n_radial_basis=n_radial_basis,
        r_max=r_max,
        use_3d_distances=use_3d_distances
    )

    if get_node_reps:
        param_dict["get_node_reps"] = True

    model = GraphWrapper(model_name, param_dict, apply_pbc, n_radial_basis, r_max)
    key = jax.random.PRNGKey(0)
    out, params = model.init_with_output(key, graph)
    
    print(f"Number of parameters: {sum([p.size for p in jax.tree_leaves(params)])}")    
    
    # Define train state and replicate across devices
    replicate = flax.jax_utils.replicate
    unreplicate = flax.jax_utils.unreplicate

    lr = optax.cosine_decay_schedule(learning_rate, 2000)
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    pstate = replicate(state)
    # early_stop = EarlyStopping(min_delta=1e-4, patience=10)
    
    print('Training...')
    losses = []
    val_losses = []
    best_val = 1e10
    best_vals = None
    best_state = None
    test_loss_ckp = None
    with trange(n_steps, ncols=120) as steps:
        train_iter = iter(train_dataset)
        for step in steps:
            # Training
            halo_batch, y_batch, tpcfs_batch = next(train_iter)
            halo_batch, y_batch, tpcfs_batch = halo_batch.numpy(), y_batch.numpy(), tpcfs_batch.numpy()
            if use_tpcf != 'none':
                tpcfs_batch = tpcfs_batch[:, tpcf_idx]
            else:
                tpcfs_batch = None
            
            halo_batch, y_batch, tpcfs_batch = split_batches(
                num_local_devices, halo_batch, y_batch, tpcfs_batch
            )
            pstate, metrics = train_step(
                pstate, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, r_max, use_3d_distances
            )
            train_loss = unreplicate(metrics["loss"])

            if step % eval_every == 0:
                # Validation
                val_iter = iter(val_dataset)
                running_val_loss = np.zeros((1,2))
                count = 0
                for halo_val_batch, y_val_batch, tpcfs_val_batch in val_iter:
                    halo_val_batch, y_val_batch, tpcfs_val_batch = halo_val_batch.numpy(), y_val_batch.numpy(), tpcfs_val_batch.numpy()
                    if use_tpcf != 'none':
                        tpcfs_val_batch = tpcfs_val_batch[:, tpcf_idx]
                    else:
                        tpcfs_val_batch = None

                    halo_val_batch, y_val_batch, tpcfs_val_batch = split_batches(
                        num_local_devices, halo_val_batch, y_val_batch, tpcfs_val_batch
                    )
                    outputs, metrics = eval_step(
                        pstate, halo_val_batch, y_val_batch, tpcfs_val_batch, apply_pbc, n_radial_basis, r_max, use_3d_distances
                    )
                    val_loss = unreplicate(metrics["loss"])
                    running_val_loss += val_loss
                    count +=1 
                avg_val_loss = running_val_loss/count

                if avg_val_loss.mean() < best_val:
                    best_val = avg_val_loss.mean()
                    best_vals = avg_val_loss
                    best_state = pstate
                    tag = " (best)"

                    test_iter = iter(test_dataset)
                    running_test_loss = np.zeros((1,2))
                    count = 0
                    for halo_test_batch, y_test_batch, tpcfs_test_batch in test_iter:
                        halo_test_batch, y_test_batch, tpcfs_test_batch = halo_test_batch.numpy(), y_test_batch.numpy(), tpcfs_test_batch.numpy()
                        if use_tpcf != 'none':
                            tpcfs_test_batch = tpcfs_test_batch[:, tpcf_idx]
                        else:
                            tpcfs_test_batch = None

                        halo_test_batch, y_test_batch, tpcfs_test_batch = split_batches(
                            num_local_devices, halo_test_batch, y_test_batch, tpcfs_test_batch
                        )
                        test_outputs, metrics = eval_step(
                            pstate, halo_test_batch, y_test_batch, tpcfs_test_batch, apply_pbc, n_radial_basis, r_max, use_3d_distances
                        )
                        test_loss = unreplicate(metrics["loss"])
                        running_test_loss += test_loss
                        count += 1
                    avg_test_loss = running_test_loss/count

                    test_loss_ckp = avg_test_loss
                else:
                    tag = ""

            
            steps.set_postfix_str('train_loss: {:.5f}, avg_val_loss: {:.5f}, avg_ckp_test_loss: {:.5F}'.format(train_loss.mean(),
                                                                                                   avg_val_loss.mean(),
                                                                                                   test_loss_ckp.mean()))
            losses.append(train_loss)
            val_losses.append(avg_val_loss)

            # early_stop = early_stop.update(avg_val_loss)
            # if early_stop.should_stop:
            #     print(f'Met early stopping criteria, breaking at epoch {step}')
            # break
            
        print(
            "Training done.\n"
            f"Final checkpoint test loss {test_loss_ckp}.\n"
        )
        
    if plotting:
        y_test_batch = y_test_batch.reshape((batch_size, -1))
        test_outputs = test_outputs.reshape((batch_size, -1))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), dpi=300)
        ax1.scatter(np.vstack(y_test_batch[:, 0]), test_outputs[:, 0], color='firebrick')
        ax1.plot(np.vstack(y_test_batch[:, 0]), np.vstack(y_test_batch[:, 0]), color='gray')
        ax1.set_title('Omega_m Predictions')
        ax1.set_xlabel('True')
        ax1.set_ylabel('Predicted')

        ax2.scatter(np.vstack(y_test_batch[:, 1]), test_outputs[:, 1], color='firebrick')
        ax2.plot(np.vstack(y_test_batch[:, 1]), np.vstack(y_test_batch[:, 1]), color='gray')
        ax2.set_title('sigma_8 Predictions')
        ax2.set_xlabel('True')
        fig.savefig(current_experiment_dir / "y_preds.png")
        
    np.save(current_experiment_dir / "train_losses.npy", losses)
    np.save(current_experiment_dir / "val_losses.npy", val_losses)

    if save_model:
        device_cpu = jax.devices('cpu')[0]
        with jax.default_device(device_cpu):
            loss_dict = {
                "train_loss": np.array(losses),
                'val_loss': np.array(val_losses),
                'test_loss': np.array(test_loss_ckp),
            }
            best_state = unreplicate(best_state)
            checkpoints.save_checkpoint(
                ckpt_dir=str(current_experiment_dir),
                target=best_state,
                step=step,
                overwrite=True,
            )
            with open(f"{current_experiment_dir}/loss_dict.pkl", "wb") as f:
                pickle.dump(loss_dict, f)
            with open(f"{current_experiment_dir}/results.txt", "w") as text_file:
                text_file.write(f"Train loss: {losses[-1]}\n")
                text_file.write(f"Val loss: {best_vals}\n")
                text_file.write(f"Test loss: {test_loss_ckp}\n")
        
    
def main(model, feats, lr, decay, steps, batch_size, n_train, use_tpcf, k, data_dir):
    if model == 'MLP':
        MLP_PARAMS['d_hidden'] = MLP_PARAMS['feature_sizes'][0]
        params = MLP_PARAMS
    elif model == "GNN":
        params = GNN_PARAMS
    elif model == "EGNN":
        params = EGNN_PARAMS
    elif model == "SEGNN":
        params = SEGNN_PARAMS
    elif model == "NequIP":
        params = NEQUIP_PARAMS
    elif model == "DiffPool":
        DIFFPOOL_PARAMS["gnn_kwargs"] = {
            "d_hidden": 64,
            "n_layers": 3,
            "message_passing_steps": 4,
            "task": "node",
        }
        DIFFPOOL_PARAMS["d_hidden"] = DIFFPOOL_PARAMS["gnn_kwargs"]["d_hidden"]
        params = DIFFPOOL_PARAMS
    elif model == "PointNet":
        POINTNET_PARAMS["gnn_kwargs"] = {
            "d_hidden": 64,
            "n_layers": 4,
            "message_passing_steps": 3,
            "task": "node",
        }
        POINTNET_PARAMS["d_hidden"] = POINTNET_PARAMS["gnn_kwargs"]["d_hidden"]
        params = POINTNET_PARAMS
    else:
        raise NotImplementedError

    run_expt(
        model,
        feats,
        params,
        data_dir,
        learning_rate=lr,
        weight_decay=decay,
        n_steps=steps,
        batch_size=batch_size,
        n_train=n_train,
        use_tpcf=use_tpcf
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model", default="GNN")
    parser.add_argument(
        "--feats", help="Features to use", default="pos", choices=["pos", "all"]
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=3e-4)
    parser.add_argument("--decay", type=float, help="Weight decay", default=1e-5)
    parser.add_argument("--steps", type=int, help="Number of steps", default=5000)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument("--n_train", type=int, help="Number of training samples", default=1248)
    parser.add_argument(
        "--use_tpcf",
        type=str,
        help="Which tpcf features to include",
        default="none",
        choices=["none", "small", "large", "all"],
    )
    parser.add_argument(
        "--k", type=int, help="Number of neighbors for kNN graph", default=10
    )
    parser.add_argument("--data_dir", type=str, help="Path to Quijote records", default='quijote_records')

    args = parser.parse_args()

    K = args.k

    main(**vars(args))
    