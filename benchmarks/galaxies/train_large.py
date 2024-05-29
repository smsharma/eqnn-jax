import flax
from flax.training.train_state import TrainState
from functools import partial
import flax.linen as nn
from flax.training.early_stopping import EarlyStopping
import optax

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
import jax
import jraph
from jax.experimental.sparse import BCOO

import e3nn_jax as e3nn

import sys
sys.path.append("../../")

from models.utils.graph_utils import build_graph, get_apply_pbc
from models.utils.irreps_utils import balanced_irreps
from models.utils.equivariant_graph_utils import get_equivariant_graph

from models.mlp import MLP
from models.gnn import GNN
from models.egnn import EGNN
# from models.equivariant_transformer import EquivariantTransformer
from models.segnn import SEGNN
from models.nequip import NequIP
from models.diffpool import DiffPool

from benchmarks.galaxies.dataset_large import get_halo_dataset

MLP_PARAMS = {
    "feature_sizes": [128, 128, 128, 1],
}

GNN_PARAMS = {
    "d_hidden": 128,
    "n_layers": 3,
    "message_passing_steps": 3,
    "message_passing_agg": "mean",
    "activation": "gelu",
    "norm": "layer",
    "task": "graph",
    "n_outputs": 2,
    "readout_agg": "mean",
    "mlp_readout_widths": (4, 2, 2),
    "position_features": True,
    "residual": False,
}

EGNN_PARAMS = {
    "message_passing_steps": 2,
    "d_hidden": 64,
    "n_layers": 3,
    "activation": "gelu",
    "soft_edges": True,
    "positions_only": True,
    "tanh_out": False,
    "n_radial_basis": 32,
    "r_max": 0.3,
    "decouple_pos_vel_updates": True,
    "message_passing_agg": "mean",
    "readout_agg": "mean",
    "mlp_readout_widths": [8, 2, 2],
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
    "l_max_hidden": 1,
    "hidden_irreps": None,
    "residual": False,
}

NEQUIP_PARAMS = {
    "d_hidden": 128,
    "l_max_hidden":1,
    "l_max_attr":1,
    "sphharm_norm": 'component',
    "irreps_out": e3nn.Irreps("1x0e"),
    "message_passing_steps": 3,
    "n_layers": 3,
    "message_passing_agg": "mean",
    "readout_agg": "mean",
    "mlp_readout_widths": [4, 2, 2],
    "task": "graph",
    "n_outputs": 2,
    "n_radial_basis": 64,
    "r_cutoff": 0.6,
    "sphharm_norm": "component"
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
                velocities=None,
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
    return np.mean((pred_batch - cosmo_batch) ** 2)


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
        # outputs, assignments = state.apply_fn(params, halo_graph)
        if len(outputs.shape) > 2:
            outputs = np.squeeze(outputs, axis=-1)
        loss = loss_mse(outputs, y_batch)
        return loss

    # Get loss, grads, and update state
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}

    # outputs, assignments = state.apply_fn(state.params, halo_graph, return_assignments=True)

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

    # outputs, assignments = state.apply_fn(state.params, halo_graph)
    if len(outputs.shape) > 2:
        outputs = np.squeeze(outputs, axis=-1)
    loss = jax.lax.stop_gradient(loss_mse(outputs, y_batch))

    return outputs, {"loss": jax.lax.pmean(loss, "batch")} #, assignments


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

def generator_to_array(iterator, n, batch_size):
    x_train, params_train = [], []
    for _ in tqdm(range(n // batch_size)):
        x, params = next(iterator)
        x_train.append(np.array(x))
        params_train.append(np.array(params))

    x_train = np.concatenate(x_train, axis=0)
    params_train = np.concatenate(params_train, axis=0)
    return x_train, params_train

def run_expt(
    model_name,
    feats,
    target,
    param_dict,
    data_dir,
    use_pbc=True,
    use_edges=True,
    n_radial_basis=64,
    r_max=0.6,
    use_3d_distances=False,
    use_tpcf="none",
    radius=None,
    n_steps=1000,
    batch_size=32,
    n_train=2000, #31550
    n_val=200, #631
    n_test=200, #587
    learning_rate=5e-5,
    weight_decay=1e-5,
    eval_every=200,
    get_node_reps=False,
    plotting=True,
):
    num_local_devices = jax.local_device_count()
    
    # Create experiment directory
    experiments_base_dir = Path(__file__).parent / "experiments/"
    d_hidden = param_dict["d_hidden"]
    experiment_id = (
        f"{model_name}_{feats}_{batch_size}b_{n_steps}s_{d_hidden}d_{K}k_{n_radial_basis}rbf"
        + "tpcf-" + use_tpcf
        + f"_{radius}r"
    )

    current_experiment_dir = experiments_base_dir / experiment_id
    current_experiment_dir.mkdir(parents=True, exist_ok=True)

    print('Loading dataset...')
    if feats == 'pos':
        features = ['x', 'y', 'z']
    elif feats == 'all':
        features = ['x', 'y', 'z', 'vx', 'vy', 'vz']
    else:
        raise NotImplementedError

    target = ['Omega_m', 'sigma_8']
        
    train_dataset, n_train, mean, std, _, _ = get_halo_dataset(batch_size=batch_size,  # Batch size
                                                                num_samples=n_train,  # If not None, will only take a subset of the dataset
                                                                split='train',  # 'train', 'val', 'test'
                                                                standardize=True,  # If True, will standardize the features
                                                                return_mean_std=True,  # If True, will return (dataset, num_total, mean, std, mean_params, std_params), else (dataset, num_total)
                                                                seed=42,  # Random seed
                                                                features=features,  # Features to include
                                                                params=target,  # Parameters to include
                                                            )
    mean, std = mean.numpy(), std.numpy()
    norm_dict = {'mean': mean, 'std': std}
    train_iter = iter(train_dataset)
    halo_train, y_train = generator_to_array(train_iter, n_train, batch_size)

    val_dataset, n_val = get_halo_dataset(batch_size=None,  
                                           num_samples=n_val, 
                                           split='val',
                                           standardize=True, 
                                           return_mean_std=False,  
                                           seed=42,
                                           features=features, 
                                           params=target,
                                        )
    val_iter = iter(val_dataset)
    halo_val, y_val = next(val_iter)
    val_batch_size = n_val
    # halo_val, y_val = generator_to_array(val_iter, n_val, batch_size)


    test_dataset, n_test = get_halo_dataset(batch_size=None,  
                                           num_samples=n_test, 
                                           split='test',
                                           standardize=True, 
                                           return_mean_std=False,  
                                           seed=42,
                                           features=features, 
                                           params=target,
                                        )
    test_iter = iter(test_dataset)
    halo_test, y_test = next(test_iter)
    test_batch_size = n_test
    # halo_test, y_test = generator_to_array(test_iter, n_test, batch_size)

    print('Train-Val-Test split:', n_train, n_val, n_test)
    
    tpcfs_train = None
    tpcfs_val = None
    tpcfs_test = None
    init_tpcfs = None
    # TO DO: implement tpcfs

    apply_pbc = get_apply_pbc(std=std / 1000.,) if use_pbc else None

    if model_name in ['EGNN', 'PointNet']:
        param_dict['apply_pbc'] = apply_pbc

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
    
    # Number of parameters
    print(f"Number of parameters: {sum([p.size for p in jax.tree_leaves(params)])}")    
    
    # Define train state and replicate across devices
    replicate = flax.jax_utils.replicate
    unreplicate = flax.jax_utils.unreplicate

    lr = optax.cosine_decay_schedule(3e-4, 2000)
    tx = optax.adamw(learning_rate=lr, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    pstate = replicate(state)
    early_stop = EarlyStopping(min_delta=1e-4, patience=10)
    
    # Run training loop
    print('Training...')
    losses = []
    val_losses = []
    best_val = 1e10
    test_loss_ckp = 1e10
    with trange(n_steps, ncols=120) as steps:
        for step in steps:
            # Training
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(key, n_train, shape=(batch_size,))

            halo_batch = halo_train[idx]
            y_batch = y_train[idx]
            if use_tpcf != "none":
                tpcfs_batch = tpcfs_train[idx]
            else:
                tpcfs_batch = None

            halo_batch, y_batch, tpcfs_batch = split_batches(
                num_local_devices, halo_batch, y_batch, tpcfs_batch
            )
            pstate, metrics = train_step(
                pstate, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, radius, use_3d_distances
            )
            train_loss = unreplicate(metrics["loss"])

            if step % eval_every == 0:
                # Validation
                val_iter = iter(val_dataset)
                running_val_loss = 0.0
                for _ in range(n_val // val_batch_size):
                    halo_batch, y_batch = next(val_iter)
                    halo_batch, y_batch = halo_batch.numpy(), y_batch.numpy()
                    tpcfs_batch = None
                
                    halo_batch, y_batch, tpcfs_batch = split_batches(
                        num_local_devices, halo_batch, y_batch, tpcfs_batch
                    )
                    outputs, metrics = eval_step(
                        pstate, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, r_max, use_3d_distances
                    )
                    val_loss = unreplicate(metrics["loss"])
                    running_val_loss += val_loss
                avg_val_loss = running_val_loss/(n_val // val_batch_size)

                if avg_val_loss < best_val:
                    best_val = avg_val_loss
                    tag = " (best)"

                    test_iter = iter(test_dataset)
                    running_test_loss = 0.0
                    for _ in range(n_test // test_batch_size):
                        halo_batch, y_batch = next(test_iter)
                        halo_batch, y_batch = halo_batch.numpy(), y_batch.numpy()
                        tpcfs_batch = None
                    
                        halo_batch, y_batch, tpcfs_batch = split_batches(
                            num_local_devices, halo_batch, y_batch, tpcfs_batch
                        )
                        outputs, metrics = eval_step(
                            pstate, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, r_max, use_3d_distances
                        )
                        test_loss = unreplicate(metrics["loss"])
                        running_test_loss += test_loss
                    avg_test_loss = running_test_loss/(n_test // test_batch_size)

                    test_loss_ckp = avg_test_loss
                else:
                    tag = ""

            
            steps.set_postfix_str('avg loss: {:.5f}, val_loss: {:.5f}, ckp_test_loss: {:.5F}'.format(train_loss,
                                                                                                   val_loss,
                                                                                                   test_loss_ckp))
            losses.append(train_loss)
            val_losses.append(avg_val_loss)

            # early_stop = early_stop.update(avg_val_loss)
            # if early_stop.should_stop:
            #     print(f'Met early stopping criteria, breaking at epoch {step}')
            # break
            
        print(
            "Training done.\n"
            f"Final checkpoint test loss {test_loss_ckp:.6f}.\n"
        )
        
    if plotting:
        plt.scatter(np.vstack(y_test), outputs, color='firebrick')
        plt.plot(np.vstack(y_test), np.vstack(y_test), color='gray')
        plt.title('True vs. predicted y')
        plt.xlabel('True')
        plt.ylabel('Predicted')
        plt.savefig(current_experiment_dir / "y_preds.png")
        
    np.save(current_experiment_dir / "train_losses.npy", losses)
    np.save(current_experiment_dir / "val_losses.npy", val_losses)
    
    
def main(model, feats, target, lr, decay, steps, batch_size, n_radial_basis, use_tpcf, k, radius):
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
        # GNN_PARAMS['task'] = 'node'
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

    data_dir = Path(__file__).parent / "data/"

    run_expt(
        model,
        feats,
        target,
        params,
        data_dir,
        learning_rate=lr,
        weight_decay=decay,
        n_steps=steps,
        batch_size=batch_size,
        n_radial_basis=n_radial_basis,
        use_tpcf=use_tpcf,
        radius=radius
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model", default="GNN")
    parser.add_argument(
        "--feats", help="Features to use", default="pos", choices=["pos", "all"]
    )
    parser.add_argument(
        "--target", help="Target to predict", default="Omega_m", choices=["Omega_m", "sigma_8"]
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--decay", type=float, help="Weight decay", default=1e-5)
    parser.add_argument("--steps", type=int, help="Number of steps", default=5000)
    parser.add_argument("--batch_size", type=int, help="Batch size", default=32)
    parser.add_argument(
        "--n_radial_basis",
        type=int,
        help="Number of radial basis functions.",
        default=0,
    )

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
    parser.add_argument(
        "--radius",
        type=float,
        help="Use radial cutoff to build graph",
        default=None,
    )
    args = parser.parse_args()

    K = args.k

    main(**vars(args))
    