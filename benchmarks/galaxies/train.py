import flax
from flax.training.train_state import TrainState
from functools import partial
import flax.linen as nn
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
from models.utils.irreps_utils import weight_balanced_irreps
from models.utils.equivariant_graph_utils import get_equivariant_graph

from models.mlp import MLP
from models.gnn import GNN
from models.egnn import EGNN
# from models.equivariant_transformer import EquivariantTransformer
from models.segnn import SEGNN
from models.diffpool import DiffPool
from models.nequip import NequIP
from models.pointnet import PointNet

from benchmarks.galaxies.dataset import GalaxyDataset

MLP_PARAMS = {
    "feature_sizes": [128, 128, 128, 1],
}

GNN_PARAMS = {
    "n_outputs": 1,
    "message_passing_steps": 2,
    "n_layers": 3,
    "d_hidden": 64,
    "d_output": 64,
    "activation": "gelu",
    "message_passing_agg": "mean",
    "readout_agg": "mean",
    "readout_only_positions": False,
    "task": "graph",
    "mlp_readout_widths": [8, 2, 2],
    "norm": "layer",
}

EGNN_PARAMS = {
    "n_outputs": 1,
    "message_passing_steps": 2,
    "n_layers": 3,
    "d_hidden": 64,
    "activation": "gelu",
    "message_passing_agg": "mean",
    "readout_agg": "mean",
    "readout_only_positions": False,
    "task": "graph",
    "mlp_readout_widths": [8, 2, 2],
    "use_fourier_features": True,
    "tanh_out": False,
    "soft_edges": True,
    "decouple_pos_vel_updates": True,
    "normalize_messages": True,
    "positions_only": True,
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
    "d_hidden": 64,
    "l_max_hidden": 2,
    "num_blocks": 2,
    "num_message_passing_steps": 3,
    "intermediate_hidden_irreps": True,
    "task": "graph",
    "output_irreps": e3nn.Irreps("1x0e"),
    "hidden_irreps": weight_balanced_irreps(
        lmax=2,
        scalar_units=64,
        irreps_right=e3nn.Irreps.spherical_harmonics(1),
    ),
    "normalize_messages": True,
    "message_passing_agg": "mean",
    "readout_agg": "mean",
    "mlp_readout_widths": [8, 2, 2],
}

NEQUIP_PARAMS = {
    "d_hidden": 64,
    "l_max_hidden":2,
    "l_max_attr":2,
    "sphharm_norm": 'norm',
    "num_message_passing_steps": 3,
    "n_layers": 3,
    "task": "graph",
    "irreps_out": e3nn.Irreps("1x0e"),
    "normalize_messages": True,
    "message_passing_agg": "mean",
    "readout_agg": "mean",
    "mlp_readout_widths": [8, 2, 2],
    "n_radial_basis": 8
}


class GraphWrapper(nn.Module):
    model_name: str
    param_dict: Dict
    apply_pbc: Callable = None

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
                lmax_attributes=1,
                apply_pbc=self.apply_pbc
            )
            return jax.vmap(SEGNN(**self.param_dict))(st_graph)
        elif self.model_name == "NequIP":
            if x.nodes.shape[-1] == 3:
                # nodes = e3nn.IrrepsArray("1o", x.nodes[..., :])
                ones = np.ones(x.nodes[..., :].shape[:2] + (4,))
                nodes = np.concatenate([x.nodes[..., :], ones], axis=-1)
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
    static_broadcasted_argnums=(4,5,6)
)
def train_step(state, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, radius):
    halo_graph = build_graph(
        halo_batch, 
        tpcfs_batch, 
        k=K, 
        apply_pbc=apply_pbc, 
        use_edges=True, 
        n_radial_basis=n_radial_basis,
        radius=radius
    )

    def loss_fn(params):
        # outputs = state.apply_fn(params, halo_graph)
        outputs, assignments = state.apply_fn(params, halo_graph)
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
    static_broadcasted_argnums=(4,5,6)
)
def eval_step(state, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, radius):
    # Build graph
    halo_graph = build_graph(
        halo_batch, 
        tpcfs_batch, 
        k=K, 
        apply_pbc=apply_pbc, 
        use_edges=True, 
        n_radial_basis=n_radial_basis,
        radius=radius
    )

    # outputs = state.apply_fn(state.params, halo_graph)
    outputs, assignments = state.apply_fn(state.params, halo_graph)
    if len(outputs.shape) > 2:
        outputs = np.squeeze(outputs, axis=-1)
    loss = jax.lax.stop_gradient(loss_mse(outputs, y_batch))

    return outputs, {"loss": jax.lax.pmean(loss, "batch")} , assignments


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
    target,
    param_dict,
    data_dir,
    use_pbc=True,
    use_edges=True,
    n_radial_basis=0,
    use_tpcf="none",
    radius=None,
    n_steps=1000,
    n_batch=32,
    n_train=1600,
    learning_rate=5e-5,
    weight_decay=1e-5,
    eval_every=200,
    get_node_reps=False,
    plotting=True,
):

    # Create experiment directory
    experiments_base_dir = Path(__file__).parent / "experiments/"
    d_hidden = param_dict["d_hidden"]
    experiment_id = (
        f"{model_name}_{feats}_{n_batch}b_{n_steps}s_{d_hidden}d_{K}k_{n_radial_basis}rbf"
        + "tpcf-" + use_tpcf
        + f"_{radius}r"
    )

    current_experiment_dir = experiments_base_dir / experiment_id
    current_experiment_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    if feats == "pos":
        D = GalaxyDataset(data_dir, use_pos=True, use_vel=False, use_tpcf=use_tpcf)
    elif feats == "all":
        D = GalaxyDataset(data_dir, use_pos=True, use_vel=True, use_mass=True, use_tpcf=use_tpcf)
    else:
        raise NotImplementedError

    halo_train, halo_val, halo_test = D.halos_train, D.halos_val, D.halos_test
    num_train_sims = halo_train.shape[0]
    mean, std = D.halos_mean, D.halos_std

    apply_pbc = get_apply_pbc(std=std) if use_pbc else None
    if model_name in ['EGNN', 'PointNet']:
        param_dict['apply_pbc'] = apply_pbc

    if target == 'Omega_m':
        y_train, y_val, y_test = (
            D.targets_train[:, [0]],
            D.targets_val[:, [0]],
            D.targets_test[:, [0]],
        )
    else:
        y_train, y_val, y_test = (
            D.targets_train[:, [1]],
            D.targets_val[:, [1]],
            D.targets_test[:, [1]],
        )

    if use_tpcf != "none":
        tpcfs_train = D.tpcfs_train
        tpcfs_val = D.tpcfs_val
        tpcfs_test = D.tpcfs_test
        init_tpcfs = tpcfs_train[:2]
    else:
        tpcfs_train = None
        tpcfs_val = None
        tpcfs_test = None
        init_tpcfs = None

    graph = build_graph(
        halo_train[:2],
        init_tpcfs,
        k=K,
        apply_pbc=apply_pbc,
        use_edges=use_edges,
        n_radial_basis=n_radial_basis,
        radius=radius
    )


    # Split eval batches across devices
    num_local_devices = jax.local_device_count()
    halo_val, y_val, tpcfs_val = split_batches(
        num_local_devices, halo_val, y_val, tpcfs_val
    )
    halo_test, y_test, tpcfs_test = split_batches(
        num_local_devices, halo_test, y_test, tpcfs_test
    )

    if get_node_reps:
        param_dict["get_node_reps"] = True

    model = GraphWrapper(model_name, param_dict, apply_pbc)
    key = jax.random.PRNGKey(0)
    out, params = model.init_with_output(key, graph)

    # Define train state and replicate across devices
    replicate = flax.jax_utils.replicate
    unreplicate = flax.jax_utils.unreplicate

    lr_scheduler = optax.linear_onecycle_schedule(n_steps//2, learning_rate)
    tx = optax.adamw(learning_rate=lr_scheduler, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    pstate = replicate(state)


    # Run training loop
    print("Training...")
    losses = []
    val_losses = []
    best_val = 1e10
    with trange(n_steps, ncols=100) as steps:
        for step in steps:
            key, subkey = jax.random.split(key)
            idx = jax.random.choice(key, num_train_sims, shape=(n_batch,))

            halo_batch = halo_train[:n_train][idx]
            y_batch = y_train[:n_train][idx]
            if use_tpcf != "none":
                tpcfs_batch = tpcfs_train[idx]
            else:
                tpcfs_batch = None

            # Split batches across devices
            halo_batch, y_batch, tpcfs_batch = split_batches(
                num_local_devices, halo_batch, y_batch, tpcfs_batch
            )
            pstate, metrics = train_step(
                pstate, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, radius
            )
            train_loss = unreplicate(metrics["loss"])

            if step % eval_every == 0:
                # outputs, val_metrics = eval_step(
                #     pstate, halo_val, y_val, tpcfs_val, apply_pbc, n_radial_basis, radius
                # )
                outputs, val_metrics, assignments = eval_step(
                    pstate, halo_val, y_val, tpcfs_val, apply_pbc, n_radial_basis, radius
                )
                val_loss = unreplicate(val_metrics["loss"])
                
                # print(type(assignments))
                # assignments = np.reshape(assignments, (num_local_devices*assignments.shape[1], assignments.shape[-1]))
                
                for a, assignment in enumerate(assignments):
                    if a == 0:
                        c = assignment.reshape((200, 5000, 1000))
                        colors = np.argmax(c, axis=-1)[0]
                        np.save(current_experiment_dir / f"colors.npy", colors)
                    np.save(current_experiment_dir / f"assignments_{a}.npy", assignment)

                if val_loss < best_val:
                    best_val = val_loss
                    tag = " (best)"

                    # outputs, test_metrics = eval_step(
                    #     pstate, halo_test, y_test, tpcfs_test, apply_pbc, n_radial_basis, radius
                    # )
                    outputs, test_metrics, assignments = eval_step(
                        pstate, halo_test, y_test, tpcfs_test, apply_pbc, n_radial_basis, radius
                    )
                    test_loss_ckp = unreplicate(test_metrics["loss"])
                else:
                    tag = ""

            steps.set_postfix_str(
                "loss: {:.5f}, val_loss: {:.5f}, ckp_test_loss: {:.5F}".format(
                    train_loss, val_loss, test_loss_ckp
                )
            )
            losses.append(train_loss)
            val_losses.append(val_loss)

        # outputs, test_metrics = eval_step(
        #     pstate, halo_test, y_test, tpcfs_test, apply_pbc, n_radial_basis, radius
        # )
        outputs, test_metrics, assignments = eval_step(
            pstate, halo_test, y_test, tpcfs_test, apply_pbc, n_radial_basis, radius
        )
        test_loss = unreplicate(test_metrics["loss"])
        print(
            "Training done.\n"
            f"Final test loss {test_loss:.6f} - Checkpoint test loss {test_loss_ckp:.6f}.\n"
        )

    if plotting:
        plt.scatter(np.vstack(y_test), outputs, color="firebrick")
        plt.plot(np.vstack(y_test), np.vstack(y_test), color="gray")
        plt.title("True vs. predicted " + target)
        plt.xlabel("True")
        plt.ylabel("Predicted")
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
            "d_output": 64,
            "n_layers": 3,
            "message_passing_steps": 4,
            "task": "node",
        }
        DIFFPOOL_PARAMS["d_hidden"] = DIFFPOOL_PARAMS["gnn_kwargs"]["d_hidden"]
        params = DIFFPOOL_PARAMS
    elif model == "PointNet":
        POINTNET_PARAMS["gnn_kwargs"] = {
            "d_hidden": 64,
            "d_output": 64,
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
        n_batch=batch_size,
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
        "--k", type=int, help="Number of neighbors for kNN graph", default=20
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