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

import wandb


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
            # pos = e3nn.IrrepsArray("1o", x.nodes[..., :3])
            # feat = e3nn.IrrepsArray("1o", x.nodes[..., 3:])

            # return jax.vmap(EquivariantTransformer(**self.param_dict))(pos, feat, x.senders, x.receivers,)
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
    return np.mean((pred_batch - cosmo_batch) ** 2, axis=0)


@partial(
    jax.pmap,
    axis_name="batch",
    static_broadcasted_argnums=(4,5,6,7)
)
def train_step(state, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, k, radius):
    halo_graph = build_graph(
        halo_batch, 
        tpcfs_batch, 
        k=k, 
        apply_pbc=apply_pbc, 
        use_edges=True, 
        n_radial_basis=n_radial_basis,
        radius=radius
    )

    def loss_fn(params):
        outputs = state.apply_fn(params, halo_graph)
        # outputs, assignments = state.apply_fn(params, halo_graph)
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
def eval_step(state, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, k, radius):
    # Build graph
    halo_graph = build_graph(
        halo_batch, 
        tpcfs_batch, 
        k=k, 
        apply_pbc=apply_pbc, 
        use_edges=True, 
        n_radial_basis=n_radial_basis,
        radius=radius
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
    target,
    param_dict,
    data_dir,
    use_pbc=True,
    use_edges=True,
    n_radial_basis=0,
    use_tpcf="none",
    k=20,
    radius=None,
    n_steps=1000,
    n_batch=32,
    n_train=1600,
    learning_rate=5e-5,
    weight_decay=1e-5,
    eval_every=200,
    get_node_reps=False,
    plotting=False,
):

    # Create experiment directory
    experiments_base_dir = Path(__file__).parent / "experiments/"
    d_hidden = param_dict["d_hidden"]
    experiment_id = (
        f"{model_name}_{feats}_{n_batch}b_{n_steps}s_{d_hidden}d_{k}k_{n_radial_basis}rbf"
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

    if target == ['Omega_m']:
        tgt_idx = np.array([0])
    elif target == ['sigma_8']:
        tgt_idx = np.array([1])
    else:
        tgt_idx = np.array([0, 1])
    
    y_train, y_val, y_test = (
        D.targets_train[:, tgt_idx],
        D.targets_val[:, tgt_idx],
        D.targets_test[:, tgt_idx],
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
        k=k,
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

    lr_scheduler = optax.linear_onecycle_schedule(n_steps, learning_rate)
    tx = optax.adamw(learning_rate=lr_scheduler, weight_decay=weight_decay)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    pstate = replicate(state)

    # Run training loop
    print("Training...")
    losses = []
    val_losses = []
    best_val = 1e10
    for step in range(n_steps):
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
            pstate, halo_batch, y_batch, tpcfs_batch, apply_pbc, n_radial_basis, k, radius
        )
        train_loss = unreplicate(metrics["loss"])

        if step % eval_every == 0:
            outputs, val_metrics = eval_step(
                pstate, halo_val, y_val, tpcfs_val, apply_pbc, n_radial_basis, k, radius
            )
            val_loss = unreplicate(val_metrics["loss"])

            if val_loss.mean() < best_val:
                best_val = val_loss.mean()
    
                outputs, test_metrics = eval_step(
                    pstate, halo_test, y_test, tpcfs_test, apply_pbc, n_radial_basis, k, radius
                )
                test_loss_ckp = unreplicate(test_metrics["loss"])


        if len(target) > 1:
            wandb.log({f'train_avg_loss': train_loss.mean(),
                        f'val_avg_loss': val_loss.mean(),
                        f'ckp_test_avg_loss': test_loss_ckp.mean()}, commit=False)

        for i in range(len(target)):
            to_commit = i == len(target)-1 
            wandb.log({
                    f'train_{target[i]}_loss': train_loss[i],
                    f'val_{target[i]}_loss': val_loss[i],
                    f'ckp_test_{target[i]}_loss': test_loss_ckp[i]}, commit=to_commit)

        losses.append(train_loss)
        val_losses.append(val_loss)

    outputs, test_metrics = eval_step(
        pstate, halo_test, y_test, tpcfs_test, apply_pbc, n_radial_basis, k, radius
    )
    test_loss = unreplicate(test_metrics["loss"])
    print(
        "Training done.\n"
        f"Final avg test loss {test_loss.mean():.6f} - Checkpoint avg test loss {test_loss_ckp.mean():.6f}.\n"
    )

    if plotting:
        plt.scatter(np.vstack(y_test), outputs, color="firebrick")
        plt.plot(np.vstack(y_test), np.vstack(y_test), color="gray")
        plt.title("True vs. predicted Omega_m")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.savefig(current_experiment_dir / "y_preds.png")

    np.save(current_experiment_dir / "train_losses.npy", losses)
    np.save(current_experiment_dir / "val_losses.npy", val_losses)


def main(config=None):
    with wandb.init(config=config):
        config = wandb.config
        n_targets = len(config.target)

        if config.model == 'MLP':
            params = {
                "feature_sizes": [4*config.d_hidden, 2*config.d_hidden, 2*config.d_hidden] + [n_targets],
                "d_hidden": config.d_hidden
            }
        elif config.model == "GNN":
            params = {
                "n_outputs": n_targets,
                "message_passing_steps": config.message_passing_steps,
                "n_layers": config.n_layers,
                "d_hidden": config.d_hidden,
                "activation": "gelu",
                "message_passing_agg": config.message_passing_agg,
                "readout_agg": config.readout_agg,
                "readout_only_positions": False,
                "task": "graph",
                "mlp_readout_widths": [4, 2, 2],
                "norm": "layer",
            }
        elif config.model == "EGNN":
            params = EGNN_PARAMS
        elif config.model == "SEGNN":
            params = {
                "d_hidden": config.d_hidden,
                "num_blocks": config.num_blocks,
                "num_message_passing_steps": config.message_passing_steps,
                "intermediate_hidden_irreps": True,
                "task": "graph",
                "output_irreps": e3nn.Irreps(f"{n_targets}x0e"),
                "hidden_irreps": weight_balanced_irreps(
                    lmax=2,
                    scalar_units=config.d_hidden,
                    irreps_right=e3nn.Irreps.spherical_harmonics(1),
                ),
                "normalize_messages": config.normalize_messages,
                "message_passing_agg":  config.message_passing_agg,
                "readout_agg": config.readout_agg,
                "mlp_readout_widths": [4, 2, 2],
                "residual": config.residual,
                "intermediate_hidden_irreps": config.intermediate_hidden_irreps
            }
        elif config.model == "NequIP":
            params = {
                "d_hidden": config.d_hidden,
                "l_max_hidden": config.l_max_hidden,
                "l_max_attr": config.l_max_attr,
                "sphharm_norm": config.sphharm_norm,
                "num_message_passing_steps": config.message_passing_steps,
                "task": "graph",
                "irreps_out": e3nn.Irreps(f"{n_targets}x0e"),
                "normalize_messages": config.normalize_messages,
                "message_passing_agg":  config.message_passing_agg,
                "readout_agg": config.readout_agg,
                "mlp_readout_widths": [4, 2, 2],
                "n_radial_basis": config.n_radial_basis
            }
        elif config.model == "DiffPool":
            params = {
                "d_hidden": config.d_hidden,
                "n_downsamples": config.n_downsamples,
                "d_downsampling_factor": config.d_downsampling_factor,
                "k": config.k_downsample,
                "combine_hierarchies_method": config.combine_hierarchies_method,
                "use_edge_features": config.use_edge_features,
                "task": "graph",
                "mlp_readout_widths": [4, 2, 2],
                "n_outputs": n_targets
            }

            params["gnn_kwargs"] = {
                "d_hidden": config.d_hidden,
                "d_output": config.d_hidden,
                "n_layers": config.n_layers,
                "message_passing_steps": config.message_passing_steps,
                "task": "node",
            }
        elif config.model == "PointNet":
            params = {
                "d_hidden": config.d_hidden,
                "n_downsamples": config.n_downsamples,
                "d_downsampling_factor": config.d_downsampling_factor,
                "k": config.k_downsample,
                "radius": config.r_downsample,
                "combine_hierarchies_method": config.combine_hierarchies_method,
                "use_edge_features": config.use_edge_features,
                "task": "graph",
                "mlp_readout_widths": [4, 2, 2],
                "n_outputs": n_targets
            }

            params["gnn_kwargs"] = {
                "d_hidden": config.d_hidden,
                "d_output": config.d_hidden,
                "n_layers": config.n_layers,
                "message_passing_steps": config.message_passing_steps,
                "task": "node",
            }
        else:
            raise NotImplementedError

        data_dir = Path(__file__).parent / "data/"

        run_expt(
            config.model,
            config.feats,
            config.target,
            params,
            data_dir,
            k = config.k,
            learning_rate=config.learning_rate,
            weight_decay=config.decay,
            n_steps=config.steps,
            n_batch=config.batch_size,
            n_radial_basis=config.n_radial_basis,
            use_tpcf=config.use_tpcf,
            radius=config.radius
        )


if __name__ == "__main__":
    main()