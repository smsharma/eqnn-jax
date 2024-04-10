import flax
from flax.training.train_state import TrainState
from functools import partial
import flax.linen as nn
import optax
from tqdm import trange, tqdm
from time import sleep
from datetime import datetime
from typing import Dict
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

from models.utils.graph_utils import build_graph
from models.utils.irreps_utils import weight_balanced_irreps
from models.utils.equivariant_graph_utils import get_equivariant_graph

from models.mlp import MLP
from models.gnn import GNN
from models.egnn import EGNN

# from models.equivariant_transformer import EquivariantTransformer
from models.segnn import SEGNN
from models.diffpool import DiffPool

from benchmarks.galaxies.dataset import GalaxyDataset


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
    "mlp_readout_widths": [8, 2],
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
    "k": 15,
    "gnn_kwargs": GNN_PARAMS,
    "combine_hierarchies_method": "concat",
    "use_edge_features": True,
    "task": "graph",
    "mlp_readout_widths": [8, 2, 2],
}


SEGNN_PARAMS = {
    "d_hidden": 64,
    "l_max_hidden": 1,
    "num_blocks": 3,
    "num_message_passing_steps": 2,
    "intermediate_hidden_irreps": True,
    "task": "graph",
    "output_irreps": e3nn.Irreps("1x0e"),
    "hidden_irreps": weight_balanced_irreps(
        lmax=1,
        scalar_units=64,
        irreps_right=e3nn.Irreps.spherical_harmonics(1),
    ),
    "normalize_messages": True,
    "message_passing_agg": "mean",
}
# SEGNN_PARAMS['hidden_irreps'] = None


class GraphWrapper(nn.Module):
    model_name: str
    param_dict: Dict
    norm_dict: Dict

    @nn.compact
    def __call__(self, x):
        if self.model_name == "DeepSets":
            raise NotImplementedError
        elif self.model_name == "GNN":
            return jax.vmap(GNN(**self.param_dict))(x)
        elif self.model_name == "EGNN":
            return jax.vmap(EGNN(**self.param_dict))(x)
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
                senders=x.senders,
                receivers=x.receivers,
                n_node=x.n_node,
                n_edge=x.n_edge,
                globals=x.globals,
                edges=None,
                lmax_attributes=1,
                periodic_boundaries=True,
                norm_dict=self.norm_dict,
                unit_cell=np.array(
                    [
                        [
                            1.0,
                            0.0,
                            0.0,
                        ],
                        [0.0, 1.0, 0.0],
                        [0.0, 0.0, 1.0],
                    ]
                ),
            )

            return jax.vmap(SEGNN(**self.param_dict))(st_graph)
        elif self.model_name == "DiffPool":
            return jax.vmap(DiffPool(**self.param_dict))(x)
        else:
            raise ValueError("Please specify a valid model name.")


def loss_mse(pred_batch, cosmo_batch):
    return np.mean((pred_batch - cosmo_batch) ** 2)


@partial(
    jax.pmap,
    axis_name="batch",
)
def train_step(state, halo_batch, omega_m_batch, tpcfs_batch):
    halo_graph = build_graph(
        halo_batch, tpcfs_batch, k=K, use_pbc=True, use_edges=True, use_rbf=True
    )

    def loss_fn(params):
        outputs = state.apply_fn(params, halo_graph)
        if len(outputs.shape) > 2:
            outputs = np.squeeze(outputs, axis=-1)
        loss = loss_mse(outputs, omega_m_batch)
        return loss

    # Get loss, grads, and update state
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}

    outputs = state.apply_fn(state.params, halo_graph)

    return new_state, metrics


@partial(
    jax.pmap,
    axis_name="batch",
)
def eval_step(state, halo_batch, omega_m_batch, tpcfs_batch):
    # Build graph
    halo_graph = build_graph(
        halo_batch, tpcfs_batch, k=K, use_pbc=True, use_edges=True, use_rbf=True
    )

    outputs = state.apply_fn(state.params, halo_graph)
    if len(outputs.shape) > 2:
        outputs = np.squeeze(outputs, axis=-1)
    loss = jax.lax.stop_gradient(loss_mse(outputs, omega_m_batch))

    return outputs, {"loss": jax.lax.pmean(loss, "batch")}


def split_batches(num_local_devices, halo_batch, omega_m_batch, tpcfs_batch):
    halo_batch = jax.tree_map(
        lambda x: np.split(x, num_local_devices, axis=0), halo_batch
    )
    omega_m_batch = jax.tree_map(
        lambda x: np.split(x, num_local_devices, axis=0), omega_m_batch
    )
    halo_batch, omega_m_batch = np.array(halo_batch), np.array(omega_m_batch)

    if tpcfs_batch is not None:
        tpcfs_batch = jax.tree_map(
            lambda x: np.split(x, num_local_devices, axis=0), tpcfs_batch
        )
        tpcfs_batch = np.array(tpcfs_batch)

    return halo_batch, omega_m_batch, tpcfs_batch


def run_expt(
    model_name,
    feats,
    param_dict,
    data_dir,
    use_pbc=True,
    use_edges=True,
    use_rbf=False,
    use_tpcf="none",
    n_steps=1000,
    n_batch=32,
    n_train=1600,
    n_test_batch=32,
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
        f"{model_name}_{feats}_{n_batch}_{n_steps}_{d_hidden}_{K}_"
        + use_tpcf
        + "_rbf" * use_rbf
    )

    current_experiment_dir = experiments_base_dir / experiment_id
    current_experiment_dir.mkdir(parents=True, exist_ok=True)

    print("Loading dataset...")
    if feats == "pos":
        D = GalaxyDataset(data_dir, use_pos=True, use_vel=False, use_tpcf=use_tpcf)
    elif feats == "all":
        D = GalaxyDataset(data_dir, use_pos=True, use_vel=True, use_tpcf=use_tpcf)
    else:
        raise NotImplementedError

    halo_train, halo_val, halo_test = D.halos_train, D.halos_val, D.halos_test
    num_train_sims = halo_train.shape[0]
    mean, std = D.halos_mean, D.halos_std
    norm_dict = {"mean": mean, "std": std}

    omega_m_train, omega_m_val, omega_m_test = (
        D.omega_m_train,
        D.omega_m_val,
        D.omega_m_test,
    )

    if use_tpcf != "none":
        if use_tpcf == "small":
            tpcf_idx = list(range(8))
        elif use_tpcf == "large":
            tpcf_idx = list(range(15, 24))
        else:
            tpcf_idx = list(range(24))

        tpcfs_train = D.tpcfs_train[:, tpcf_idx]
        tpcfs_val = D.tpcfs_val[:, tpcf_idx]
        tpcfs_test = D.tpcfs_test[:, tpcf_idx]

        graph = build_graph(
            halo_train[:2],
            tpcfs_train[:2],
            k=K,
            use_pbc=use_pbc,
            use_edges=use_edges,
            use_rbf=use_rbf,
            mean=mean,
            std=std,
        )
    else:
        tpcfs_train = None
        tpcfs_val = None
        tpcfs_test = None

        graph = build_graph(
            halo_train[:2],
            None,
            k=K,
            use_pbc=use_pbc,
            use_edges=use_edges,
            use_rbf=use_rbf,
            mean=mean,
            std=std,
        )

    # Split eval batches across devices
    num_local_devices = jax.local_device_count()
    halo_val, omega_m_val, tpcfs_val = split_batches(
        num_local_devices, halo_val, omega_m_val, tpcfs_val
    )
    halo_test, omega_m_test, tpcfs_test = split_batches(
        num_local_devices, halo_test, omega_m_test, tpcfs_test
    )

    if get_node_reps:
        param_dict["get_node_reps"] = True

    model = GraphWrapper(model_name, param_dict, norm_dict)
    key = jax.random.PRNGKey(0)
    out, params = model.init_with_output(key, graph)

    # Define train state and replicate across devices
    replicate = flax.jax_utils.replicate
    unreplicate = flax.jax_utils.unreplicate
    tx = optax.adamw(learning_rate=learning_rate, weight_decay=weight_decay)
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
            omega_m_batch = omega_m_train[:n_train][idx]
            if use_tpcf != "none":
                tpcfs_batch = tpcfs_train[idx]
            else:
                tpcfs_batch = None

            # Split batches across devices
            halo_batch, omega_m_batch, tpcfs_batch = split_batches(
                num_local_devices, halo_batch, omega_m_batch, tpcfs_batch
            )
            pstate, metrics = train_step(pstate, halo_batch, omega_m_batch, tpcfs_batch)
            train_loss = unreplicate(metrics["loss"])

            if step % eval_every == 0:
                outputs, val_metrics = eval_step(
                    pstate, halo_val, omega_m_val, tpcfs_val
                )
                val_loss = unreplicate(val_metrics["loss"])

                if val_loss < best_val:
                    best_val = val_loss
                    tag = " (best)"

                    outputs, test_metrics = eval_step(
                        pstate, halo_test, omega_m_test, tpcfs_test
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

        outputs, test_metrics = eval_step(pstate, halo_test, omega_m_test, tpcfs_test)
        test_loss = unreplicate(test_metrics["loss"])
        print(
            "Training done.\n"
            f"Final test loss {test_loss:.6f} - Checkpoint test loss {test_loss_ckp:.6f}.\n"
        )

    if plotting:
        plt.scatter(np.vstack(omega_m_test), outputs, color="firebrick")
        plt.plot(np.vstack(omega_m_test), np.vstack(omega_m_test), color="gray")
        plt.title("True vs. predicted Omega_m")
        plt.xlabel("True")
        plt.ylabel("Predicted")
        plt.savefig(current_experiment_dir / "omega_m_preds.png")

    np.save(current_experiment_dir / "train_losses.npy", losses)
    np.save(current_experiment_dir / "val_losses.npy", val_losses)


def main(model, feats, lr, decay, steps, batch_size, use_rbf, use_tpcf, k):
    if model == "GNN":
        params = GNN_PARAMS
    elif model == "EGNN":
        params = EGNN_PARAMS
    elif model == "SEGNN":
        params = SEGNN_PARAMS
    elif model == "DiffPool":
        # GNN_PARAMS['task'] = 'node'
        DIFFPOOL_PARAMS["gnn_kwargs"] = {
            "d_hidden": 64,
            "d_output": 16,
            "n_layers": 2,
            "message_passing_steps": 2,
            "task": "node",
        }
        DIFFPOOL_PARAMS["d_hidden"] = DIFFPOOL_PARAMS["gnn_kwargs"]["d_hidden"]
        params = DIFFPOOL_PARAMS
    else:
        raise NotImplementedError

    data_dir = Path("/n/holystore01/LABS/iaifi_lab/Lab/set-diffuser-data/val_split/")

    run_expt(
        model,
        feats,
        params,
        data_dir,
        learning_rate=lr,
        weight_decay=decay,
        n_steps=steps,
        n_batch=batch_size,
        use_rbf=use_rbf,
        use_tpcf=use_tpcf,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Name of model", default="GNN")
    parser.add_argument(
        "--feats", help="Features to use", default="pos", choices=["pos", "all"]
    )
    parser.add_argument("--lr", type=float, help="Learning rate", default=1e-4)
    parser.add_argument("--decay", type=float, help="Weight decay", default=1e-5)
    parser.add_argument("--steps", type=int, help="Number of steps", default=5000)
    parser.add_argument("--batch_size", help="Batch size", default=32)
    parser.add_argument(
        "--use_rbf",
        type=bool,
        help="Whether to include RBF kernel for edge features",
        default=False,
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
    args = parser.parse_args()

    K = args.k

    main(**vars(args))
