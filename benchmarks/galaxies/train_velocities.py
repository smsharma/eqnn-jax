import sys

sys.path.append("../../")
import argparse
from functools import partial
from pathlib import Path
from tqdm import trange
from typing import Dict
import pickle


import numpy as np
import jax

import matplotlib.pyplot as plt
import jax.numpy as jnp

import flax
import flax.linen as nn
from flax.training.train_state import TrainState
from flax.training import checkpoints
import optax
import e3nn_jax as e3nn
import jraph


from dataset_large import get_halo_dataset
from models.gnn import GNN
from models.segnn import SEGNN
from models.utils.graph_utils import build_graph

from models.utils.graph_utils import get_apply_pbc
from models.gnn import GNN
from models.segnn import SEGNN
from models.nequip import NequIP


from models.utils.graph_utils import build_graph
from models.utils.equivariant_graph_utils import get_equivariant_graph

# make sure iterators have the right amounts of samples

replicate = flax.jax_utils.replicate
unreplicate = flax.jax_utils.unreplicate


def get_iterators(
    training_set_size,
    features=["x", "y", "z", "v_x", "v_y", "v_z"],
    tfrecords_path="/pscratch/sd/c/cuesta/quijote_tfrecords",
    batch_size=64,
):
    dataset, _, _, std, _, _ = get_halo_dataset(
        tfrecords_path=tfrecords_path,
        batch_size=batch_size,
        num_samples=training_set_size,
        split="train",
        return_mean_std=True,
        features=features,
    )
    std = np.array(std)
    train_iterator = iter(dataset)
    val_dataset, _ = get_halo_dataset(
        batch_size=batch_size,
        tfrecords_path=tfrecords_path,
        num_samples=512,
        split="val",
        features=features,
    )
    val_iterator = iter(val_dataset)
    test_dataset, _ = get_halo_dataset(
        batch_size=batch_size,
        tfrecords_path=tfrecords_path,
        num_samples=512,
        split="test",
        features=features,
    )
    test_iterator = iter(test_dataset)
    iterators = {
        "train": train_iterator,
        "val": val_iterator,
        "test": test_iterator,
    }
    return iterators, std


@partial(
    jax.pmap,
    axis_name="batch",
)
def train_step(state, x_batch_masked, x_batch, mask):
    # Set those velocities in x_batch (only indices 3:6 of last dimension of x_batch) to 0
    graph = fixed_build_graph(
        x_batch_masked,
    )

    def loss_fn(params):
        outputs = state.apply_fn(params, graph)
        loss = loss_mse(outputs.nodes, x_batch, mask)
        return loss

    # Get loss, grads, and update state
    loss, grads = jax.value_and_grad(loss_fn)(state.params)
    grads = jax.lax.pmean(grads, "batch")
    new_state = state.apply_gradients(grads=grads)
    metrics = {"loss": jax.lax.pmean(loss, "batch")}

    return new_state, metrics


def initialize_state(model, graph, n_steps):
    lr = optax.cosine_decay_schedule(3e-4, n_steps)
    tx = optax.adamw(learning_rate=lr, weight_decay=1e-5)
    _, params = model.init_with_output(jax.random.PRNGKey(0), graph)
    state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)
    return state


def mask_batch(
    x_batch,
    key,
    fraction_masked,
    infill_value,
):
    x_batch = jnp.array(x_batch)
    mask = jax.random.bernoulli(
        key,
        fraction_masked,
        shape=(len(x_batch), x_batch.shape[1], 1),
    )
    # Set those velocities in x_batch (indices 3:6 of last dimension) to 0
    x_batch_masked = x_batch.at[:, :, 3:6].set(
        jnp.where(mask, infill_value, x_batch[:, :, 3:6])
    )
    return x_batch_masked, mask


def split_across_devices(x_batch_masked, x_batch, mask, num_local_devices):
    # Split batches across devices
    x_batch_masked = jax.tree.map(
        lambda x: np.split(x, num_local_devices, axis=0), x_batch_masked
    )
    x_batch_masked = jnp.array(x_batch_masked)

    x_batch = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), x_batch)
    x_batch = jnp.array(x_batch)

    mask = jax.tree.map(lambda x: np.split(x, num_local_devices, axis=0), mask)
    mask = jnp.array(mask)
    return x_batch_masked, x_batch, mask


def mask_and_split_across_devices(
    x_batch, key, fraction_masked, infill_value, num_local_devices
):
    if fraction_masked < 1.0:
        x_batch_masked, mask = mask_batch(
            x_batch,
            key,
            fraction_masked=fraction_masked,
            infill_value=infill_value,
        )
    elif fraction_masked == 1.0:
        x_batch_masked = jnp.array(x_batch[..., :3]).copy()
        mask = jnp.ones((x_batch_masked.shape[0], x_batch_masked.shape[1], 1))

    x_batch_masked, x_batch, mask = split_across_devices(
        x_batch_masked=x_batch_masked,
        x_batch=x_batch,
        mask=mask,
        num_local_devices=num_local_devices,
    )
    return x_batch_masked, x_batch, mask


@partial(
    jax.pmap,
    axis_name="batch",
)
def eval_step(
    pstate,
    x_batch_masked,
    x_batch,
    mask,
):
    graph = fixed_build_graph(
        x_batch_masked,
    )
    outputs = pstate.apply_fn(pstate.params, graph)
    loss = jax.lax.stop_gradient(loss_mse(outputs.nodes, x_batch, mask))
    return jax.lax.pmean(loss, "batch")


def get_loss_for_iterator(iterator, pstate, key, num_local_devices):
    loss_value, count = 0.0, 0
    for x_batch, _ in iterator:
        x_batch_masked, x_batch, mask = mask_and_split_across_devices(
            x_batch,
            key,
            fraction_masked=fraction_masked,
            infill_value=infill_value,
            num_local_devices=num_local_devices,
        )
        loss_value += eval_step(pstate, x_batch_masked, x_batch, mask)
        count += 1
    return loss_value / count


def train_model(
    iterators,
    model,
    eval_every: int = 100,
    n_steps=2000,
    fraction_masked=0.1,
    num_local_devices=1,
    save_model=False,
    ckpt_dir=Path("velocity_task"),
):
    x_train, _ = next(iterators["train"])
    if fraction_masked == 1.0:
        x_example = x_train[:2][..., :3]
    else:
        x_example = x_train[:2]
    graph = fixed_build_graph(
        jnp.array(x_example),
    )
    state = initialize_state(model, graph, n_steps=n_steps)
    pstate = replicate(state)
    # Number of parameters
    print(
        f"Number of parameters: {sum([p.size for p in jax.tree.leaves(state.params)])}"
    )
    key = jax.random.PRNGKey(0)
    best_val_loss = 1.0e9
    best_state = None
    train_losses, val_losses = [], []
    with trange(n_steps) as steps:
        for step in steps:
            key, key_eval = jax.random.split(
                key,
            )
            x_batch, _ = next(iterators["train"])
            x_batch_masked, x_batch, mask = mask_and_split_across_devices(
                x_batch=x_batch,
                key=key,
                fraction_masked=fraction_masked,
                infill_value=infill_value,
                num_local_devices=num_local_devices,
            )
            pstate, metrics = train_step(pstate, x_batch_masked, x_batch, mask)
            train_loss = unreplicate(metrics["loss"])
            train_losses.append(train_loss)
            if step % eval_every == 0:
                x_batch, _ = next(iterators["val"])
                x_batch_masked, x_batch, mask = mask_and_split_across_devices(
                    x_batch,
                    key_eval,
                    fraction_masked=fraction_masked,
                    infill_value=infill_value,
                    num_local_devices=num_local_devices,
                )
                # evaluate loss over iterator
                val_loss = eval_step(
                    pstate,
                    x_batch_masked,
                    x_batch,
                    mask,
                )
                val_loss = unreplicate(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = pstate
                val_losses.append(val_loss)
            steps.set_postfix_str(
                f"loss: {train_loss:.5f}, val_loss: {val_loss:.5f}, best_val_loss: {best_val_loss:.5f}"
            )

    test_loss = get_loss_for_iterator(
        iterators["test"],
        best_state,
        key,
        num_local_devices,
    )
    test_loss = unreplicate(test_loss)
    best_state = unreplicate(best_state)
    print("Test loss = ", test_loss)

    if save_model:
        loss_dict = {
            "train_loss": np.array(train_losses),
            "val_loss": np.array(val_losses),
            "test_loss": np.array(test_loss),
        }
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        checkpoints.save_checkpoint(
            ckpt_dir=str(ckpt_dir),
            target=best_state,
            step=step,
            overwrite=True,
        )
        with open(f"{ckpt_dir}/loss_dict.pkl", "wb") as f:
            pickle.dump(loss_dict, f)


if __name__ == "__main__":
    # - Run nequip
    # - Run all three with radial basis = 64
    # - Run with lmax = 2
    # - Run from 10 to 30k in 6 steps

    # take arguments with argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--training_set_size", type=int, default=1248)
    parser.add_argument("--fraction_masked", type=float, default=1.0)
    args = parser.parse_args()
    tfrecords_path = "/pscratch/sd/c/cuesta/quijote_tfrecords_consistent_trees"
    output_path = Path(
        "/global/u1/c/cuesta/eqnn-jax/benchmarks/galaxies/velocity_task_consistent_trees/"
    )
    output_path.mkdir(parents=True, exist_ok=True)

    # HYPERPARAMETERS
    boxsize = 1000.0
    use_pbcs = True
    k = 10
    n_radial = 0
    position_features = True
    use_3d_distances = True
    r_max = 0.6
    l_max = 1
    infill_value = -2.0
    d_hidden = 128
    n_layers = 3
    message_passing_steps = 3
    message_passing_agg = "sum"
    activation = "gelu"
    readout_agg = "mean"
    mlp_readout_widths = (4, 2, 2)
    residual = True
    n_steps = 2_500
    fraction_masked = args.fraction_masked
    batch_size = 32
    num_local_devices = jax.local_device_count()
    if args.training_set_size is not None:
        if args.training_set_size < batch_size:
            batch_size = args.training_set_size
        if args.training_set_size < num_local_devices:
            num_local_devices = args.training_set_size

    GNN_PARAMS = {
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "message_passing_steps": message_passing_steps,
        "message_passing_agg": message_passing_agg,
        "activation": activation,
        "norm": "none",
        "task": "node",
        "n_outputs": 3,
        "readout_agg": "mean",
        "mlp_readout_widths": mlp_readout_widths,
        "position_features": True,
        "residual": residual,
    }
    SEGNN_PARAMS = {
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "message_passing_steps": message_passing_steps,
        "message_passing_agg": message_passing_agg,
        "scalar_activation": activation,
        "gate_activation": "sigmoid",
        "task": "node",
        "output_irreps": e3nn.Irreps("1x1o"),
        "readout_agg": "mean",
        "mlp_readout_widths": mlp_readout_widths,
        "l_max_hidden": 1,
        "hidden_irreps": None,
        "residual": residual,
    }

    NEQUIP_PARAMS = {
        "d_hidden": d_hidden,
        "n_layers": n_layers,
        "message_passing_steps": message_passing_steps,
        "message_passing_agg": message_passing_agg,
        "task": "node",
        "irreps_out": e3nn.Irreps("1x1o"),
        "readout_agg": "mean",
        "mlp_readout_widths": mlp_readout_widths,
        "l_max": 1,
        # "n_outputs": 2,
        "n_radial_basis": n_radial,
        "r_cutoff": r_max,
        "sphharm_norm": "integral",
    }
    # LOAD DATA
    iterators, std = get_iterators(
        tfrecords_path=tfrecords_path,
        training_set_size=args.training_set_size,
        batch_size=batch_size,
    )

    apply_pbc = (
        get_apply_pbc(
            std=std / boxsize,
        )
        if use_pbcs
        else None
    )

    class GraphWrapperGNN(nn.Module):
        param_dict: Dict

        @nn.compact
        def __call__(self, x):
            return jax.vmap(GNN(**self.param_dict))(x)

    gnn = GraphWrapperGNN(
        GNN_PARAMS,
    )

    class GraphWrapper(nn.Module):
        param_dict: Dict

        @nn.compact
        def __call__(self, x):
            positions = e3nn.IrrepsArray("1o", x.nodes[..., :3])

            if x.nodes.shape[-1] == 3:
                nodes = e3nn.IrrepsArray("1o", x.nodes[..., :])
            else:
                nodes = e3nn.IrrepsArray("1o + 1o", x.nodes[..., :])
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
                lmax_attributes=l_max,
                apply_pbc=apply_pbc,
                n_radial_basis=n_radial,
                r_max=r_max,
            )

            return jax.vmap(SEGNN(**self.param_dict))(st_graph)

    segnn_model = GraphWrapper(
        SEGNN_PARAMS,
    )
    SEGNN_PARAMS_l2 = SEGNN_PARAMS.copy()
    SEGNN_PARAMS_l2["l_max_hidden"] = 2
    segnn_model_lmax2 = GraphWrapper(
        SEGNN_PARAMS_l2,
    )

    class GraphWrapperNequIP(nn.Module):
        param_dict: Dict

        @nn.compact
        def __call__(self, x):
            if x.nodes.shape[-1] == 3:
                #ones = jnp.ones(x.nodes[..., :].shape[:2] + (1,))
                #nodes = jnp.concatenate([x.nodes[..., :], ones], axis=-1)
                #nodes = e3nn.IrrepsArray("1o + 1x0e", nodes)
                nodes = e3nn.IrrepsArray("1o", x.nodes)
            else:
                nodes = e3nn.IrrepsArray("1o + 1o", x.nodes[..., :])
            graph = jraph.GraphsTuple(
                n_node=x.n_node,
                n_edge=x.n_edge,
                edges=None,
                globals=x.globals,
                nodes=nodes,
                senders=x.senders,
                receivers=x.receivers,
            )
            return jax.vmap(NequIP(**self.param_dict))(graph)

    nequip_model = GraphWrapperNequIP(
        NEQUIP_PARAMS,
    )
    NEQUIP_PARAMS_l2 = NEQUIP_PARAMS.copy()
    NEQUIP_PARAMS_l2["l_max"] = 2

    nequip_model_lmax2 = GraphWrapperNequIP(
        NEQUIP_PARAMS_l2,
    )
    fixed_build_graph = lambda x: build_graph(
        x,
        None,
        k=k,
        apply_pbc=apply_pbc,
        use_edges=True,
        n_radial_basis=n_radial,
        r_max=r_max,
        use_3d_distances=use_3d_distances,
    )

    def loss_mse(pred_batch, halo_batch, mask):
        # Only compute MSE based on mask (values which are 1)
        if isinstance(pred_batch, e3nn.IrrepsArray):
            pred_batch = (
                pred_batch.array
            )  # Euclidean distance is preserved by MSE, so we are safe doing this
        return jnp.sum(
            jnp.where(mask, (pred_batch - halo_batch[..., 3:6]) ** 2, 0.0)
        ) / jnp.sum(mask)

    # iterators, std = get_iterators(tfrecords_path=tfrecords_path, training_set_size=args.training_set_size, batch_size=batch_size,)
    # print(f'Training GNN with {args.training_set_size} samples and {args.fraction_masked} masked values')
    # train_model(
    #     iterators,
    #     model=gnn,
    #     n_steps=n_steps,
    #     fraction_masked=fraction_masked,
    #     num_local_devices=num_local_devices,
    #     save_model=True,
    #     ckpt_dir= output_path / f"gnn_{args.training_set_size}_f{fraction_masked:.1f}",
    # )
    iterators, std = get_iterators(
        tfrecords_path=tfrecords_path,
        training_set_size=args.training_set_size,
        batch_size=batch_size,
    )
    print(
        f"Training Nequip with {args.training_set_size} samples and {args.fraction_masked} masked values"
    )
    train_model(
        iterators,
        model=nequip_model,
        n_steps=n_steps,
        fraction_masked=fraction_masked,
        num_local_devices=num_local_devices,
        save_model=True,
        ckpt_dir=output_path
        / f"nequip_{args.training_set_size}_f{fraction_masked:.1f}",
    )

    iterators, std = get_iterators(
        tfrecords_path=tfrecords_path,
        training_set_size=args.training_set_size,
        batch_size=batch_size,
    )
    print(
        f"Training Nequip lmax=2 with {args.training_set_size} samples and {args.fraction_masked} masked values"
    )
    train_model(
        iterators,
        model=nequip_model_lmax2,
        n_steps=n_steps,
        fraction_masked=fraction_masked,
        num_local_devices=num_local_devices,
        save_model=True,
        ckpt_dir=output_path / f"gnn_{args.training_set_size}_f{fraction_masked:.1f}",
    )
    # print(f'Training SEGNN with {args.training_set_size} samples and {args.fraction_masked} masked values')
    # iterators, std = get_iterators(tfrecords_path=tfrecords_path, training_set_size=args.training_set_size, batch_size=batch_size,)
    # train_model(
    #     iterators,
    #     model=segnn_model,
    #     n_steps=n_steps,
    #     fraction_masked=fraction_masked,
    #     num_local_devices=num_local_devices,
    #     save_model=True,
    #     ckpt_dir= output_path / f"segnn_{args.training_set_size}_f{fraction_masked:.1f}",
    # )
    # print(f'Training SEGNN lmax=2 with {args.training_set_size} samples and {args.fraction_masked} masked values')
    # iterators, std = get_iterators(training_set_size=args.training_set_size, batch_size=batch_size,)
    # train_model(
    #     iterators,
    #     model=segnn_model_lmax2,
    #     n_steps=n_steps,
    #     fraction_masked=fraction_masked,
    #     num_local_devices=num_local_devices,
    #     save_model=True,
    #     ckpt_dir= output_path / f"segnn_l2_{args.training_set_size}",
    # )
