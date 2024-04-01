from functools import partial
import pickle
import itertools
from pathlib import Path

from typing import Callable, Tuple, Dict

import haiku as hk
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
from jax import jit

from dataset import setup_nbody_data, SteerableGraphsTuple

import e3nn_jax as e3nn

import sys

sys.path.append("../../")
from models.segnn import SEGNN
from models.gnn import GNN
from models.egnn import EGNN
from models.utils.irreps_utils import weight_balanced_irreps


@partial(jit, static_argnames=["model_fn", "criterion", "eval_trn", "steerable", "egnn", "global_task"])
def loss_fn_wrapper(
    params: hk.Params,
    st_graph: SteerableGraphsTuple,
    target: jnp.ndarray,
    model_fn: Callable,
    criterion: Callable,
    steerable: bool,
    egnn: bool,
    global_task: bool = False,
) -> Tuple[float, hk.State]:
    if not global_task:
        if steerable:
            pred = model_fn(params, st_graph).nodes.array
        else:
            pred = model_fn(params, st_graph).nodes
            if egnn:
                pred = pred[...,:3]
    else:
        if steerable:
            pred = model_fn(params, st_graph).array.squeeze()
        else:
            pred = model_fn(params, st_graph)
    assert target.shape == pred.shape
    return jnp.mean(criterion(pred, target))


@partial(jit, static_argnames=["loss_fn", "opt_update"])
def update(
    params: hk.Params,
    graph: SteerableGraphsTuple,
    target: jnp.ndarray,
    opt_state: optax.OptState,
    loss_fn: Callable,
    opt_update: Callable,
) -> Tuple[float, hk.Params, hk.State, optax.OptState]:
    loss, grads = jax.value_and_grad(
        loss_fn,
    )(params, graph, target)
    updates, opt_state = opt_update(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state


def evaluate(
    loader,
    params: hk.Params,
    loss_fn: Callable,
    graph_transform: Callable,
) -> Tuple[float, float]:
    eval_loss = 0.0
    for data in loader:
        graph, target = graph_transform(data, training=False)
        loss = jax.lax.stop_gradient(loss_fn(params, graph, target))
        eval_loss += jax.block_until_ready(loss)
    return eval_loss / len(loader)


def train(
    key,
    segnn,
    loader_train,
    loader_val,
    loader_test,
    loss_fn,
    graph_transform,
    n_steps=10_000,#10_000_000,
    lr=5.0e-4,
    lr_scheduling=False,
    weight_decay=1.0e-12,
    eval_every=500,
):
    init_data = next(iter(loader_train))
    init_graph, _ = graph_transform(init_data)
    params = segnn.init(
        key,
        init_graph,
    )
    print(
        f"Starting {n_steps} steps"
        f" with {hk.data_structures.tree_size(params)} parameters.\n"
        "Jitting..."
    )

    # set up learning rate and optimizer
    learning_rate = lr
    if lr_scheduling:
        learning_rate = optax.piecewise_constant_schedule(
            learning_rate,
            boundaries_and_scales={
                int(n_steps * 0.7): 0.1,
                int(n_steps * 0.9): 0.1,
            },
        )
    opt_init, opt_update = optax.adamw(
        learning_rate=learning_rate, weight_decay=weight_decay
    )

    model_fn = jit(segnn.apply)

    loss_fn = partial(loss_fn, model_fn=model_fn)
    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)
    eval_fn = partial(evaluate, loss_fn=loss_fn, graph_transform=graph_transform)

    opt_state = opt_init(params)
    best_val = 1e10

    iter_loader = iter(loader_train)
    iter_loader = itertools.cycle(iter_loader)

    loss_curve = {
        'step': [],
        'train': [],
        'val': [],
    }
    for step in range(n_steps):
        data = next(iter_loader)
        graph, target = graph_transform(data)
        loss, params, opt_state = update_fn(
            params=params,
            graph=graph,
            target=target,
            opt_state=opt_state,
        )
        if step % eval_every == 0:
            print(
                f"[Step {step}] train loss {loss:.6f}",
                end="",
            )
            print()
            val_loss = eval_fn(
                loader_val,
                params=params,
            )
            if val_loss < best_val:
                best_val = val_loss
                tag = " (best)"
                test_loss_ckp = eval_fn(
                    loader_test,
                    params=params,
                )
            else:
                tag = ""
            loss_curve['step'].append(step)
            loss_curve['train'].append(loss)
            loss_curve['val'].append(val_loss)

            print(f" - val loss {val_loss:.6f}{tag}", end="")
            print()

    test_loss = eval_fn(
        loader_test,
        params=params,
    )
    loss_curve['test'] = test_loss
    # ignore compilation time
    print(
        "Training done.\n"
        f"Final test loss {test_loss:.6f} - checkpoint test loss {test_loss_ckp:.6f}.\n"
    )
    return loss_curve


class GraphWrapper(nn.Module):
    model_name: str
    param_dict: Dict

    @nn.compact
    def __call__(self, x):
        if self.model_name == "SEGNN":
            return jax.vmap(SEGNN(**self.param_dict))(x)
        elif self.model_name == "GNN":
            return jax.vmap(GNN(**self.param_dict))(x)
        elif self.model_name == "EGNN":
            return jax.vmap(EGNN(**self.param_dict))(x)
        else:
            raise ValueError("Please specify a valid model name.")


if __name__ == "__main__":
    out_path = Path('/n/holystore01/LABS/itc_lab/Users/ccuestalazaro/benchmark_equivariance_results')
    key = jax.random.PRNGKey(1337)
    model = 'segnn'
    task = "graph"
    steerable = True if model == 'segnn' else False 
    hidden_units = 64
    lmax_attributes = 1
    lmax_hidden = 1
    batch_size = 100
    node_irreps = e3nn.Irreps("2x1o + 1x0e")
    output_irreps = e3nn.Irreps("1x1o")
    additional_message_irreps = e3nn.Irreps("2x0e")
    attr_irreps = e3nn.Irreps.spherical_harmonics(lmax_attributes)
    # hidden_irreps = balanced_irreps(lmax=lmax_hidden, feature_size=hidden_units, use_sh=True)
    # TODO: Why so many scalars?
    hidden_irreps = weight_balanced_irreps(
        lmax=lmax_hidden,
        scalar_units=hidden_units,
        irreps_right=attr_irreps,
    )
    if model == 'segnn':
        hparams = {
            "d_hidden": hidden_units,
            "l_max_hidden": lmax_hidden,
            "num_blocks": 2,
            "num_message_passing_steps": 7,
            "intermediate_hidden_irreps": True,
            "task": task,
            "output_irreps": output_irreps,
            "hidden_irreps": hidden_irreps,
            "normalize_messages": False,
            "message_passing_agg": "sum",
        }
        gnn = GraphWrapper(
            model_name="SEGNN",
            param_dict=hparams,
        )
    elif model == 'gnn':
        hparams = {
            "d_hidden": hidden_units,
            "n_layers": 2,
            "message_passing_steps": 7,
            "task": task,
            "message_passing_agg": "sum",
            "d_output": 3,
            "n_outputs": 3,
        }
        gnn = GraphWrapper(
            model_name="GNN",
            param_dict=hparams,
        )
    elif model == 'egnn':
        hparams = {
            "d_hidden": hidden_units,
            "n_layers": 2,
            "message_passing_steps": 7,
            "task": task,
            "message_passing_agg": "sum",
            "positions_only": False,
            'tanh_out': False,
            "n_outputs": 3,
        }
        gnn = GraphWrapper(
            model_name="EGNN",
            param_dict=hparams,
        )
    else:
        raise ValueError(f'{model} model is not implemented')
    print(hparams)

    loader_train, loader_val, loader_test, graph_transform = setup_nbody_data(
        lmax_attributes=lmax_attributes,
        batch_size=batch_size,
        steerable=steerable,
        task=task,
    )

    def _mse(p, t):
        return jnp.power(p - t, 2)

    loss_fn = partial(loss_fn_wrapper, criterion=_mse, steerable=steerable, egnn=True if model=='egnn' else False, global_task=True if task == 'graph' else False)

    loss_curve = train(
        key=key,
        segnn=gnn,
        loader_train=loader_train,
        loader_val=loader_val,
        loader_test=loader_test,
        loss_fn=loss_fn,
        graph_transform=graph_transform,
    )
    # store loss curve with pickle
    with open(out_path / f'task={task}_model={model}.pkl', 'wb') as f:
        pickle.dump(loss_curve, f)

