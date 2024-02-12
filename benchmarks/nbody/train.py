import time
from functools import partial
from typing import Callable, Tuple

import haiku as hk
import jax
import jax.numpy as jnp
import jraph
import optax
from jax import jit

from dataset import setup_nbody_data, SteerableGraphsTuple
from math import prod

from e3nn_jax import Irreps
import e3nn_jax as e3nn

import sys

sys.path.append("../../")
from models.segnn import SEGNN
from models.utils.irreps_utils import balanced_irreps


def weight_balanced_irreps(
    scalar_units: int, irreps_right: Irreps, use_sh: bool = True, lmax: int = None
) -> Irreps:
    """
    Determines irreps_left such that the parametrized tensor product
        Linear(tensor_product(irreps_left, irreps_right))
    has (at least) scalar_units weights.

    Args:
        scalar_units: number of desired weights
        irreps_right: irreps of the right tensor
        use_sh: whether to use spherical harmonics
        lmax: maximum level of spherical harmonics
    """
    # irrep order
    if lmax is None:
        lmax = irreps_right.lmax
    # linear layer with squdare weight matrix
    linear_weights = scalar_units**2
    # raise hidden features until enough weigths
    n = 0
    while True:
        n += 1
        if use_sh:
            irreps_left = (
                (Irreps.spherical_harmonics(lmax) * n).sort().irreps.simplify()
            )
        else:
            irreps_left = balanced_irreps(lmax, n)
        # number of paths
        tp_weights = sum(
            prod([irreps_left[i_1].mul ** 2, irreps_right[i_2].mul])
            for i_1, (_, ir_1) in enumerate(irreps_left)
            for i_2, (_, ir_2) in enumerate(irreps_right)
            for _, (_, ir_out) in enumerate(irreps_left)
            if ir_out in ir_1 * ir_2
        )
        if tp_weights >= linear_weights:
            break
    return Irreps(irreps_left)


@partial(jit, static_argnames=["model_fn", "criterion", "do_mask", "eval_trn"])
def loss_fn_wrapper(
    params: hk.Params,
    st_graph: SteerableGraphsTuple,
    target: jnp.ndarray,
    model_fn: Callable,
    criterion: Callable,
    do_mask: bool = True,
    eval_trn: Callable = None,
) -> Tuple[float, hk.State]:
    pred = model_fn(params, st_graph).nodes
    if eval_trn is not None:
        pred = eval_trn(pred)

    if do_mask:
        if target.shape == st_graph.graph.nodes.shape:
            mask = jraph.get_node_padding_mask(st_graph.graph)
        else:
            mask = jraph.get_graph_padding_mask(st_graph.graph)
        # broadcast mask for vector targets
        if len(pred.shape) == 2:
            mask = mask[:, jnp.newaxis]
    else:
        mask = jnp.ones_like(target)

    target = target * mask
    pred = pred * mask

    assert target.shape == pred.shape
    return jnp.sum(criterion(pred, target)) / jnp.count_nonzero(mask)


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
    state: hk.State,
    loss_fn: Callable,
    graph_transform: Callable,
) -> Tuple[float, float]:
    eval_loss = 0.0
    eval_times = 0.0
    for data in loader:
        graph, target = graph_transform(data, training=False)
        eval_start = time.perf_counter_ns()
        loss, _ = jax.lax.stop_gradient(loss_fn(params, state, graph, target))
        eval_loss += jax.block_until_ready(loss)
        eval_times += (time.perf_counter_ns() - eval_start) / 1e6
    return eval_times / len(loader), eval_loss / len(loader)


def train(
    key,
    segnn,
    loader_train,
    loader_val,
    loader_test,
    loss_fn,
    eval_loss_fn,
    graph_transform,
    n_steps=1_000,
    lr=5.0e-3,
    lr_scheduling=True,
    weight_decay=1.0e-12,
    eval_every=100,
):
    init_graph, _ = graph_transform(next(iter(loader_train)))
    params = segnn.init(key, init_graph)
    print(
        f"Starting {n_steps} steps"
        f"with {hk.data_structures.tree_size(params)} parameters.\n"
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
    eval_loss_fn = partial(eval_loss_fn, model_fn=model_fn)
    update_fn = partial(update, loss_fn=loss_fn, opt_update=opt_update)
    eval_fn = partial(evaluate, loss_fn=eval_loss_fn, graph_transform=graph_transform)

    opt_state = opt_init(params)
    best_val = 1e10

    iter_loader = iter(loader_train)

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
    """
            eval_time, val_loss = eval_fn(loader_val, params, segnn_state)
            if val_loss < best_val:
                best_val = val_loss
                tag = " (best)"
                _, test_loss_ckp = eval_fn(loader_test, params, segnn_state)

            print(f" - val loss {val_loss:.6f}{tag}, infer {eval_time:.2f}ms", end="")
        print()

    test_loss = 0
    _, test_loss = eval_fn(loader_test, params, segnn_state)
    # ignore compilation time
    avg_time = avg_time[1:] if len(avg_time) > 1 else avg_time
    avg_time = sum(avg_time) / len(avg_time)
    print(
        "Training done.\n"
        f"Final test loss {test_loss:.6f} - checkpoint test loss {test_loss_ckp:.6f}.\n"
        f"Average (model) eval time {avg_time:.2f}ms"
    )
    """


if __name__ == "__main__":
    key = jax.random.PRNGKey(1337)
    task = "node"
    hidden_units = 64
    lmax_attributes = 1
    lmax_hidden = 1
    num_message_passing_steps = 4
    intermediate_hidden_irreps = True
    batch_size = 1

    node_irreps = e3nn.Irreps("2x1o + 1x0e")
    output_irreps = e3nn.Irreps("1x1o")
    additional_message_irreps = e3nn.Irreps("2x0e")
    attr_irreps = e3nn.Irreps.spherical_harmonics(lmax_attributes)

    hidden_irreps = weight_balanced_irreps(
        scalar_units=hidden_units,
        irreps_right=attr_irreps,
        use_sh=True,
        lmax=lmax_hidden,
    )
    segnn = SEGNN(
        num_message_passing_steps=num_message_passing_steps,
        intermediate_hidden_irreps=intermediate_hidden_irreps,
        task=task,
    )

    loader_train, loader_val, loader_test, graph_transform = setup_nbody_data(
        node_irreps=node_irreps,
        additional_message_irreps=additional_message_irreps,
        batch_size=batch_size,
    )

    def _mse(p, t):
        return jnp.power(p - t, 2)

    train_loss = partial(loss_fn_wrapper, criterion=_mse, do_mask=False)
    eval_loss = partial(loss_fn_wrapper, criterion=_mse, do_mask=False)

    train(
        key,
        segnn,
        loader_train,
        loader_val,
        loader_test,
        train_loss,
        eval_loss,
        graph_transform,
    )
