from pathlib import Path
from abc import ABC, abstractmethod
from typing import Callable, List, Optional, Tuple, Sequence, Union, NamedTuple

import numpy as np
import jax.numpy as jnp
import jax
import jax.tree_util as tree
import jraph
from jraph import GraphsTuple, segment_mean
import e3nn_jax as e3nn
import jax_dataloader as jdl



class BaseDataset(jdl.Dataset, ABC):
    """Abstract n-body dataset class."""

    def __init__(
        self,
        data_type,
        partition="train",
        max_samples=1e8,
        dataset_name="small",
        n_bodies=5,
        normalize=False,
        data_dir="data",
    ):
        self.partition = partition
        if self.partition == "val":
            self.suffix = "valid"
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        self.suffix += f"_{data_type}{n_bodies}_initvel1"
        self.data_type = data_type
        self.max_samples = int(max_samples)
        self.normalize = normalize

        self.data_dir = Path(data_dir)
        self.data = None

    def get_n_nodes(self):
        return self.data[0].shape[2]

    def _get_partition_frames(self) -> Tuple[int, int]:
        if self.dataset_name == "default":
            frame_0, frame_target = 6, 8
        elif self.dataset_name == "small":
            frame_0, frame_target = 30, 40
        elif self.dataset_name == "small_out_dist":
            frame_0, frame_target = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        return frame_0, frame_target

    def __len__(self) -> int:
        return len(self.data[0])

    def _load(self) -> Tuple[np.ndarray, ...]:
        loc = np.load(self.data_dir / ("loc_" + self.suffix + ".npy"))
        vel = np.load(self.data_dir / ("vel_" + self.suffix + ".npy"))
        edges = np.load(self.data_dir / ("edges_" + self.suffix + ".npy"))
        q = np.load(self.data_dir / ("charges_" + self.suffix + ".npy"))
        return loc, vel, edges, q

    def _normalize(self, x: np.ndarray) -> np.ndarray:
        std = x.std(axis=0)
        x = x - x.mean(axis=0)
        return np.divide(x, std, out=x, where=std != 0)

    @abstractmethod
    def load(self):
        raise NotImplementedError

    @abstractmethod
    def preprocess(self, *args) -> Tuple[np.ndarray, ...]:
        raise NotImplementedError


class ChargedDataset(BaseDataset):
    """N-body charged dataset class."""

    def __init__(
        self,
        partition="train",
        max_samples=1e8,
        dataset_name="small",
        n_bodies=5,
        normalize=False,
        data_dir="data",
    ):
        super().__init__(
            "charged",
            partition,
            max_samples,
            dataset_name,
            n_bodies,
            normalize,
            data_dir=data_dir,
        )
        self.data, self.edges = self.load()

    def preprocess(self, *args) -> Tuple[np.ndarray, ...]:
        # swap n_nodes - n_features dimensions
        loc, vel, edges, charges = args
        loc, vel = np.transpose(loc, (0, 1, 3, 2)), np.transpose(vel, (0, 1, 3, 2))
        n_nodes = loc.shape[2]
        loc = loc[0 : self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0 : self.max_samples, :, :, :]  # speed when starting the trajectory
        charges = charges[0 : self.max_samples]
        edge_attr = []

        # Initialize edges and edge_attributes
        rows, cols = [], []
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_attr.append(edges[:, i, j])
                    rows.append(i)
                    cols.append(j)
        edges = [rows, cols]
        # swap n_nodes - batch_size and add nf dimension
        edge_attr = np.array(edge_attr).T
        edge_attr = np.expand_dims(edge_attr, 2)

        if self.normalize:
            loc = self._normalize(loc)
            vel = self._normalize(vel)
            charges = self._normalize(charges)

        return loc, vel, edge_attr, edges, charges

    def load(self):
        loc, vel, edges, q = self._load()

        loc, vel, edge_attr, edges, charges = self.preprocess(loc, vel, edges, q)
        return (loc, vel, edge_attr, charges), edges

    def __getitem__(self, i: Union[Sequence, int]) -> Tuple[np.ndarray, ...]:
        frame_0, frame_target = self._get_partition_frames()
        loc, vel, edge_attr, charges = self.data
        loc, vel, edge_attr, charges, target_loc = (
            loc[i, frame_0],
            vel[i, frame_0],
            edge_attr[i],
            charges[i],
            loc[i, frame_target],
        )
        return loc, vel, edge_attr, charges, target_loc


class SteerableGraphsTuple(NamedTuple):
    """Pack (steerable) node and edge attributes with jraph.GraphsTuple."""

    graph: jraph.GraphsTuple
    node_attributes: Optional[e3nn.IrrepsArray] = None
    edge_attributes: Optional[e3nn.IrrepsArray] = None
    # NOTE: additional_message_features is in a separate field otherwise it would get
    #  updated by jraph.GraphNetwork. Actual graph edges are used only for the messages.
    additional_message_features: Optional[e3nn.IrrepsArray] = None



def NbodyGraphTransform(
    dataset_name: str,
    n_nodes: int,
    batch_size: int,
) -> Callable:
    """
    Build a function that converts torch DataBatch into SteerableGraphsTuple.
    """

    if dataset_name == "charged":
        # charged system is a connected graph
        edges = [(i, j) for i in range(n_nodes) for j in range(n_nodes) if i!=j] 
        full_edge_indices = jnp.array([edges for _ in range(batch_size)])

    def _to_steerable_graph(
        data: List, training: bool = True
    ) -> Tuple[SteerableGraphsTuple, jnp.ndarray]:
        _ = training
        loc, vel, _, q, targets = data
        # substract center of the system:
        loc = loc - loc.mean(axis=1, keepdims=True)
        senders, receivers = full_edge_indices[:,:,0], full_edge_indices[:,:, 1]
        vel_modulus = jnp.linalg.norm(vel, axis=-1, keepdims=True)
        print('charges = ', q)
        st_graph = SteerableGraphsTuple(
            graph=GraphsTuple(
                nodes=e3nn.IrrepsArray("1o + 1o + 2x0e",jnp.concatenate([loc, vel, vel_modulus, q], axis=-1)),
                edges=None,
                senders=senders,
                receivers=receivers,
                n_node=jnp.array([n_nodes] * len(loc)),
                n_edge=jnp.array([senders.shape[-1]]*len(loc)),
                globals=None,
            )
        )
        targets = targets - loc
        return st_graph, targets

    return _to_steerable_graph


def numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return jnp.vstack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [numpy_collate(samples) for samples in transposed]
    else:
        return jnp.array(batch)


def setup_nbody_data(
    node_irreps,
    additional_message_irreps,
    o3_layer="tpl",
    lmax_attributes=1,
    dataset="charged",
    dataset_name="small",
    max_samples=3000,
    n_bodies=5,
    batch_size=100,
    target="pos",
):
    if dataset == "charged":
        dataset_train = ChargedDataset(
            partition="train",
            dataset_name=dataset_name,
            max_samples=max_samples,
            n_bodies=n_bodies,
        )
        dataset_val = ChargedDataset(
            partition="val",
            dataset_name=dataset_name,
            n_bodies=n_bodies,
        )
        dataset_test = ChargedDataset(
            partition="test",
            dataset_name=dataset_name,
            n_bodies=n_bodies,
        )
    graph_transform = NbodyGraphTransform(
        n_nodes=n_bodies,
        batch_size=batch_size,
        dataset_name=dataset,
    )
    loader_train = jdl.DataLoader(
        dataset_train,
        backend="jax",
        batch_size=batch_size,
        shuffle=True,
    )
    loader_val = jdl.DataLoader(
        dataset_val,
        backend="jax",
        batch_size=batch_size,
        shuffle=False,
    )
    loader_test = jdl.DataLoader(
        dataset_test,
        backend="jax",
        batch_size=batch_size,
        shuffle=False,
    )
    return loader_train, loader_val, loader_test, graph_transform


if __name__ == "__main__":
    # 1) load ChargedDataset
    dataset = ChargedDataset()
    loc, vel, edge_attr, charges, target_loc = dataset[0]
    print("loc = ", loc.shape)
    print("vel = ", vel.shape)
    print("edge_attr = ", edge_attr.shape)
    print("charges = ", charges.shape)
    print("target_loc = ", target_loc.shape)

    # 2) Get all dataloaders
    node_irreps = e3nn.Irreps("2x1o + 1x0e")
    additional_message_irreps = e3nn.Irreps("2x0e")
    loader_train, loader_val, loader_test, graph_transform = setup_nbody_data(
        node_irreps=node_irreps,
        additional_message_irreps=additional_message_irreps,
    )
    print(next(iter(loader_train)))
