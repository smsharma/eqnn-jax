import numpy as np
from pathlib import Path
import os

class GravityDataset():
    """
    NBodyDataset

    """

    def __init__(self, n=5, partition='train', max_samples=1e8, dataset_name="nbody_small", neighbours=6, target="pos", data_path = 'data'):
        self.partition = partition
        self.data_path = data_path
        if self.partition == 'val':
            self.suffix = 'valid'
        else:
            self.suffix = self.partition
        self.dataset_name = dataset_name
        if dataset_name == "nbody":
            self.suffix += f"_gravity{n}_initvel1"
        elif dataset_name == "nbody_small" or dataset_name == "nbody_small_out_dist":
            self.suffix += f"_gravity{n}_initvel1small"
        else:
            raise Exception("Wrong dataset name %s" % self.dataset_name)

        assert target in ["pos", "force"]

        self.max_samples = int(max_samples)
        self.dataset_name = dataset_name
        self.data = self.load()
        self.neighbours = int(neighbours)
        self.target = target
        if self.dataset_name == "nbody":
            self.d_frame = 2
        elif self.dataset_name == "nbody_small":
            self.d_frame = 10
        elif self.dataset_name == "nbody_small_out_dist":
            self.d_frame = 10
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

    def load(self):
        filepath = Path(self.data_path)

        loc = np.load(filepath  / f'loc_{self.suffix}.npy')
        vel = np.load(filepath  / f'vel_{self.suffix}.npy')
        force = np.load(filepath  / f'edges_{self.suffix}.npy')
        mass = np.load(filepath  / f'charges_{self.suffix}.npy')
        self.num_nodes = loc.shape[-1]

        loc, vel, force, mass = self.preprocess(loc, vel, force, mass)
        return (loc, vel, force, mass)

    def preprocess(self, loc, vel, force, mass):
        # cast to torch and swap n_nodes <--> n_features dimensions
        loc = torch.Tensor(loc).transpose(2, 3)
        vel = torch.Tensor(vel).transpose(2, 3)
        force = torch.Tensor(force).transpose(2, 3)
        loc = loc[0:self.max_samples, :, :, :]  # limit number of samples
        vel = vel[0:self.max_samples, :, :, :]  # speed when starting the trajectory
        force = force[0:self.max_samples, :, :, :]

        return loc, vel, force, torch.Tensor(mass)

    def set_max_samples(self, max_samples):
        self.max_samples = int(max_samples)
        self.data, self.edges = self.load()

    def get_n_nodes(self):
        return self.data[0].size(1)

    def __getitem__(self, i):
        loc, vel, force, mass = self.data
        loc, vel, force, mass = loc[i], vel[i], force[i], mass[i]

        if self.dataset_name == "nbody":
            frame_0, frame_T = 6, 8
        elif self.dataset_name == "nbody_small":
            frame_0, frame_T = 30, 40
        elif self.dataset_name == "nbody_small_out_dist":
            frame_0, frame_T = 20, 30
        else:
            raise Exception("Wrong dataset partition %s" % self.dataset_name)

        if self.target == "pos":
            y = loc[frame_T]
        elif self.target == "force":
            y = force[frame_T]

        return loc[frame_0], vel[frame_0], force[frame_0], mass, y

    def __len__(self):
        return self.data[0].shape[0]

if __name__ == "__main__":
    dataset = GravityDataset(dataset_name="nbody")

    for item in dataset:
        loc0, vel0, force0, mass, locT = item
        print(loc0.shape, vel0.shape, force0.shape, mass.shape, locT.shape)
        break