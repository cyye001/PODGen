import random
from typing import Optional, Sequence
from pathlib import Path
import pandas as pd
from torch_geometric.data import Data

import hydra
import os
import numpy as np
import omegaconf
import pytorch_lightning as pl
from omegaconf import ValueNode
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset
from torch_geometric.data import DataLoader

from utils import PROJECT_ROOT
from data_utils import get_scaler_from_data_list
from data_utils import (preprocess, preprocess_tensors, add_scaled_lattice_prop)

from torch.utils.data.distributed import DistributedSampler


def worker_init_fn(id: int):
    """
    DataLoaders workers init function.

    Initialize the numpy.random seed correctly for each worker, so that
    random augmentations between workers and/or epochs are not identical.

    If a global seed is set, the augmentations are deterministic.

    https://pytorch.org/docs/stable/notes/randomness.html#dataloader
    """
    uint64_seed = torch.initial_seed()
    ss = np.random.SeedSequence([uint64_seed])
    # More than 128 bits (4 32-bit words) would be overkill.
    np.random.seed(ss.generate_state(4))
    random.seed(uint64_seed)


class CrystDataModule():
    def __init__(
        self,
        accelerator,
        datasets: DictConfig,
        num_workers: DictConfig,
        batch_size: DictConfig,
        scaler_path=None,
    ):
        super().__init__()
        self.datasets = datasets
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.accelerator = accelerator

        self.train_dataset: Optional[Dataset] = None
        self.val_datasets: Optional[Sequence[Dataset]] = None
        self.test_datasets: Optional[Sequence[Dataset]] = None

        self.get_scaler(scaler_path)

    def prepare_data(self) -> None:
        # download only
        pass

    def get_scaler(self, scaler_path):
        # Load once to compute property scaler
        if scaler_path is None:
            train_path = self.datasets['train']['path']
            train_path = os.path.dirname(train_path)
            train_path = os.path.join(train_path, 'train_data.pt')
            # if (os.path.exists(train_path)):
            #     # train_dataset = torch.load(train_path)
            #     train_dataset = hydra.utils.instantiate(self.datasets.train)
            # else:
            train_dataset = hydra.utils.instantiate(self.datasets.train)
            # torch.save(train_dataset, train_path)
            self.lattice_scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key='scaled_lattice')
            self.scaler = get_scaler_from_data_list(
                train_dataset.cached_data,
                key=train_dataset.use_prop)
        else:
            self.lattice_scaler = torch.load(
                Path(scaler_path) / 'lattice_scaler.pt')
            self.scaler = torch.load(Path(scaler_path) / 'prop_scaler.pt')

    def setup(self, stage: Optional[str] = None):
        """
        construct datasets and assign data scalers.
        """
        if stage is None or stage == "fit":
            train_path = self.datasets['train']['path']
            train_path = os.path.dirname(train_path)
            train_path = os.path.join(train_path, 'train_data.pt')
            # if (os.path.exists(train_path)):
            #     self.train_dataset = torch.load(train_path)
            # else:
            self.train_dataset = hydra.utils.instantiate(self.datasets.train)
            # torch.save(self.train_dataset, train_path)

            val_path = self.datasets['val'][0]['path']
            val_path = os.path.dirname(val_path)
            val_path = os.path.join(val_path, 'val_data.pt')
            # if (os.path.exists(val_path)):
            #     self.val_datasets = [torch.load(val_path)]
            # else:
            self.val_datasets = [hydra.utils.instantiate(dataset_cfg)
                                 for dataset_cfg in self.datasets.val]
            # torch.save(self.val_datasets[0], val_path)

            self.train_dataset.lattice_scaler = self.lattice_scaler
            self.train_dataset.scaler = self.scaler
            for val_dataset in self.val_datasets:
                val_dataset.lattice_scaler = self.lattice_scaler
                val_dataset.scaler = self.scaler

        if stage is None or stage == "test":
            test_path = self.datasets['test'][0]['path']
            test_path = os.path.dirname(test_path)
            test_path = os.path.join(test_path, 'test_data.pt')
            # if (os.path.exists(test_path)):
            #     self.test_datasets = [torch.load(test_path)]
            # else:
            self.test_datasets = [hydra.utils.instantiate(dataset_cfg)
                                  for dataset_cfg in self.datasets.val]
            # torch.save(self.test_datasets[0], test_path)
            for test_dataset in self.test_datasets:
                test_dataset.lattice_scaler = self.lattice_scaler
                test_dataset.scaler = self.scaler

    def train_dataloader(self, shuffle = True) -> DataLoader:
        if self.accelerator == 'DDP':
            train_shuffle = False
            train_sampler = DistributedSampler(self.train_dataset)
        else:
            train_shuffle = True
            train_sampler = None

        return DataLoader(
            self.train_dataset,
            shuffle=train_shuffle,
            batch_size=self.batch_size.train,
            num_workers=self.num_workers.train,
            worker_init_fn=worker_init_fn,
            sampler=train_sampler,
        )

    def val_dataloader(self) -> Sequence[DataLoader]:
        if self.accelerator == 'DDP':
            val_samplers = [DistributedSampler(dataset) for dataset in self.val_datasets]
        else:
            val_samplers = [None for dataset in self.val_datasets]

        return [
            DataLoader(
                self.val_datasets[i],
                shuffle=False,
                batch_size=self.batch_size.val,
                num_workers=self.num_workers.val,
                worker_init_fn=worker_init_fn,
                sampler=val_samplers[i]
            )
            for i in range(len(self.val_datasets))]

    def test_dataloader(self) -> Sequence[DataLoader]:
        if self.accelerator == 'DDP':
            test_samplers = [DistributedSampler(dataset) for dataset in self.test_datasets]
        else:
            test_samplers = [None for dataset in self.test_datasets]

        return [
            DataLoader(
                self.test_datasets[i],
                shuffle=False,
                batch_size=self.batch_size.test,
                num_workers=self.num_workers.test,
                worker_init_fn=worker_init_fn,
                sampler=test_samplers[i]
            )
            for i in range(len(self.test_datasets))]

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"{self.datasets=}, "
            f"{self.num_workers=}, "
            f"{self.batch_size=})"
        )



class CrystDataset(Dataset):
    def __init__(self, name: ValueNode, path: ValueNode,
                 prop: ValueNode, use_prop: list, num_targets: ValueNode, niggli: ValueNode, primitive: ValueNode,
                 graph_method: ValueNode, preprocess_workers: ValueNode,
                 lattice_scale_method: ValueNode, save_path: ValueNode, tolerance: ValueNode, use_space_group: ValueNode, use_pos_index: ValueNode,
                 **kwargs):
        super().__init__()
        self.path = path
        self.name = name
        self.df = pd.read_csv(path)
        self.prop = prop
        self.use_prop = use_prop
        self.num_targets = num_targets
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method
        self.use_space_group = use_space_group
        self.use_pos_index = use_pos_index
        self.tolerance = tolerance

        self.preprocess(save_path, preprocess_workers, prop)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def preprocess(self, save_path, preprocess_workers, prop):
        if os.path.exists(save_path):
            self.cached_data = torch.load(save_path)
        else:
            cached_data = preprocess(
            self.path,
            preprocess_workers,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method,
            prop_list=list(prop),
            use_space_group=self.use_space_group,
            tol=self.tolerance)
            torch.save(cached_data, save_path)
            self.cached_data = cached_data

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        # scaler is set in DataModule set stage
        # prop = self.scaler.transform(data_dict[self.use_prop])
        prop = data_dict[self.use_prop]
        if self.num_targets > 1.5:
            prop1 = [0]*int(self.num_targets)
            prop1[int(prop.item())] = 1
            prop = torch.Tensor(prop1).view(1, -1)

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
            y=prop.view(1, -1),
        )

        # if self.use_space_group:
        #     data.spacegroup = torch.LongTensor([data_dict['spacegroup']])
        #     data.ops = torch.Tensor(data_dict['wyckoff_ops'])
        #     data.anchor_index = torch.LongTensor(data_dict['anchors'])
        #     data.ops_inv = torch.linalg.pinv(data.ops[:,:3,:3])

        if self.use_pos_index:
            pos_dic = {}
            indexes = []
            for atom in atom_types:
                pos_dic[atom] = pos_dic.get(atom, 0) + 1
                indexes.append(pos_dic[atom] - 1)
            data.index = torch.LongTensor(indexes)

        filtered_data = {key: data_dict[key] for key in self.prop}
        data.update(filtered_data)

        return data

    def __repr__(self) -> str:
        return f"CrystDataset({self.name=}, {self.path=})"


class TensorCrystDataset(Dataset):
    def __init__(self, crystal_array_list, niggli, primitive,
                 graph_method, preprocess_workers,
                 lattice_scale_method, **kwargs):
        super().__init__()
        self.niggli = niggli
        self.primitive = primitive
        self.graph_method = graph_method
        self.lattice_scale_method = lattice_scale_method

        self.cached_data = preprocess_tensors(
            crystal_array_list,
            niggli=self.niggli,
            primitive=self.primitive,
            graph_method=self.graph_method)

        add_scaled_lattice_prop(self.cached_data, lattice_scale_method)
        self.lattice_scaler = None
        self.scaler = None

    def __len__(self) -> int:
        return len(self.cached_data)

    def __getitem__(self, index):
        data_dict = self.cached_data[index]

        (frac_coords, atom_types, lengths, angles, edge_indices,
         to_jimages, num_atoms) = data_dict['graph_arrays']

        # atom_coords are fractional coordinates
        # edge_index is incremented during batching
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/batching.html
        data = Data(
            frac_coords=torch.Tensor(frac_coords),
            atom_types=torch.LongTensor(atom_types),
            lengths=torch.Tensor(lengths).view(1, -1),
            angles=torch.Tensor(angles).view(1, -1),
            edge_index=torch.LongTensor(
                edge_indices.T).contiguous(),  # shape (2, num_edges)
            to_jimages=torch.LongTensor(to_jimages),
            num_atoms=num_atoms,
            num_bonds=edge_indices.shape[0],
            num_nodes=num_atoms,  # special attribute used for batching in pytorch geometric
        )
        return data

    def __repr__(self) -> str:
        return f"TensorCrystDataset(len: {len(self.cached_data)})"


@hydra.main(config_path=str(PROJECT_ROOT / "conf"), config_name="default")
def main(cfg: omegaconf.DictConfig):
    datamodule =  hydra.utils.instantiate(
        cfg.data.datamodule, _recursive_=False
    )
    datamodule.setup('fit')
    import pdb
    pdb.set_trace()


if __name__ == "__main__":
    main()
