import os
from os import path as osp
from fgl.data.global_dataset_loader import load_global_dataset
from torch_geometric.data import Dataset
from torch_geometric.utils import remove_self_loops, to_undirected
import copy
import torch
import json


class FGLDataset(Dataset):

    def __init__(self, args, transform=None, pre_transform=None, pre_filter=None):
        self.check_args(args)
        self.args = args
        super(FGLDataset, self).__init__(
            args.root, transform, pre_transform, pre_filter)
        self.load_data()

    @property
    def global_root(self) -> str:
        return osp.join(self.root, "global")

    @property
    def distrib_root(self) -> str:
        return osp.join(self.root, "distrib")

    @property
    def raw_dir(self) -> str:
        return self.root

    def check_args(self, args):
        from fgl.config import supported_subgraph_fl_datasets
        for dataset in args.dataset:
            assert dataset in supported_subgraph_fl_datasets, f"Invalid subgraph_fl dataset '{dataset}'."

    @property
    def processed_dir(self) -> str:
        simulation_name = f"subgraph_fl_louvain_{self.args.louvain_resolution}"

        fmt_dataset_list = copy.deepcopy(self.args.dataset)
        fmt_dataset_list = sorted(fmt_dataset_list)

        return osp.join(self.distrib_root,
                        "_".join([simulation_name, "_".join(fmt_dataset_list), f"client_{self.args.num_clients}"]))

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self) -> str:
        files_names = ["data_{}.pt".format(i)
                       for i in range(self.args.num_clients)]
        return files_names

    def get_client_data(self, client_id):
        data = torch.load(osp.join(self.processed_dir,
                          "data_{}.pt".format(client_id)))
        if hasattr(data, "x"):
            data.x = data.x.to(torch.float32)
        if hasattr(data, "y"):
            data.y = data.y.squeeze()
        if hasattr(data, "edge_attr"):
            data.edge_index, data.edge_attr = remove_self_loops(
                *to_undirected(data.edge_index, data.edge_attr))
        else:
            data.edge_index = remove_self_loops(
                to_undirected(data.edge_index))[0]
        data.edge_index = data.edge_index.to(torch.int64)
        data._data_list = None
        return data

    def save_client_data(self, data, client_id):
        torch.save(data, osp.join(self.processed_dir,
                   "data_{}.pt".format(client_id)))

    def process(self):
        if len(self.args.dataset) == 1:
            global_dataset = load_global_dataset(
                self.global_root, dataset=self.args.dataset[0])
        else:
            global_dataset = [load_global_dataset(
                self.global_root, dataset=dataset_i) for dataset_i in self.args.dataset]

        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        from fgl.data.simulation import subgraph_fl_louvain
        self.local_data = subgraph_fl_louvain(self.args, global_dataset)

        for client_id in range(self.args.num_clients):
            self.save_client_data(self.local_data[client_id], client_id)

        self.save_dataset_description()

    def save_dataset_description(self):
        file_path = os.path.join(self.processed_dir, "description.txt")
        args_str = json.dumps(vars(self.args), indent=4)
        with open(file_path, 'w') as file:
            file.write(args_str)
            print(f"Saved dataset arguments to {file_path}.")

    def load_data(self):
        self.local_data = [self.get_client_data(
            client_id) for client_id in range(self.args.num_clients)]

        if len(self.args.dataset) == 1:
            global_dataset = load_global_dataset(
                self.global_root, dataset=self.args.dataset[0])
            self.global_data = global_dataset
            if hasattr(self.global_data, "x"):
                self.global_data.x = self.global_data.x.to(torch.float32)
            if hasattr(self.global_data, "y"):
                self.global_data.y = self.global_data.y.squeeze()
            if hasattr(self.global_data, "edge_attr"):
                self.global_data.edge_index, self.global_data.edge_attr = remove_self_loops(
                    *to_undirected(self.global_data.edge_index, self.global_data.edge_attr))
            else:
                self.global_data.edge_index = remove_self_loops(
                    to_undirected(self.global_data.edge_index))[0]
            self.global_data._data_list = None

            self.global_data.num_global_classes = global_dataset.num_classes
        else:
            self.global_data = None
