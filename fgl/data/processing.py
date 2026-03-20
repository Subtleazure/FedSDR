import torch
import numpy as np
import copy
from torch_geometric.utils import to_undirected
from torch_geometric.utils import add_random_edge


def random_topology_noise(splitted_data: dict, noise_prob: float = 0.1):

    edge_index = splitted_data.edge_index

    retained_edge_index, added_edge_index = add_random_edge(
        edge_index=edge_index, p=noise_prob, force_undirected=True)

    directed_edge_index_ids = (
        edge_index[0, :] > edge_index[1, :]).nonzero().squeeze().tolist()
    if type(directed_edge_index_ids) is not list:
        directed_edge_index_ids = [directed_edge_index_ids]

    num_remained = int(
        round(len(directed_edge_index_ids) * (1-noise_prob)))
    remained_ids = np.random.choice(
        directed_edge_index_ids, num_remained, replace=False)
    remained_edge_index = to_undirected(
        edge_index=edge_index[:, remained_ids])

    noised_edge_index = torch.hstack(
        (remained_edge_index, added_edge_index))

    noised_splitted_data = copy.deepcopy(splitted_data)
    noised_splitted_data.edge_index = noised_edge_index

    return noised_splitted_data
