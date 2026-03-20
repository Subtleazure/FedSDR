import torch
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.data import Data
from sknetwork.clustering import Louvain
import torch_geometric.utils
from tqdm import tqdm
import torch_geometric


def get_subgraph_pyg_data(global_dataset, node_list):
    global_edge_index = global_dataset.edge_index
    node_id_set = set(node_list)
    global_id_to_local_id = {}
    local_id_to_global_id = {}
    local_edge_list = []
    for local_id, global_id in enumerate(node_list):
        global_id_to_local_id[global_id] = local_id
        local_id_to_global_id[local_id] = global_id

    for edge_id in tqdm(range(global_edge_index.shape[1]), desc="Processing Edge Mapping"):
        src = global_edge_index[0, edge_id].item()
        tgt = global_edge_index[1, edge_id].item()
        if src in node_id_set and tgt in node_id_set:
            local_id_src = global_id_to_local_id[src]
            local_id_tgt = global_id_to_local_id[tgt]
            local_edge_list.append((local_id_src, local_id_tgt))

    local_edge_index = torch.tensor(local_edge_list).T

    local_subgraph = Data(
        x=global_dataset.x[node_list], edge_index=local_edge_index, y=global_dataset.y[node_list])
    local_subgraph.global_map = local_id_to_global_id

    if hasattr(global_dataset, "num_classes"):
        local_subgraph.num_global_classes = global_dataset.num_classes
    else:
        local_subgraph.num_global_classes = global_dataset.num_global_classes
    return local_subgraph


def subgraph_fl_louvain(args, global_dataset):
    louvain = Louvain(modularity='newman',
                      resolution=args.louvain_resolution, return_aggregate=True)
    num_nodes = global_dataset[0].x.shape[0]
    adj_csr = to_scipy_sparse_matrix(global_dataset[0].edge_index)
    from scipy.sparse import csr_matrix

    if not isinstance(adj_csr, csr_matrix):
        adj_csr = csr_matrix(adj_csr)

    fit_result = louvain.fit_predict(adj_csr)
    partition = {}
    for node_id, com_id in enumerate(fit_result):
        partition[node_id] = int(com_id)

    groups = []

    for key in partition.keys():
        if partition[key] not in groups:
            groups.append(partition[key])
    print(groups)
    partition_groups = {group_i: [] for group_i in groups}

    for key in partition.keys():
        partition_groups[partition[key]].append(key)

    group_len_max = num_nodes // args.num_clients - args.louvain_delta
    for group_i in groups:
        while len(partition_groups[group_i]) > group_len_max:
            long_group = list.copy(partition_groups[group_i])
            partition_groups[group_i] = list.copy(long_group[:group_len_max])
            new_grp_i = max(groups) + 1
            groups.append(new_grp_i)
            partition_groups[new_grp_i] = long_group[group_len_max:]
    print(groups)

    len_list = []
    for group_i in groups:
        len_list.append(len(partition_groups[group_i]))

    len_dict = {}

    for i in range(len(groups)):
        len_dict[groups[i]] = len_list[i]
    sort_len_dict = {
        k: v
        for k, v in sorted(len_dict.items(), key=lambda item: item[1], reverse=True)
    }

    owner_node_ids = {owner_id: [] for owner_id in range(args.num_clients)}

    owner_nodes_len = num_nodes // args.num_clients
    owner_list = [i for i in range(args.num_clients)]
    owner_ind = 0

    give_up = 1000

    for group_i in sort_len_dict.keys():
        while (
            len(owner_list) >= 2
            and len(owner_node_ids[owner_list[owner_ind]]) >= owner_nodes_len
        ):
            owner_list.remove(owner_list[owner_ind])
            owner_ind = owner_ind % len(owner_list)
        cnt = 0
        while (
            len(owner_node_ids[owner_list[owner_ind]]) +
                len(partition_groups[group_i])
            >= owner_nodes_len + args.louvain_delta
        ):
            owner_ind = (owner_ind + 1) % len(owner_list)
            cnt += 1
            if cnt > give_up:
                cnt = 0
                min_v = 1e15
                for i in range(len(owner_list)):
                    if len(owner_node_ids[owner_list[owner_ind]]) < min_v:
                        min_v = len(owner_node_ids[owner_list[owner_ind]])
                        owner_ind = i
                break

        owner_node_ids[owner_list[owner_ind]] += partition_groups[group_i]

    local_data = []
    for client_id in range(args.num_clients):
        local_subgraph = get_subgraph_pyg_data(
            global_dataset, owner_node_ids[client_id])
        if local_subgraph.edge_index.dim() == 1:
            local_subgraph.edge_index, _ = torch_geometric.utils.add_random_edge(
                local_subgraph.edge_index.view(2, -1))
        local_data.append(local_subgraph)

    return local_data
