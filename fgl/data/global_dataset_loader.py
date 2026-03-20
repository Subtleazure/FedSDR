from os import path as osp


def load_global_dataset(root, dataset):
    if dataset in ["PubMed"]:
        from torch_geometric.datasets import Planetoid
        return Planetoid(root=osp.join(root, "subgraph_fl"), name=dataset)
    elif dataset in ["CS", "Physics"]:
        from torch_geometric.datasets import Coauthor
        return Coauthor(root=osp.join(root, "subgraph_fl"), name=dataset)
    elif dataset in ["Roman-empire"]:
        from torch_geometric.datasets import HeterophilousGraphDataset
        return HeterophilousGraphDataset(root=osp.join(root, "subgraph_fl"), name=dataset)
    elif dataset in ["Actor"]:
        from torch_geometric.datasets import Actor
        return Actor(root=osp.join(root, "subgraph_fl", "Actor"))
