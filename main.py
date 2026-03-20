import fgl.config as config
from fgl.flcore.trainer import FGLTrainer
from function import *
from config_datasets import attribute

args = config.args

args.root = "your_data_corrupted_root"
args.dataset = ["Actor"]
args.num_clients, args.classes, args.num_rounds, args.lr = attribute(
    args.dataset)

corrupted_client_dir = "your_data_corrupted_root_distrib"
client_data_dir = "your_data_root"


client_data_list = create_corrupted_client(corruption_ratio=args.corruption_ratio, args=args,
                                           client_data_dir=client_data_dir,
                                           corrupted_client_dir=corrupted_client_dir)

trainer = FGLTrainer(args)

trainer.train()
