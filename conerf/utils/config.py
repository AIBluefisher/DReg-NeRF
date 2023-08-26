import argparse


def config_parser():
    parser = argparse.ArgumentParser()

    ################################## Base configs ######################################
    parser.add_argument("--local_rank",
                        type=int,
                        default=0,
                        help='rank for distributed training')
    parser.add_argument('--distributed', action='store_true', help='if use distributed training')
    parser.add_argument('--seed', type=int, default=3407, help='seed for torch and numpy')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--max_iterations", type=int, default=20000)
    parser.add_argument("--num_process", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-4)
    # parser.add_argument("--fine_tune", type=bool, default=False)
    parser.add_argument("--finetune",
                        action="store_true",
                        help="whether to finetune model")
    
    #################################### Dataset #########################################
    parser.add_argument("--dataset",
                        type=str,
                        default="",
                        choices=["mipnerf_360", "nerf_llff_data", "nerf_synthetic", "objaverse",
                                 "scannerf", "Synthetic_NSVF", "Hypersim", "dtu", "BlendedMVS"],
                        help="dataset name")
    parser.add_argument("--json_dir",
                        type=str,
                        default="",
                        help="absolute path for storing register task's json")
    parser.add_argument("--data_split_json",
                        type=str,
                        default="",
                        help="absolute path of data split json file")
    parser.add_argument("--factor",
                        type=int,
                        default=4,
                        choices=[1, 2, 4, 8],
                        help="downsample factor of images")
    parser.add_argument("--train_split",
                        type=str,
                        default="trainval",
                        # choices=["train", "trainval"],
                        help="which train split to use",)
    parser.add_argument("--root_dir",
                        type=str,
                        default="/home/chenyu/Datasets/nerf_synthetic")
    parser.add_argument("--scene",
                        type=str,
                        default="",
                        help="which scene to use",)
    parser.add_argument("--expname",
                        type=str,
                        default="chair_reg",
                        help='experiment name')
    parser.add_argument("--aabb",
                        type=lambda s: [float(item) for item in s.split(",")],
                        # default="-0.5,-0.5,-0.5,0.5,0.5,0.5",
                        # default="-0.7,-0.7,-0.7,0.7,0.7,0.7",
                        default="-1.5,-1.5,-1.5,1.5,1.5,1.5",
                        # default="-100,-100,-100,100,100,100",
                        help="delimited list input")
    parser.add_argument("--test_chunk_size",
                        type=int,
                        default=8192)
    parser.add_argument("--unbounded",
                        action="store_true",
                        help="whether to use unbounded rendering")
    parser.add_argument("--auto_aabb",
                        action="store_true",
                        help="whether to automatically compute the aabb")
    parser.add_argument("--cone_angle", type=float, default=0.0)

    ##################################### multi blocks ########################################
    parser.add_argument("--multi_blocks",
                        action='store_true',
                        help='whether train nerf from different blocks')
    parser.add_argument("--num_blocks",
                        type=int,
                        default=3,
                        help="number of blocks to partition a scene")
    parser.add_argument("--min_num_blocks",
                        type=int,
                        default=2,
                        help="minimum blocks to partition a scene")
    parser.add_argument("--max_num_blocks",
                        type=int,
                        default=4,
                        help="maximum blocks to partition a scene")
    ##################################### registration ########################################
    parser.add_argument("--position_embedding_type",
                        type=str,
                        default="sine",
                        help="which kind of positional embedding to use in transformer")
    parser.add_argument("--position_embedding_dim",
                        type=int,
                        default=256,
                        help="dimensionality of position embeddings")
    parser.add_argument("--position_embedding_scaling",
                        type=float,
                        default=1.0,
                        help="position embedding scale factor")
    parser.add_argument("--num_downsample",
                        type=int,
                        default=6,
                        help="how many layers used to downsample points")
    parser.add_argument("--robust_loss",
                        action="store_true",
                        help="whether to use robust loss function")

    ##################################### checkpoints #########################################
    parser.add_argument("--ckpt_path", type=str, default="",
                        help='specific weights npy file to reload for coarse network')
    parser.add_argument("--no_load_opt",
                        action='store_true',
                        help='do not load optimizer when reloading')
    parser.add_argument("--no_load_scheduler",
                        action='store_true',
                        help='do not load scheduler when reloading')

    ##################################### log/saving options ##################################
    parser.add_argument("--enable_tensorboard",
                        action='store_true',
                        help='if use tensorboard')
    parser.add_argument("--enable_visdom",
                        action='store_true',
                        help='if use visdom to visualize camera poses')
    parser.add_argument("--n_tensorboard",
                        type=int,
                        default=30,
                        help='frequency of terminal printout')
    parser.add_argument("--n_validation",
                        type=int,
                        default=2500,
                        help='frequency of tensorboard image logging')
    parser.add_argument("--n_checkpoint",
                        type=int,
                        default=5000,
                        help='frequency of weight ckpt saving')

    args = parser.parse_args()

    return args
