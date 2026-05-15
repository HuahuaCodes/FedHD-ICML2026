from __future__ import print_function
import argparse
import os
from datetime import datetime
import numpy as np
import logging
from model.FedHD.SynGenerator import FedWSIDDG
from model.FedHD.LocalTrainer import FedWSIDDM
import torch

parser = argparse.ArgumentParser(description='FedHD: Federated Distillation for WSI')

# Experiment
parser.add_argument('--repeat', type=int, default=5)
parser.add_argument('--seed', type=int, default=1)
parser.add_argument('--debug', action='store_true', default=False)
parser.add_argument('--results_dir', default='./exp')
parser.add_argument('--exp_code', type=str, required=True)
parser.add_argument('--task', type=str, required=True, help='CAMELYON16 | CAMELYON17 | IDH')
parser.add_argument('--module', type=str, default='base', choices=['base', 'syn_data', 'local_train'])

# Data
parser.add_argument('--data_root_dir', type=str, default=None)
parser.add_argument('--n_classes', type=int, default=2)
parser.add_argument('--feature_type', type=str, default='R50_features')
parser.add_argument('--use_h5', action='store_true', default=False)

# Model
parser.add_argument('--ft_model', type=str, default='ResNet50')
parser.add_argument('--mil_method', type=str, default='CLAM_SB')
parser.add_argument('--heter_model', action='store_true', default=False)
parser.add_argument('--model_size', type=str, default='small', choices=['tiny', 'ultra_small', 'small', 'big'])
parser.add_argument('--drop_out', action='store_true', default=False)
parser.add_argument('--subtyping', action='store_true', default=False)

# Training
parser.add_argument('--local_epochs', type=int, default=50)
parser.add_argument('--lr', type=float, default=3e-3)
parser.add_argument('--opt', type=str, default='adamw')
parser.add_argument('--reg', type=float, default=1e-5)
parser.add_argument('--bag_loss', type=str, default='ce', choices=['svm', 'ce', 'mag'])
parser.add_argument('--weighted_sample', action='store_true', default=False)
parser.add_argument('--early_stopping', action='store_true', default=False)
parser.add_argument('--accumulate_grad_batches', type=int, default=1)

# Distillation
parser.add_argument('--ipc', type=int, default=10, help='synthetic slides per class (without O2O)')
parser.add_argument('--nps', type=int, default=1000, help='synthetic patches per slide (T)')
parser.add_argument('--dc_iterations', type=int, default=1000)
parser.add_argument('--image_lr', type=float, default=0.1)
parser.add_argument('--image_opt', type=str, default='sgd')
parser.add_argument('--syn_size', type=int, default=1024, help='feature dimension d')
parser.add_argument('--slide_batch', type=int, default=1)
parser.add_argument('--test_iter', type=int, default=100)
parser.add_argument('--use_latent_prior', action='store_true', default=False, help='FDD: feature-space distillation')
parser.add_argument('--instance_learn', action='store_true', default=False, help='O2O: one synthetic per real slide')
parser.add_argument('--cluster', action='store_true', default=False, help='GMA: GMM-based alignment')
parser.add_argument('--faiss_cluster', action='store_true', default=False)
parser.add_argument('--load_syn_data', action='store_true', default=False)
parser.add_argument('--syn_data_dir', type=str, default='syn_data')

# Misc / legacy args (kept for compatibility with utils)
parser.add_argument('--global_epochs', type=int, default=10)
parser.add_argument('--global_epochs_dm', type=int, default=50)
parser.add_argument('--kd_iters', type=int, default=100)
parser.add_argument('--pretrain_kd', action='store_true', default=False)
parser.add_argument('--image_batch_size', type=int, default=128)
parser.add_argument('--image_size', type=int, default=224)
parser.add_argument('--ensemble_epochs', type=int, default=50)
parser.add_argument('--no_inst_cluster', action='store_true', default=False)
parser.add_argument('--inst_loss', type=str, default=None, choices=['svm', 'ce', None])
parser.add_argument('--bag_weight', type=float, default=0.7)
parser.add_argument('--B', type=int, default=8)
parser.add_argument('--top_k', type=int, default=-1)
parser.add_argument('--n_heads', type=int, default=1)
parser.add_argument('--numLayer_Res', type=int, default=0)
parser.add_argument('--numGroup', type=int, default=4)
parser.add_argument('--total_instance', type=int, default=4)
parser.add_argument('--grad_clipping', type=float, default=5.0)
parser.add_argument('--num_MeanInference', type=int, default=1)
parser.add_argument('--distill_type', type=str, default='AFS')
parser.add_argument('--use_generate_prior', action='store_true', default=False)
parser.add_argument('--use_mixup', action='store_true', default=False)
parser.add_argument('--dd_mix', action='store_true', default=False)
parser.add_argument('--use_adaptive_sampling', action='store_true', default=False)
parser.add_argument('--use_adaptive_augmentation', action='store_true', default=False)
parser.add_argument('--sn', action='store_true', default=False, help='stain normalization for synthetic images')
parser.add_argument('--init_real', action='store_true', default=False, help='init syn image with real images')
parser.add_argument('--aug_type', type=str, default='color_crop_cutout', help='DiffAug strategy')
parser.add_argument('--mag', type=float, default=0.5, help='margin for FeatMag bag loss')
parser.add_argument('--scale_factor', type=float, default=1.0)
parser.add_argument('--alpha', type=float, default=0.4)
parser.add_argument('--pretrained_dir', type=str, default='')
parser.add_argument('--ld_proto', type=float, default=0.1)
parser.add_argument('--fed_method', type=str, default='fed_wsidd')
parser.add_argument('--fed_split', type=str, default='FeatureSynthesisLabel')

args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

os.makedirs(f"logs/{args.task}", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    filemode="w",
    format="%(name)s: %(asctime)s | %(filename)s:%(lineno)s | %(message)s",
    filename=f"logs/{args.task}/{args.mil_method}_{args.feature_type}_{args.exp_code}_logs.txt"
)
logger = logging.getLogger(__name__)

args.results_dir = os.path.join(args.results_dir, f"{args.task}/{args.mil_method}_{args.feature_type}_{args.exp_code}")
os.makedirs(args.results_dir, exist_ok=True)
logger.info(f'Results: {args.results_dir}')

def seed_torch(seed=7):
    import random
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    if args.debug:
        args.repeat = 1
        args.dc_iterations = 1
        args.local_epochs = 1
        args.global_epochs = 1
        args.test_iter = 1

    for rep in range(args.repeat):
        args.rep = rep
        logger.info(f'===== Run {rep} starts =====')
        seed_torch(int(datetime.now().timestamp()))

        if args.module == 'syn_data':
            runner = FedWSIDDG(args, logger=logger)
        elif args.module == 'local_train':
            runner = FedWSIDDM(args, logger=logger)
        else:
            raise ValueError("Unknown module: " + args.module + ". Use syn_data or local_train.")

        runner.run(rep)
