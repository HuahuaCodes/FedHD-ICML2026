import sys
sys.path.append('..')
from data.cam16.fed_cam_dataset import FedCamelyon16
# from data.cam17.fed_cam_dataset import FedCamelyon17
from data.cam17.fed_cam_pat_dataset import FedCamelyon17Pat
from data.cam16.fed_cam_image_dataset import FedCamelyon16Image
from data.tcga_idh.fed_tcga_dataset import FedTCGAIDH

import torch
from torch.utils.data import Dataset
import random

class MixupDataset(Dataset):
    def __init__(self, real_dataset, syn_dataset, alpha=0.2):
        self.real_dataset = real_dataset
        self.syn_dataset = syn_dataset
        self.alpha = alpha
        self.length = max(len(real_dataset), len(syn_dataset))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        # sample a real and a synthetic image
        real_img, real_label = self.real_dataset[idx % len(self.real_dataset)]
        syn_img, syn_label = self.syn_dataset[random.randrange(len(self.syn_dataset))]

        # sample lambda from Beta distribution
        lam = torch.distributions.Beta(self.alpha, self.alpha).sample().item()

        # perform mixup
        mixed_img = lam * real_img + (1 - lam) * syn_img
        mixed_label = lam * real_label + (1 - lam) * syn_label   # if labels are one-hot

        return mixed_img, mixed_label, real_label, lam

def define_data(args, logger, feature_type_list=None, **kwargs):
    def _ft(i): return feature_type_list[i] if feature_type_list else args.feature_type
    if args.task == 'CAMELYON16_IMAGE':
        train_dataset_c0 = FedCamelyon16Image(center=0, train=True, data_path=args.data_root_dir, logger=logger, **kwargs)
        train_dataset_c1 = FedCamelyon16Image(center=1, train=True, data_path=args.data_root_dir, logger=logger, **kwargs)
        test_dataset_c0 = FedCamelyon16Image(center=0, train=False, data_path=args.data_root_dir, logger=logger, **kwargs)
        test_dataset_c1 = FedCamelyon16Image(center=1, train=False, data_path=args.data_root_dir, logger=logger, **kwargs)
        train_dataset = [train_dataset_c0, train_dataset_c1] # each dataset is a list of datasets
        test_dataset = [test_dataset_c0, test_dataset_c1]
        agent_group = [0, 1]
        for i in range(len(train_dataset)):
            # count number of test sample per class
            test_idx_label = test_dataset[i].get_idx_per_class()
            print(f'Center {i} has {len(train_dataset[i])} training samples and {len(test_dataset[i])} testing samples')
            for class_id, count in test_idx_label.items():
                print(f'  Class {class_id}: {count} samples')
    elif args.task == 'CAMELYON16':
        train_dataset_c0 = FedCamelyon16(center=0, train=True, data_path=args.data_root_dir, logger=logger, feature_type=_ft(0), **kwargs)
        train_dataset_c1 = FedCamelyon16(center=1, train=True, data_path=args.data_root_dir, logger=logger, feature_type=_ft(1), **kwargs)
        test_dataset_c0 = FedCamelyon16(center=0, train=False, data_path=args.data_root_dir, logger=logger, feature_type=_ft(0), **kwargs)
        test_dataset_c1 = FedCamelyon16(center=1, train=False, data_path=args.data_root_dir, logger=logger, feature_type=_ft(1), **kwargs)
        train_dataset = [train_dataset_c0, train_dataset_c1]
        test_dataset = [test_dataset_c0, test_dataset_c1]
        agent_group = [0, 1]
        for i in range(len(train_dataset)):
            # count number of test sample per class
            print(f'Center {i} has {len(train_dataset[i])} training samples and {len(test_dataset[i])} testing samples')
            cls_idx_cont = {}
            for x, y in test_dataset[i]:
                if y.item() not in cls_idx_cont:
                    cls_idx_cont[y.item()] = 0
                cls_idx_cont[y.item()] += 1
            print(cls_idx_cont)
            # all = []
            # train_idx_label = train_dataset[i].get_idx_per_class()
            # test_idx_label = test_dataset[i].get_idx_per_class()
            # for key in train_idx_label:
            #     all.append(train_idx_label[key] + test_idx_label[key])
            # print(all)
    elif args.task == 'CAMELYON17':
        centers = ['center_0', 'center_1', 'center_2', 'center_3', 'center_4']
        train_dataset = [FedCamelyon17Pat(center=center, train=True, data_path=args.data_root_dir, logger=logger, feature_type=_ft(i), **kwargs) for i, center in enumerate(centers)]
        test_dataset = [FedCamelyon17Pat(center=center, train=False, data_path=args.data_root_dir, logger=logger, feature_type=_ft(i), **kwargs) for i, center in enumerate(centers)]
        agent_group = list(range(len(centers)))
        for i in range(len(centers)):
            print(f'Center {i}: {centers[i]} has {len(train_dataset[i])} training samples and {len(test_dataset[i])} testing samples')
    elif 'IDH' in args.task:
        centers = ['Henry Ford Hospital',
                   'Thomas Jefferson University',
                   'Mayo Clinic - Rochester',
                   'Duke',
                   'Case Western',
                   'Case Western - St Joes',
                   'Dept of Neurosurgery at University of Heidelberg',
                   'MD Anderson Cancer Center',] #'Case Western - St Joes',  'Emory University', 'Cedars Sinai'
        train_dataset = [FedTCGAIDH(center=center, train=True, data_path=args.data_root_dir, logger=logger, feature_type=_ft(i), **kwargs) for i, center in enumerate(centers)]
        test_dataset = [FedTCGAIDH(center=center, train=False, data_path=args.data_root_dir, logger=logger, feature_type=_ft(i), **kwargs) for i, center in enumerate(centers)]
        agent_group = list(range(len(centers)))
        for i in range(len(centers)):
            print(f'Center {i}: {centers[i]} has {len(train_dataset[i])} training samples and {len(test_dataset[i])} testing samples')
            # train_num = train_dataset[i].get_numer_instances_per_class()
            # test_num = test_dataset[i].get_numer_instances_per_class()
            # all = [train_num[0]+test_num[0], train_num[1]+test_num[1]]
            # print(all)

    else:
        raise NotImplementedError
    return train_dataset, test_dataset, agent_group

if __name__ == '__main__':
    centers = ['center_0', 'center_1', 'center_2', 'center_3', 'center_4']
    train_dataset = [FedCamelyon17Pat(center=center, train=True, data_path='/g/data/iq24/CAMELYON17_patches/centers/', logger=None, feature_type='R50_features', **kwargs) for center in centers]
    test_dataset = [FedCamelyon17Pat(center=center, train=False, data_path='/g/data/iq24/CAMELYON17_patches/centers/', logger=None, feature_type='R50_features', **kwargs) for center in centers]