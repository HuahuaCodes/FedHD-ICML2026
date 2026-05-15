# train feature augmentation, ensure samesize
from sklearn.preprocessing import label_binarize
from copy import deepcopy
from utils.core_util import clam_runner, transmil_runner, hipt_runner, frmil_runner, abmil_runner
from tqdm import tqdm
from utils.Get_model import define_model
from utils.Get_data import define_data
from utils.trainer_util import get_optim, get_loss
import torch
import os
from torch import nn
import copy
import numpy as np
import h5py
from torch.utils.data import DataLoader
from utils.data_utils import get_split_loader, CategoriesSampler
from utils.core_util import clam_runner, raw_feature_extract
from torch.utils.data import DataLoader, TensorDataset
from model.FedHD.distillation import distribution_matching, distribution_matching_woMIL, distribution_matching_woMIL_wG, distribution_matching_woMIL_latent, distribution_matching_latent_instance, faiss_kmeans_cluster
from sklearn.metrics import roc_curve, f1_score, roc_auc_score
import random
import torch.nn.functional as F
import warnings

class GCELoss(nn.Module):
    """
    Generalized Cross Entropy (GCE) Loss.
    Reference:
        Zhang & Sabuncu, "Generalized Cross Entropy Loss for Training Deep Neural Networks with Noisy Labels", NeurIPS 2018.
    
    Args:
        q (float): Robustness parameter in (0, 1]. Smaller q → more like CE; q=1 → like MAE.
        reduction (str): 'mean', 'sum', or 'none' (same as PyTorch reduction semantics).
    """

    def __init__(self, q=0.7, reduction='mean'):
        super(GCELoss, self).__init__()
        assert q > 0 and q <= 1, "q must be in (0, 1]"
        assert reduction in ['mean', 'sum', 'none'], "Invalid reduction mode"
        self.q = q
        self.reduction = reduction

    def forward(self, logits, targets):
        """
        Args:
            logits (Tensor): Model outputs before softmax, shape (N, C)
            targets (Tensor): Ground truth class indices, shape (N,)
        Returns:
            loss (Tensor): scalar if reduction='mean' or 'sum', else shape (N,)
        """
        # Softmax probabilities
        probs = F.softmax(logits, dim=1)
        # Pick the probability of the true class for each sample
        probs_true = probs.gather(1, targets.unsqueeze(1)).squeeze(1).clamp(min=1e-8, max=1.0)
        
        # GCE formula: (1 - p_y^q) / q
        loss = (1.0 - probs_true.pow(self.q)) / self.q

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss


def get_aggregation_weights(client_acc, client_loss):
    # client_acc and client_loss are dictionary, tranform into list
    client_acc_list = list(client_acc.values())
    client_loss_list = list(client_loss.values())
    client_acc_list = np.array(client_acc_list)/np.sum(client_acc_list)
    client_loss_list = np.array(client_loss_list)/np.sum(client_loss_list)
    print('Normalized client acc: ', client_acc_list)
    print('Normalized client loss: ', client_loss_list)
    # client-wise aggregation weights=alpha*acc+beta*loss
    alpha = 0.5
    beta = 0.5
    weights = alpha*client_acc_list+beta*client_loss_list
    print('Client-wise aggregation weights: ', weights)
    # adaptive aggregation
    return weights

def check_sublists_equal_size(lst):
    if not lst:  # Handle the case of an empty list
        return True

    # Get the size of the first sublist
    first_size = len(lst[0])

    # Check if all other sublists have the same size
    return all(len(sublist) == first_size for sublist in lst)

def load_img_from_pth(args, img_pth, class_proto):
    if 'CAMELYON16' in args.task:
        stage = img_pth.split('/')[5:7]
        img_h5_pth = img_pth.replace('pt', 'h5')
        slide_id_h5 = img_h5_pth.split('/')[-1]
        slide_id_pt = img_pth.split('/')[-1]
        local_pth = '/scratch/iq24/cc0395/FedHD/data/cam16'
        local_h5_pth = f'{local_pth}/{stage[0]}/{stage[1]}/R50_features/h5_files/{slide_id_h5}'
        local_pt_pth = f'{local_pth}/{stage[0]}/{stage[1]}/R50_features/pt_files/{slide_id_pt}'
        with h5py.File(local_h5_pth, "r") as file:
            X_prob = np.array(file['sampling_prob'][:]) 
            # Convert to torch tensor and normalize if needed
            X_prob = torch.from_numpy(X_prob).float()
            if not torch.isclose(X_prob.sum(), torch.tensor(1.0), atol=1e-2):
                X_prob = X_prob / X_prob.sum()
        X = torch.load(local_pt_pth)   
        N = X.shape[0]
        if args.use_adaptive_sampling:
            # Ensure X_prob matches the size of X
            if len(X_prob) != N:
                print(f'Warning: X_prob size ({len(X_prob)}) != X size ({N}), truncating/padding X_prob')
                raise ValueError('X_prob size does not match X size')
            num_sampled = int(N * args.scale_factor)
            # Ensure num_sampled doesn't exceed available patches
            num_sampled = min(num_sampled, N)
            sampled_indices = torch.multinomial(X_prob, num_sampled, replacement=True)
            print('Sampled %d patches from original %d patches' % (num_sampled, N))
            X = X[sampled_indices]
        if args.use_adaptive_augmentation:
            # Sample mixup lambdas
            # lambdas = torch.distributions.Beta(args.alpha, args.alpha).sample((N,))
            # lambdas = lambdas.unsqueeze(1)  # shape (N, 1)
            # # Scale lambdas by density (common patches get higher λ)
            # scaled_lambdas = X_prob.unsqueeze(1) * lambdas  # (N, 1)
            # # Perform prototype-based mixup
            # X = (1 - scaled_lambdas) * X + scaled_lambdas * class_proto[c]
            X = X + class_proto[c]
    else:
        X = torch.load(img_pth).reshape((-1, 1024))
        X_prob = torch.ones(X.shape[0])
    return X, X_prob

def get_images(args, images_all, indices_class, c, n, class_proto): # get random n images from class c
    print(f'\n Sample {n} images from class {c} with {len(indices_class[c])} images')
    X_all = []
    X_prob_all = []
    if n <= len(indices_class[c]):
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
    else:
        idx_shuffle_0 = np.random.permutation(indices_class[c])
        idx_shuffle_1 = np.random.permutation(indices_class[c])[:n-len(indices_class[c])]
        idx_shuffle = np.concatenate([idx_shuffle_0, idx_shuffle_1], axis=0)
    for i in idx_shuffle:
        img_pth = images_all[i]
        X, X_prob = load_img_from_pth(args, img_pth, class_proto)
        X_all.append(X)
        X_prob_all.append(X_prob)
    X_all = torch.cat(X_all, dim=0)
    X_prob_all = torch.cat(X_prob_all, dim=0)
    return X_all, X_prob_all

def calculate_kd_loss(y_pred_student, y_pred_teacher, y_true, loss_fn, temp=20., distil_weight=0.9):
    """
    Function used for calculating the KD loss during distillation

    :param y_pred_student (torch.FloatTensor): Prediction made by the student model
    :param y_pred_teacher (torch.FloatTensor): Prediction made by the teacher model
    :param y_true (torch.FloatTensor): Original label
    """

    soft_teacher_out = F.softmax(y_pred_teacher / temp, dim=1)
    soft_student_out = F.log_softmax(y_pred_student / temp, dim=1)

    loss = (1. - distil_weight) * F.cross_entropy(y_pred_student, y_true)
    loss += (distil_weight * temp * temp) * loss_fn(
        soft_student_out, soft_teacher_out
    )
    return loss


class FedWSIDDG:
    def __init__(self, args, logger):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.args = args
        self.logger = logger
        self.train_dataset, self.test_dataset, self.n_clients = define_data(args, logger,
                                                                            image_size=args.syn_size)
        self.init_loss_fn()
        print('Number of clients:', len(self.n_clients))
        self.get_data_weight()
        for i in range(len(self.n_clients)):
            print(f'    Train: {len(self.train_dataset[i])}; Test: {len(self.test_dataset[i])}')

    def get_train_loader(self, ds):
        if 'frmil' in self.args.mil_method:
            train_sampler = CategoriesSampler(ds.labels,
                                              n_batch=len(ds.slide_data),
                                              n_cls=self.args.n_classes,
                                              n_per=1)
            train_loader = DataLoader(dataset=ds, batch_sampler=train_sampler, num_workers=4, pin_memory=False)
        else:
            train_loader = get_split_loader(ds, training=True, weighted=self.args.weighted_sample)
        return train_loader

    def get_test_loader(self, ds):
        if 'frmil' in self.args.mil_method:
            test_loader = DataLoader(dataset=ds, batch_size=1, shuffle=False, num_workers=4, pin_memory=False)
        else:
            test_loader = get_split_loader(ds)
        return test_loader

    def init_loss_fn(self):
        self.crossentropy_loss = nn.NLLLoss(reduce=False)
        self.dist_loss = nn.MSELoss()
        self.cos_loss = torch.nn.CosineSimilarity(dim=-1)
        self.CE_loss = nn.CrossEntropyLoss()
        self.mil_loss = get_loss(self.args)
        self.ensemble_loss = nn.KLDivLoss(reduction="batchmean")

    def init_dataset(self, local_train_ds, local_test_ds):
        train_dataset = local_train_ds
        test_dataset = local_test_ds
        return self.get_train_loader(train_dataset), self.get_test_loader(test_dataset)

    def get_data_weight(self):
        n_clnt = len(self.train_dataset)
        weight_list = np.asarray([len(self.train_dataset[i]) for i in range(n_clnt)])
        self.weight_list = weight_list / np.sum(weight_list)

    def mil_run(self, model,
                data,
                label,
                loss_fn,
                return_lgt=False,
                return_feature=False,
                raw_image=False):
        if 'CLAM' in model.__class__.__name__:
            if return_feature and return_lgt:
                loss, error, pred_prob, feature, lgt = clam_runner(self.args, model, data, label, loss_fn, return_feature=True,
                                                        return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, feature, lgt
            elif return_lgt and not return_feature:
                loss, error, pred_prob, lgt = clam_runner(self.args, model, data, label, loss_fn, return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, lgt
            elif return_feature and not return_lgt:
                loss, error, pred_prob, feature = clam_runner(self.args, model, data, label, loss_fn, return_feature=True, raw_image=raw_image)
                return loss, error, pred_prob, feature
            else:
                loss, error, pred_prob = clam_runner(self.args, model, data, label, loss_fn, raw_image=raw_image)
                return loss, error, pred_prob
        elif 'TransMIL' in model.__class__.__name__:
            if return_feature and return_lgt:
                loss, error, pred_prob, feature, lgt = transmil_runner(self.args, model, data, label, loss_fn, return_feature=True,
                                                        return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, feature, lgt
            elif return_lgt and not return_feature:
                loss, error, pred_prob, lgt = transmil_runner(self.args, model, data, label, loss_fn, return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, lgt
            elif return_feature and not return_lgt:
                loss, error, pred_prob, feature = transmil_runner(self.args, model, data, label, loss_fn, return_feature=True, raw_image=raw_image)
                return loss, error, pred_prob, feature
            else:
                loss, error, pred_prob = transmil_runner(self.args, model, data, label, loss_fn, raw_image=raw_image)
                return loss, error, pred_prob
        elif 'Attention' in model.__class__.__name__:
            if return_feature and return_lgt:
                loss, error, pred_prob, feature, lgt = abmil_runner(self.args, model, data, label, loss_fn, return_feature=True,
                                                        return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, feature, lgt
            elif return_lgt and not return_feature:
                loss, error, pred_prob, lgt = abmil_runner(self.args, model, data, label, loss_fn, return_lgt=True, raw_image=raw_image)
                return loss, error, pred_prob, lgt
            elif return_feature and not return_lgt:
                loss, error, pred_prob, feature = abmil_runner(self.args, model, data, label, loss_fn, return_feature=True, raw_image=raw_image)
                return loss, error, pred_prob, feature
            else:
                loss, error, pred_prob = abmil_runner(self.args, model, data, label, loss_fn, raw_image=raw_image)
                return loss, error, pred_prob

    def local_train(self, agent_idx, local_model, local_optim, train_loader, test_loader=None):
        local_model.train()
        epoch_loss = 0.
        best_test_error = 1
        for ep in range(self.args.local_epochs):
            batch_loss = 0.
            batch_error = 0.
            for batch_idx, (images, labels) in enumerate(train_loader):
                if len(images.shape) == 3 and images.shape[0] == 1:
                    images = images.squeeze(0)
                images, labels = images.to(self.device), labels.to(self.device)
                local_model.zero_grad()
                loss, error, y_prob = self.mil_run(local_model, images, labels, self.mil_loss)
                loss.backward()
                local_optim.step()
                batch_loss += loss.item()

            current_epoch_loss = batch_loss / len(train_loader)
            batch_error /= len(train_loader)
            epoch_loss += current_epoch_loss
            if ep % 1 == 0:
                if test_loader is not None:
                    total_loss, total_error, fpr, tpr, f1, auc, all_probs = self.local_test(local_model, test_loader)
                    self.logger.info(f'[MIL Training]Agent: {agent_idx}, Iter: {ep}, Loss: {current_epoch_loss}, Test Acc: {1-total_error}, Test F1: {f1}, Test AUC: {auc}')
                    if total_error < best_test_error:
                        best_test_error = total_error
                        torch.save(local_model.state_dict(), f'{self.args.results_dir}/client_{agent_idx}_{local_model.__class__.__name__}_pretrain.pt')
                else:
                    self.logger.info(f'[MIL Training]Agent: {agent_idx}, Iter: {ep}, Loss: {current_epoch_loss}')
        if test_loader is not None:
            return epoch_loss / self.args.local_epochs, total_error
        return epoch_loss / self.args.local_epochs

    def local_test(self, model, test_loader):
        model.eval()
        total_loss = 0.
        total_error = 0.
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch_idx, (images, labels) in tqdm(enumerate(test_loader)):
                images, labels = images.to(self.device), labels.to(self.device)
                # print('Image size:', images.size(), 'Label size:', labels.size(), 'Required grad:', images.requires_grad)
                loss, error, Y_prob = self.mil_run(model, images, labels, self.mil_loss)
                total_loss += loss.item()
                total_error += error
                probs = Y_prob.detach().cpu().numpy()
                all_probs.append(probs)
                all_labels.append(labels.item())
            total_loss /= len(test_loader)
            total_error /= len(test_loader)
            all_probs = np.concatenate(all_probs, axis=0)
            all_labels = np.array(all_labels)
            if self.args.n_classes == 2:
                fpr, tpr, thresholds = roc_curve(all_labels, all_probs[:, 1])
            else:
                fpr = dict()
                tpr = dict()
                y_true_bin = label_binarize(all_labels, classes=list(range(self.args.n_classes)))
                for i in range(y_true_bin.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], np.array(all_probs)[:, i])
        # I need F1 and AUC 
        # Suppress warnings for missing classes in federated learning scenarios
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=UserWarning)
            if self.args.n_classes == 2:
                # For binary classification, use only the positive class probabilities
                auc = roc_auc_score(all_labels, all_probs[:, 1])
                f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), zero_division=0)
            else:
                # For multi-class, use multi_class='ovo' and specify all possible labels
                auc = roc_auc_score(all_labels, all_probs, multi_class='ovo', labels=list(range(self.args.n_classes)))
                f1 = f1_score(all_labels, np.argmax(all_probs, axis=1), average='macro', labels=list(range(self.args.n_classes)), zero_division=0)
        return total_loss, total_error, fpr, tpr, f1, auc, all_probs

    def pretrain_clients(self, repeat, save_best=False):
        clients_models_pre = []
        optimizers_pre = []
        MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']# more will be added
        for idx in range(len(self.n_clients)):
            if self.args.heter_model:
                seed = 33 + repeat + len(MIL_pool)
                random.seed(seed)
                self.args.mil_method = random.choice(MIL_pool)
                if len(MIL_pool) > 0:
                    MIL_pool.remove(self.args.mil_method)
                if len(MIL_pool) == 0:
                    MIL_pool = ['CLAM_SB', 'TransMIL', 'ABMIL_att']
                print(f'=====> Agent {idx} uses {self.args.mil_method} {self.args.opt}')
                clients_models_pre.append(define_model(self.args))
                optimizers_pre.append(get_optim(self.args, clients_models_pre[idx]))
            else:
                clients_models_pre.append(define_model(self.args))
                optimizers_pre.append(get_optim(self.args, clients_models_pre[idx]))
        # clients_models_pre = [define_model(self.args) for _ in range(len(self.n_clients))]
        # optimizers_pre = [get_optim(self.args, model) for model in clients_models_pre]
        self.logger.info('=========================Pretraining Clients=========================')
        for client_idx in range(len(self.n_clients)):
            train_loader, test_loader = self.init_dataset(self.train_dataset[client_idx], self.test_dataset[client_idx])
            pretrained_model_path = f'{self.args.results_dir}/client_{client_idx}_{clients_models_pre[client_idx].__class__.__name__}_pretrain.pt'
            # if os.path.exists(pretrained_model_path):
            #     clients_models_pre[client_idx].load_state_dict(torch.load(pretrained_model_path))
            #     self.logger.info(f'Client {client_idx} Pretrained model loaded from {pretrained_model_path}')
            # else:
            clients_models_pre[client_idx].to(self.device)
            best_test_error = 1
            if save_best:
                train_loss = self.local_train(client_idx, clients_models_pre[client_idx],
                                            optimizers_pre[client_idx], train_loader, test_loader)
                clients_models_pre[client_idx].load_state_dict(torch.load(pretrained_model_path))
            else:
                train_loss = self.local_train(client_idx, clients_models_pre[client_idx],
                                            optimizers_pre[client_idx], train_loader)
            test_loss, test_error,_,_,_,_,_ = self.local_test(clients_models_pre[client_idx], test_loader)
            ''' Save checkpoint '''
            self.logger.info(f'[Pretrain Done]Client {client_idx} Test Loss: {test_loss}, Test Acc: {1-test_error}')
            torch.save(clients_models_pre[client_idx].state_dict(), pretrained_model_path)
        return clients_models_pre

    def run(self, repeat):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ''' Pretrain/Load local models '''
        clients_models_pre = self.pretrain_clients(repeat, save_best=True)
        '''Train/Load virtual data'''
        self.logger.info('=========================Training synthetic data=========================')
        client_dd_acc = {}
        client_dd_loss = {}
        client_dd_probs = {}
        client_dd_best_iter = {}
        self.train_dataset_label_idices = [None for _ in range(len(self.n_clients))]
        client_label_list = [[] for _ in range(len(self.n_clients))]
        label_syns = [None for _ in range(len(self.n_clients))]
        image_syns = [None for _ in range(len(self.n_clients))]
        for client_idx in range(len(self.n_clients)):
            # organize the real dataset
            indices_class = [[] for c in range(self.args.n_classes)]
            images_all, labels_all = [], []
            for i in range(len(self.train_dataset[client_idx])):
                images_all.append(self.train_dataset[client_idx].__getitem__(i, path=True)[2])
                labels_all.append(self.train_dataset[client_idx].__getitem__(i)[1])

            for idx, lab in enumerate(labels_all):
                lab_item = int(lab.item()) if isinstance(lab, torch.Tensor) else int(lab)
                indices_class[lab_item].append(idx)
            self.train_dataset_label_idices[client_idx] = indices_class
            for i in range(len(indices_class)):
                if len(indices_class[i]) == 0:
                    print(f'[WARNINNG] Client {client_idx} has no label {i}')
                    indices_class.pop(i)
                else:
                    client_label_list[client_idx].append(i)

            # organize syn data
            for i in range(len(labels_all)):
                if isinstance(labels_all[i], torch.Tensor):
                    labels_all[i] = labels_all[i].numpy()
            label_syns[client_idx] = torch.tensor(np.array(labels_all), dtype=torch.long,
                                      requires_grad=False, device=self.device).view(-1)
            if self.args.use_latent_prior:
                image_syns[client_idx] = torch.randn(size=(len(images_all), self.args.nps, self.args.syn_size),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)
            else:
                image_syns[client_idx] = torch.randn(size=(len(images_all), self.args.nps, 3, self.args.syn_size, self.args.syn_size),
                                  dtype=torch.float,
                                  requires_grad=True,
                                  device=self.device)
            print('Number of element that require grad in image_syns:', image_syns[client_idx].size(), label_syns[client_idx].size(), image_syns[0].numel())

            if self.args.image_opt == 'adam':
                optimizer_img = torch.optim.Adam([image_syns[client_idx], ], lr=self.args.image_lr,)
            elif self.args.image_opt == 'sgd':
                optimizer_img = torch.optim.SGD([image_syns[client_idx], ], lr=self.args.image_lr,
                                                momentum=0.5)
            inv_iters = self.args.dc_iterations
            image_batch = self.args.slide_batch
            dd_loss_avg = 0
            if client_idx not in client_dd_acc:
                client_dd_acc[client_idx] = 0
            if client_idx not in client_dd_loss:
                client_dd_loss[client_idx] = 0
            if client_idx not in client_dd_probs:
                client_dd_probs[client_idx] = 0
            if client_idx not in client_dd_best_iter:
                client_dd_best_iter[client_idx] = inv_iters - 1
            pbar_dd = tqdm(range(inv_iters), desc=f'Client {client_idx} - DD loss {dd_loss_avg}')
            # pre-compute class proto
            class_proto = {}
            for c in range(len(indices_class)):
                class_proto[c] = [images_all[idx] for idx in indices_class[c]]
                # Load and compute mean in one go to save memory
                patch_features = [torch.load(patch_feature).reshape(-1, self.args.syn_size) for patch_feature in class_proto[c]]
                class_proto[c] = torch.cat(patch_features, dim=0).mean(dim=0)
                del patch_features  # Free memory immediately
            
            # Cache loaded images to avoid repeated disk I/O
            if self.args.instance_learn:
                self.logger.info(f'Caching {len(images_all)} images for client {client_idx}...')
                cached_images = []
                cached_probs = []
                for i in range(len(images_all)):
                    img_pth = images_all[i]
                    image_real, image_real_prob = load_img_from_pth(self.args, img_pth, class_proto)
                    cached_images.append(image_real.to(self.device))
                    cached_probs.append(image_real_prob.to(self.device))
                self.logger.info(f'Caching complete for client {client_idx}')
                
                # Pre-compute clustering once if using cluster mode
                cluster_ids_cache = None
                if self.args.cluster and self.args.faiss_cluster:
                    self.logger.info('Pre-computing FAISS clusters for all cached images...')
                    all_cached = torch.cat(cached_images, dim=0)
                    cluster_ids_cache, _ = faiss_kmeans_cluster(all_cached, n_clusters=16)
                    del all_cached
                    torch.cuda.empty_cache()
            
            for it in pbar_dd:
                # Clear cache at the start of each iteration
                torch.cuda.empty_cache()
                # get real images for each class (n_slide, n_patches, 1024)
                if self.args.instance_learn:
                    # Batch processing with gradient accumulation
                    batch_size = min(8, len(images_all))  # Process 8 instances at a time
                    num_batches = (len(images_all) + batch_size - 1) // batch_size
                    
                    for batch_idx in range(num_batches):
                        start_idx = batch_idx * batch_size
                        end_idx = min(start_idx + batch_size, len(images_all))
                        batch_loss = 0.0
                        
                        for i in range(start_idx, end_idx):
                            image_real = cached_images[i]
                            # Use pre-computed clusters if available
                            if cluster_ids_cache is not None:
                                # Extract cluster IDs for this image's patches
                                offset = sum(cached_images[j].size(0) for j in range(i))
                                img_cluster_ids = cluster_ids_cache[offset:offset+image_real.size(0)]
                                loss, updated_syn = distribution_matching_latent_instance(
                                    image_real, image_syns[client_idx][i], class_proto, 
                                    optimizer_img, self.args, precomputed_cluster_ids=img_cluster_ids
                                )
                            else:
                                loss, updated_syn = distribution_matching_latent_instance(
                                    image_real, image_syns[client_idx][i], class_proto, 
                                    optimizer_img, self.args
                                )
                            image_syns[client_idx][i].data = updated_syn.data
                            batch_loss += loss
                        
                        dd_loss_avg += batch_loss / (end_idx - start_idx)
                    
                    dd_loss_avg /= num_batches
                else:
                    image_real = []
                    image_real_prob = []
                    for c in range(len(indices_class)):
                        ft, prob = get_images(self.args, images_all, indices_class, c, len(indices_class[c]), class_proto[c])
                        image_real.append(ft)
                        image_real_prob.append(prob)
                    # check if image_syns[client_idx] requries grad update  syn_image_ft[client_idx]
                    if self.args.use_generate_prior:
                        loss, image_syns[client_idx]= distribution_matching_woMIL_wG(
                                                                G,
                                                                image_real,
                                                                image_syns[client_idx],
                                                                optimizer_img,
                                                                len(indices_class),
                                                                self.args.syn_size,
                                                                self.args.ipc,
                                                                self.args.nps,
                                                                args=self.args)
                        # Clear image_real after use to free memory
                        del image_real, image_real_prob
                    elif self.args.use_latent_prior:
                        loss, image_syns[client_idx]= distribution_matching_woMIL_latent(
                                                                image_real,
                                                                image_syns[client_idx],
                                                                image_real_prob,
                                                                label_syns[client_idx],
                                                                class_proto,
                                                                optimizer_img,
                                                                len(indices_class),
                                                                args=self.args)
                        # Clear image_real after use to free memory
                        del image_real, image_real_prob
                    else:
                        loss, image_syns[client_idx]= distribution_matching_woMIL(
                                                                image_real,
                                                                image_syns[client_idx],
                                                                image_real_prob,
                                                                label_syns[client_idx],
                                                                optimizer_img,
                                                                3,
                                                                len(indices_class),
                                                                self.args.syn_size,
                                                                len(images_all),
                                                                self.args.nps,
                                                                args=self.args,
                                                                loss_fn=self.mil_loss,)
                        # Clear image_real after use to free memory
                        del image_real, image_real_prob
                    # report averaged loss
                    dd_loss_avg += loss
                    dd_loss_avg /= self.args.n_classes
                pbar_dd.set_description(f'Client {client_idx} - DD loss {dd_loss_avg}')
                if it % self.args.test_iter == 0 or it == inv_iters - 1:
                    # if self.args.use_adaptive_aggregation:
                    syn_image_ft = [[] for _ in range(len(self.n_clients))]
                    if self.args.use_latent_prior:
                        syn_image_ft[client_idx] = image_syns[client_idx]
                    else:
                        syn_image_ft[client_idx] = raw_feature_extract(self.args, image_syns[client_idx])
                        syn_image_ft[client_idx] = syn_image_ft[client_idx].reshape((self.args.ipc * self.args.n_classes, self.args.nps, -1))
                    virtual_dataset = TensorDataset(syn_image_ft[client_idx].detach(), label_syns[client_idx].detach())
                    virtual_dataloader = DataLoader(virtual_dataset, batch_size=1, shuffle=True)
                    test_model = define_model(self.args)#copy.deepcopy(clients_models_pre[client_idx])
                    test_model_opt = get_optim(self.args, test_model)
                    self.local_train(client_idx, test_model, test_model_opt, virtual_dataloader)     
                    local_test_loader = self.get_test_loader(self.test_dataset[client_idx])
                    _, test_err, _, _, _, _, test_probs = self.local_test(test_model, local_test_loader)                        
                    self.logger.info('====> client = %2d, iter = %2d, loss = %.4f, acc = %.4f' % (client_idx, it, dd_loss_avg, 1-test_err))
                    # save_syn_img(image_syns[client_idx], self.args.results_dir, iter=it, client_idx=client_idx, G=G)
                    # save current synthetic images
                    if 1-test_err > client_dd_acc[client_idx]:
                        client_dd_acc[client_idx] = 1-test_err
                        client_dd_loss[client_idx] = dd_loss_avg
                        client_dd_probs[client_idx] = test_probs
                        client_dd_best_iter[client_idx] = it
                        save_client = os.path.join(self.args.results_dir, f'{repeat}/client{client_idx}/{client_dd_best_iter[client_idx]}')
                        if not os.path.exists(save_client):
                            os.makedirs(save_client)
                        torch.save(image_syns[client_idx].detach().cpu(), f'{save_client}/synthetic_images.pt')
            # Clean up cached data
            if self.args.instance_learn:
                del cached_images, cached_probs
                if cluster_ids_cache is not None:
                    del cluster_ids_cache
            del images_all, labels_all, class_proto
            torch.cuda.empty_cache()

        self.logger.info('=========================Synthetic data training done=========================')
        self.logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        self.logger.info(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        # ''' Prepare mixup vitual data '''
        # convert synthetic data to feature
        syn_image_ft = [[] for _ in range(len(self.n_clients))]
        with torch.no_grad():
            for client_idx in range(len(self.n_clients)):
                image_syns[client_idx] = torch.load(f'{self.args.results_dir}/{repeat}/client{client_idx}/{client_dd_best_iter[client_idx]}/synthetic_images.pt')
                self.logger.info(f'Load synthetic images with informative acc: {client_dd_acc[client_idx]} and loss: {client_dd_loss[client_idx]}')
                if self.args.use_latent_prior:
                    syn_image_ft[client_idx].append(image_syns[client_idx])
                else:
                    ft_tmp = raw_feature_extract(self.args, image_syns[client_idx])
                    syn_image_ft[client_idx].append(ft_tmp)
                    del ft_tmp
                syn_image_ft[client_idx] = torch.cat(syn_image_ft[client_idx], dim=0).detach().cpu()

        self.logger.info('=========================Synthetic data to feature done=========================')
        self.logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        self.logger.info(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        self.logger.info('=========================Mixup vitual data=========================')
        global_virtual_fts = [copy.deepcopy(syn_image_ft[client_idx].detach().cpu()) for client_idx in
                                 range(len(self.n_clients))]
        global_virtual_labels = [copy.deepcopy(label_syns[client_idx].detach().cpu()) for client_idx in
                                 range(len(self.n_clients))]
        for client_idx in range(len(self.n_clients)):
            print(global_virtual_fts[client_idx].size(), global_virtual_labels[client_idx].size())
        mixup_train_set = [[] for _ in range(len(self.n_clients))]
        for client_idx in range(len(self.n_clients)):
            mixup_train_set[client_idx] = TensorDataset(global_virtual_fts[client_idx], global_virtual_labels[client_idx])
            shuffled_idx = list(range(0, len(mixup_train_set[client_idx])))
            random.shuffle(shuffled_idx)
            mixup_train_set[client_idx] = torch.utils.data.Subset(mixup_train_set[client_idx], shuffled_idx)
        self.logger.info('=========================Mixup vitual data done=========================')
        self.logger.info(f"Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB")
        self.logger.info(f"Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB")

        # the current client need to concate with other clients' mixup data
        concated_train_sets = [[] for _ in range(len(self.n_clients))]
        concated_train_loaders = [[] for _ in range(len(self.n_clients))]
        for client_idx in range(len(self.n_clients)):
            all_train_dataset = [self.train_dataset[client_idx]]
            for other_client_idx in range(len(self.n_clients)):
                if other_client_idx != client_idx:
                    all_train_dataset.append(mixup_train_set[other_client_idx])
            concated_train_sets[client_idx] =  torch.utils.data.ConcatDataset(all_train_dataset)
            concated_train_loaders[client_idx] = torch.utils.data.DataLoader(concated_train_sets[client_idx], batch_size=1, shuffle=True, num_workers=0)
        for client_idx in range(len(self.n_clients)):
            print(f'Client {client_idx} has {len(concated_train_sets[client_idx])} compared to original {len(self.train_dataset[client_idx])}')

        # '''Final training'''
        best_acc_per_agent = []
        # clients_models_pre = [None for _ in range(len(self.n_clients))]
        for client_idx in range(len(self.n_clients)):
            train_loader_client = self.get_train_loader(self.train_dataset[client_idx])
            test_loader_client = self.get_test_loader(self.test_dataset[client_idx])
            # clients_models_pre[client_idx] = define_model(self.args)
            clients_models_pre[client_idx].to(self.device)
            optimizer_client = get_optim(self.args, clients_models_pre[client_idx])
            test_acc_best = 0
            test_f1_best = 0
            test_auc_best = 0
            # proto_loss_fn = nn.KLDivLoss(reduction='batchmean')
            local_enhanced_epoch = self.args.local_epochs * 2
            for ep in range(local_enhanced_epoch):
                batch_loss = 0.
                batch_error = 0.
                if ep < self.args.local_epochs:
                    self.logger.info(f'Client {client_idx} LOCAL DATA Training Stage Epoch {ep}/{self.args.local_epochs}')
                    for batch_idx, (images, labels) in enumerate(train_loader_client):
                        images, labels = images.to(self.device), labels.to(self.device)
                        labels = labels.squeeze(0)
                        labels = labels.long()
                        optimizer_client.zero_grad()
                        # images = mix_aug(images, same_label_virtual_proto, mode='append', rate=mixup_rate, strength=0.5)
                        loss, error, y_prob = self.mil_run(clients_models_pre[client_idx], images, labels.unsqueeze(0), self.mil_loss)
                else:
                    self.logger.info(f'Client {client_idx} MIXUP DATA Training Stage Epoch {ep}/{self.args.local_epochs}')
                    for batch_idx, (images, labels) in enumerate(concated_train_loaders[client_idx]):
                        images, labels = images.to(self.device), labels.to(self.device)
                        labels = labels.long()
                        optimizer_client.zero_grad()
                        loss, error, y_prob = self.mil_run(clients_models_pre[client_idx], images[0], labels, GCELoss(q=0.7))
                    loss.backward()
                    optimizer_client.step()
                    batch_loss += loss.item()

                batch_loss /= len(train_loader_client)
                batch_error /= len(train_loader_client)
                test_loss, test_error, fpr, tpr, f1, auc, _ = self.local_test(clients_models_pre[client_idx], test_loader_client)

                test_acc_new = 1-test_error
                if test_acc_new > test_acc_best:
                    test_acc_best = test_acc_new
                    test_f1_best = f1
                    test_auc_best = auc
                self.logger.info(f'Client {client_idx} Test Loss: {test_loss}, Test Acc: {1-test_error}, Best Test Acc: {test_acc_best}, Best Test F1: {test_f1_best}, Best Test AUC: {test_auc_best}')

            self.logger.info(f'============ Client {client_idx} Best Test Acc: {test_acc_best} =================')
            best_acc_per_agent.append(test_acc_best)

        best_acc_overall = np.mean(best_acc_per_agent)
        list_acc_wt = [0] * len(self.n_clients)
        for i in range(len(self.n_clients)):
            list_acc_wt[i] = best_acc_per_agent[i] * self.weight_list[i]
        train_acc_wt = sum(list_acc_wt)
        return best_acc_overall, train_acc_wt, best_acc_per_agent