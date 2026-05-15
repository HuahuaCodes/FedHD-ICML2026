import time
import numpy as np
import gc
import faiss
import torch.nn.functional as F
from tqdm import tqdm
from model.resnet_custom import resnet50_baseline
import torch
from scipy.spatial.distance import cdist
from torchvision import transforms
from utils.Get_model import define_model
from utils.core_util import clam_runner
from utils.augment import DiffAug
import torch
import torch.nn.functional as F
from sklearn.cluster import MiniBatchKMeans

from torch.utils.data import TensorDataset, DataLoader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ======== Helper: Pairwise Distance ============
def compute_pairwise_distances(x, y):
    # x: [N, D], y: [M, D]
    return torch.cdist(x, y, p=2)  # Euclidean distance

# ======== Helper: RBF Kernel ===================
def rbf_kernel(x, y, sigma=5.0):
    dists = compute_pairwise_distances(x, y)
    return torch.exp(-dists ** 2 / (2 * sigma ** 2))

def batched_rbf_kernel(x, y, sigma=5.0, batch_size=1024):
    result = []
    for i in range(0, x.size(0), batch_size):
        chunk = x[i:i+batch_size]
        dists = torch.cdist(chunk, y, p=2)
        k = torch.exp(-dists ** 2 / (2 * sigma ** 2))
        result.append(k)
    return torch.cat(result, dim=0)

# ======== MMD Loss =============================
def compute_mmd(x, y, output_prob, sigma=5.0):
    sampled_indices = torch.multinomial(output_prob, 1024, replacement=True)
    x = x[sampled_indices]
    K_xx = rbf_kernel(x, x, sigma).mean()
    K_yy = rbf_kernel(y, y, sigma).mean()
    K_xy = rbf_kernel(x, y, sigma).mean()
    return K_xx + K_yy - 2 * K_xy

# ======== Diversity Loss (optional) ============
def diversity_loss(syn_features):
    # Encourage synthetic features to be diverse (spread out)
    mean = syn_features.mean(dim=0, keepdim=True)
    var = (syn_features - mean).pow(2).mean()
    return -var  # negative variance to maximize diversity

# ======== Mean and Covariance Loss =============
def batchwise_mean_cov(features, batch_size=10000):
    N, D = features.shape
    device = features.device

    # Compute mean
    mean = torch.zeros(D, device=device)
    for i in range(0, N, batch_size):
        batch = features[i:i+batch_size]
        mean += batch.sum(dim=0)
    mean /= N

    # Compute covariance
    cov = torch.zeros(D, D, device=device)
    for i in range(0, N, batch_size):
        batch = features[i:i+batch_size]
        batch_centered = batch - mean
        cov += batch_centered.T @ batch_centered
    cov /= (N - 1)

    return mean, cov

def mean_cov_loss(real_feats, syn_feats, batch_size=10000):
    mean_real, cov_real = batchwise_mean_cov(real_feats, batch_size)
    mean_syn = syn_feats.mean(dim=0)
    syn_centered = syn_feats - mean_syn
    cov_syn = syn_centered.T @ syn_centered / (syn_feats.size(0) - 1)

    D = real_feats.size(1)
    loss_mean = torch.norm(mean_real - mean_syn, p=2) ** 2 / D
    loss_cov = torch.norm(cov_real - cov_syn, p='fro') ** 2 / (D * D)
    return loss_mean + loss_cov

def cluster_mean_cov_loss(real_feats, syn_feats, K=16):
    """
    real_feats: Tensor of shape (N_real, D)
    syn_feats:  Tensor of shape (N_syn, D)
    """
    device = real_feats.device
    D = real_feats.size(1)

    # ---- Step 1: Run KMeans on CPU for both real and synthetic ---- #
    with torch.no_grad():
        real_np = real_feats.detach().cpu().numpy()
        syn_np = syn_feats.detach().cpu().numpy()

        # Guard: NaN in features (from exploding gradients) crashes KMeans
        if np.isnan(syn_np).any() or np.isnan(real_np).any():
            return torch.tensor(0.0, device=device, requires_grad=False)
        kmeans_real = MiniBatchKMeans(n_clusters=K, batch_size=2048, n_init='auto').fit(real_np)
        kmeans_syn = MiniBatchKMeans(n_clusters=K, batch_size=2048, n_init='auto').fit(syn_np)

        real_labels = torch.tensor(kmeans_real.labels_, device=device)
        syn_labels = torch.tensor(kmeans_syn.labels_, device=device)
    # use faiss
    # real_labels, real_centroids = faiss_kmeans_cluster(real_feats, K)
    # syn_labels, syn_centroids = faiss_kmeans_cluster(syn_feats, K)
    # ---- Step 2: Cluster-wise mean and covariance loss ---- #
    loss = 0.0
    matched = 0

    for i in range(K):
        real_cluster = real_feats[real_labels == i]
        syn_cluster = syn_feats[syn_labels == i]
        if real_cluster.size(0) > 5 and syn_cluster.size(0) > 5:
            loss += mean_cov_loss(real_cluster, syn_cluster)
            matched += 1
    return loss / matched if matched > 0 else torch.tensor(0.0, device=device)
    
    # print('Shape of real_labels: ', real_labels.shape)
    # print('Shape of syn_labels: ', syn_labels.shape)

    # # ---- Step 2: Match clusters by nearest centroid ---- #
    # real_centroids = torch.tensor(kmeans_real.cluster_centers_, device=device)
    # syn_centroids = torch.tensor(kmeans_syn.cluster_centers_, device=device)

    # dist = torch.cdist(real_centroids, syn_centroids)  # (K, K)
    # matching = dist.argmin(dim=1)  # Match real cluster i to syn cluster matching[i]

    # # ---- Step 3: Compute loss over matched clusters ---- #
    # loss = 0.0
    # matched = 0
    # for i in range(K):
    #     real_i = real_feats[real_labels == i]
    #     syn_i = syn_feats[syn_labels == matching[i]]
    #     if real_i.size(0) > 5 and syn_i.size(0) > 5:
    #         loss += mean_cov_loss(real_i, syn_i)
    #         matched += 1

    # if matched > 0:
    #     loss = loss / matched
    # return loss

def faiss_kmeans_cluster(features, n_clusters=16, n_iter=20, n_init=5, use_cuda=True):
    """
    features: torch.Tensor of shape [N, D]
    Returns:
        cluster_ids: torch.LongTensor of shape [N]
        centroids: torch.Tensor of shape [n_clusters, D]
    """
    assert use_cuda and torch.cuda.is_available(), "FAISS clustering requires CUDA."
    features_np = features.detach().cpu().numpy().astype(np.float32)
    # print('features_np.shape: ', features_np.shape)
    # Setup FAISS KMeans
    d = features_np.shape[1]
    # num_gpus = faiss.get_num_gpus()
    # # print(f"Running FAISS KMeans on {num_gpus} GPUs...")

    kmeans = faiss.Kmeans(
        d=d,
        k=n_clusters,
        niter=n_iter,
        nredo=n_init,
        verbose=False,
        gpu=True
    )

    kmeans.train(features_np)
    _, cluster_ids = kmeans.index.search(features_np, 1)  # Nearest centroid index

    # Convert outputs
    cluster_ids = torch.tensor(cluster_ids.squeeze(), dtype=torch.long, device=features.device)
    centroids = torch.tensor(kmeans.centroids, dtype=torch.float, device=features.device)

    return cluster_ids, centroids

def compute_cluster_stats(feats, cluster_ids, n_clusters):
    D = feats.shape[1]
    means = torch.zeros((n_clusters, D), device=feats.device)
    covs = torch.zeros((n_clusters, D, D), device=feats.device)
    for k in range(n_clusters):
        cluster_k = feats[cluster_ids == k]
        if cluster_k.shape[0] == 0:
            continue  # skip empty clusters
        mean_k = cluster_k.mean(dim=0)
        centered_k = cluster_k - mean_k
        cov_k = centered_k.T @ centered_k / (cluster_k.shape[0] - 1)
        means[k] = mean_k
        covs[k] = cov_k
    return means, covs

def cluster_structure_loss(real_feats, real_ids, syn_feats, syn_ids, n_clusters=16,
                           lambda_mean_cov=1.0, lambda_centroid=1.0):
    # Compute per-cluster statistics
    real_means, real_covs = compute_cluster_stats(real_feats, real_ids, n_clusters)
    syn_means, syn_covs = compute_cluster_stats(syn_feats, syn_ids, n_clusters)

    # Mean matching and covariance matching
    loss_mean_cov = mean_cov_loss(real_feats, syn_feats)

    # Cluster layout matching: distances between centroids
    dist_real = torch.cdist(real_means, real_means, p=2)
    dist_syn = torch.cdist(syn_means, syn_means, p=2)
    loss_centroid = torch.norm(dist_real - dist_syn, p='fro') ** 2
    # print('loss values: ', loss_mean_cov, loss_centroid)
    # Combine
    total_loss = lambda_mean_cov * loss_mean_cov + lambda_centroid * loss_centroid
    return total_loss


# ======== Final Combined Loss ==================
def distillation_loss(output_real, syn_features, output_prob, lambda_mean=1.0, lambda_mmd=1.0, lambda_div=0.1):
    """
    output_real: tensor of shape [N, D]
    syn_features: tensor of shape [20, D] (image_syn[c])
    """
    print(output_real.size(), syn_features.size())
    # Mean Matching Loss
    loss_mean = F.mse_loss(output_real.mean(0), syn_features.mean(0))

    # MMD Loss (distribution alignment)
    loss_mmd = compute_mmd(output_real, syn_features, output_prob, sigma=5.0)
    # loss_mmd = torch.tensor(0.0).to(device)

    # Optional: Diversity Loss to spread synthetic features
    loss_div = diversity_loss(syn_features)

    # Total Loss
    total_loss = lambda_mean * loss_mean + lambda_mmd * loss_mmd + lambda_div * loss_div

    return total_loss, {
        'mean_loss': loss_mean.item(),
        'mmd_loss': loss_mmd.item(),
        'diversity_loss': loss_div.item()
    }

def mix_aug(src_feats, tgt_feats, mode='replace', rate=[0.3], strength=0.5, shift=None):
    """
    src_feats: (N, D)
    tgt_feats: (M, D)
    shift: Optional tensor for cov mode (M, K, D) or similar
    """

    assert mode in ['replace', 'append', 'interpolate', 'cov', 'joint']
    device = src_feats.device
    N, D = src_feats.shape
    M = tgt_feats.shape[0]
    #=========================================================
    # ---- Compute pairwise distance (fast using einsum / broadcast) ----
    # dist[i, j] = ||src_i - tgt_j||^2
    dist = (
        src_feats.pow(2).sum(dim=1, keepdim=True)
        - 2 * src_feats @ tgt_feats.t()
        + tgt_feats.pow(2).sum(dim=1, keepdim=True).t()
    )
    closest_idxs = torch.argmin(dist, dim=1)  # (N,)
    # Get number of unique elements
    # num_unique = torch.unique(closest_idxs).numel()
    # print("Number of unique elements:", num_unique)
    rate_tensor = torch.tensor(rate, device=device)
    rate_tensor = rate_tensor[closest_idxs] 

    # ---- Random mask for operations ----
    prob_mask = torch.rand(N, device=device) <= rate_tensor

    # Collect results dynamically using list (still efficient)
    auged_feats = [src_feats.clone()]  # start with original

    # ---- Replace ----
    if mode in ['replace', 'joint']:
        replace_mask = torch.rand(N, device=device) <= rate_tensor if mode == 'joint' else prob_mask
        replaced = src_feats.clone()
        replaced[replace_mask] = tgt_feats[closest_idxs[replace_mask]]
        auged_feats[0] = replaced  # replace original entries

    # ---- Append ----
    if mode in ['append', 'joint']:
        append_mask = torch.rand(N, device=device) <= rate_tensor if mode == 'joint' else prob_mask
        auged_feats.append(tgt_feats[closest_idxs[append_mask]])

    # ---- Interpolate ----
    if mode in ['interpolate', 'joint']:
        interp_mask = torch.rand(N, device=device) <= rate_tensor if mode == 'joint' else prob_mask
        interp_base = auged_feats[0][interp_mask]
        interp_tgt = tgt_feats[closest_idxs[interp_mask]]
        generated = (1 - strength) * interp_base + strength * interp_tgt
        auged_feats.append(generated)

    # ---- Cov (optional) ----
    # if mode in ['cov', 'joint'] and shift is not None:
    #     cov_mask = torch.rand(N, device=device) <= rate if mode == 'joint' else prob_mask
    #     # Random index along shift dimension
    #     rand_idx = torch.randint(shift.size(1), (cov_mask.sum(),), device=device)
    #     cov_base = auged_feats[0][cov_mask]
    #     cov_shift = shift[closest_idxs[cov_mask], rand_idx]  # (selected, D)
    #     cov_gen = cov_base + strength * cov_shift
    #     auged_feats.append(cov_gen)

    # ---- Concatenate ----
    return torch.cat(auged_feats, dim=0)

def distribution_matching(dm_iter, image_real, image_syn, optimizer_img, channel, num_classes, im_size, ipc, nps, args=None, loss_fn=None):
    lambda_sim = 0.5

    net = define_model(args)
    net.train()
    # default we use ConvNet
    # if net == None:
    #     net = get_network('ConvNet', channel, num_classes, im_size).to(device)  # get a random model
    #     net.train()
    #     # for param in list(net.parameters()):
    #     #     param.requires_grad = False
    # elif net == 'clam':
    #     net = define_model(args)
    # else:
    #     net.train()
    #     # for param in list(net.parameters()):
    #     #     param.requires_grad = False
    #
    # embed = net.module.embed if torch.cuda.device_count() > 1 else net.embed  # for GPU parallel
    #
    # loss_avg = 0

    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(device)
    for c in range(num_classes):
        img_real = image_real[c]
        labels = torch.tensor([c] * ipc, dtype=torch.long, device=img_real.device)
        if img_real.size(0) == 0:
            continue
        img_syn = image_syn[c * ipc:(c + 1) * ipc].reshape((ipc, nps, channel, im_size, im_size))
        # print('DM image size check ', img_real.size(), img_syn.size())

        seed = int(time.time() * 1000) % 100000
        # dsa_param = ParamDiffAug()
        # img_real = DiffAugment(img_real, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)
        # img_syn = DiffAugment(img_syn, 'color_crop_cutout_flip_scale_rotate', seed=seed, param=dsa_param)

        # output_real = embed(img_real).detach()
        # output_syn = embed(img_syn)
        img_real = img_real.to(device)
        img_syn = img_syn.to(device)
        labels = labels.to(device)
        _, _, output_real = clam_runner(args,
                                         net,
                                         img_real.squeeze(0),
                                         labels[0].unsqueeze(0),
                                         loss_fn,
                                         return_feature=True)
        output_syn = []
        out_ft = []
        for slide_idx in range(ipc):
            _, _, output_syn_per_slide, data_feature = clam_runner(args,
                                                                     net,
                                                                     img_syn[slide_idx],
                                                                     labels[0].unsqueeze(0),
                                                                     loss_fn,
                                                                     return_feature=True,
                                                                     raw_image=True,
                                                                     return_raw_image=True)

            output_syn.append(output_syn_per_slide)
            out_ft.append(data_feature)
            # print('Size of output data feature', data_feature.size())
        output_syn = torch.cat(output_syn, dim=0)
        out_ft = torch.cat(out_ft, dim=0)
        output_real = output_real.detach()
        img_real = img_real.detach()
        loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
        loss += torch.sum((torch.mean(img_real.squeeze(0), dim=0) - torch.mean(out_ft, dim=0)) ** 2)

    # # l2 and total variation loss
    # loss += lambda_sim * l2_norm(img_syn)
    # loss += lambda_sim * total_variation(img_syn)

    optimizer_img.zero_grad()
    loss.backward()
    # total_norm = 0.
    # for group in optimizer_img.param_groups:
    #     for param in group['params']:
    #         if param.requires_grad:
    #             total_norm += param.grad.data.norm(2).item() ** 2.
    # total_norm = total_norm ** .5
    optimizer_img.step()
    #print the gradient of the synthetic image
    print(f'Syn image gradient max: {torch.max(image_syn.grad)} and min: {torch.min(image_syn.grad)} and avg: {torch.mean(image_syn.grad)}')
    return loss.item(), image_syn  # , total_norm

def distribution_matching_woMIL(image_real, image_syn, image_real_prob, label_syns, optimizer_img, channel, num_classes, im_size, ipc, nps, args=None, loss_fn=None):
    lambda_sim = 0.5
    # net = resnet50_baseline(pretrained=True).to(device).train()
    if 'ViT' in args.ft_model:
        vit_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=3, global_pool='avg',
                          num_classes=2)
        variant = "vit_small_patch16_224"
        net = ViT(None, vit_kwargs, variant, True).to(device).train()  # ,
    else:
        net = resnet50_baseline(pretrained=True).to(device).train()
    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(device)
    for c in range(num_classes):
        output_real = image_real[c].to(device)  # real images are already compressed to features
        output_real_prob = image_real_prob[c].to(device).detach()
        output_real = output_real.squeeze(0)
        output_real = output_real.detach()
        if output_real.size(0) == 0:
            continue
        img_syn = image_syn.reshape((ipc * nps, channel, im_size, im_size))
        # form a dataloader using the synthetic images
        syn_loader = DataLoader(TensorDataset(img_syn), batch_size=ipc, shuffle=True)
        output_syn = []
        for img_syn in syn_loader:
            img_syn = img_syn[0].to(device)
            if 'ViT' in args.ft_model:
                img_syn = transforms.Resize(224)(img_syn)
            if args.sn:
                im_mean, im_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                aug, aug_rand = diffaug(args, mean=im_mean, std=im_std)
                img_syn = aug(img_syn)
            batch_output = net(img_syn)   
            output_syn.append(batch_output)
            # Clear cache after each batch
            del img_syn
            torch.cuda.empty_cache()
        output_syn = torch.cat(output_syn, dim=0)
        output_syn = output_syn[label_syns==c]
        # loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)
        run_loss, run_loss_value = distillation_loss(output_real, output_syn, output_real_prob, lambda_mean=1.0, lambda_mmd=1.0, lambda_div=0.1)
        for loss_item in run_loss_value:
            print(f'{loss_item}: {run_loss_value[loss_item]}')  
        loss += run_loss
    optimizer_img.zero_grad()
    loss.backward()
    optimizer_img.step()
    # print the gradient of the synthetic image
    print(
        f'Syn image gradient max: {torch.max(image_syn.grad)} and min: {torch.min(image_syn.grad)} and avg: {torch.mean(image_syn.grad)}')
    return loss.item(), image_syn#,  image_syn_ft# , total_norm

def distribution_matching_woMIL_latent(image_real, image_syn, image_real_prob, label_syns, class_proto, optimizer_img, num_classes, args=None, loss_fn=None):
    ''' update synthetic data '''
    # Clear cache before processing
    torch.cuda.empty_cache()
    
    loss = torch.tensor(0.0).to(device)
    for c in range(num_classes):
        output_real = image_real[c].to(device)  # real images are already compressed to features
        # output_real_prob = image_real_prob[c].to(device).detach()
        output_real = output_real.squeeze(0)
        output_real = output_real.detach()
        if output_real.size(0) == 0:
            # Clean up tensors for this class
            del output_real
            continue
        if args.module == 'syn_data':
            image_syn_c = image_syn[label_syns==c].reshape((-1, image_syn[label_syns==c].size(-1)))
            if args.cluster:
                loss = cluster_mean_cov_loss(output_real, image_syn_c)
            else:
                loss += mean_cov_loss(output_real, image_syn_c)
        else:
            strength = np.random.uniform(0, 1)
            output_real = mix_aug(output_real, class_proto[c].unsqueeze(0).to(device),
                                shift=None,
                                rate=[0.5], strength=strength, mode='joint').to(device)
            image_syn_c = image_syn[label_syns==c].reshape((-1, image_syn[label_syns==c].size(-1)))
            loss += torch.sum((torch.mean(output_real, dim=0) - torch.mean(image_syn_c, dim=0)) ** 2)
        # run_loss, run_loss_value = distillation_loss(output_real, image_syn_c, output_real_prob, lambda_mean=1.0, lambda_mmd=1.0, lambda_div=0.1)
        # for loss_item in run_loss_value:
        #     print(f'{loss_item}: {run_loss_value[loss_item]}')  
        # loss += run_loss
        # Free memory after each class
        del output_real, image_syn_c
        torch.cuda.empty_cache()
    optimizer_img.zero_grad()
    loss.backward()
    optimizer_img.step()
    # print the gradient of the synthetic image
    # print(
    #     f'Syn image gradient max: {torch.max(image_syn.grad)} and min: {torch.min(image_syn.grad)} and avg: {torch.mean(image_syn.grad)}')
    return loss.item(), image_syn#,  image_syn_ft# , total_norm

def distribution_matching_woMIL_wG(G, image_real, image_syn, optimizer_img, num_classes, im_size, ipc, nps,
                                   args=None):
    lambda_sim = 0.5
    # define feature extractor
    if 'ViT' in args.ft_model:
        vit_kwargs = dict(patch_size=16, embed_dim=384, depth=12, num_heads=3, global_pool='avg',
                          num_classes=2)
        variant = "vit_small_patch16_224"
        net = ViT(None, vit_kwargs, variant, True).to(device).train()
    else:
        net = resnet50_baseline(pretrained=True).to(device).train()
    for param in list(net.parameters()):
        param.requires_grad = False

    # Generate synthetic images (no gradient storage needed here)
    with torch.no_grad():
        image_syn_slides = []
        _batch_size = 1
        pbar = tqdm(range(num_classes * ipc))

        for i in pbar:
            pbar.set_description(f"Generating synthetic images with G {i}/{num_classes * ipc}")
            x_i = image_syn[i]  # shape (100, 256)

            # Process patches for this slide
            slide_patches = []
            nps = x_i.size(0) // _batch_size

            for j in range(nps):
                x_i_tmp = x_i[j * _batch_size:(j + 1) * _batch_size]
                x_i_tmp = x_i_tmp.view(_batch_size, 1, -1)

                # Generate image from latent (no gradients needed here)
                out_i_tmp = G.gen_image(x_i_tmp)
                out_i_tmp = F.interpolate(out_i_tmp, size=(im_size, im_size), mode='bilinear', align_corners=False)
                slide_patches.append(out_i_tmp)

            # Concatenate patches for this slide
            slide_tensor = torch.cat(slide_patches, dim=0).unsqueeze(0)
            image_syn_slides.append(slide_tensor)

            pbar.set_postfix({'RAM': f"{torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB"})

    # Concatenate all slides and enable gradients for loss computation (n, 256)-> (n, 3, 64, 64)
    image_syn_generated = torch.cat(image_syn_slides, dim=0)
    image_syn_generated = image_syn_generated.detach().requires_grad_(True)

    ''' update synthetic data '''
    loss = torch.tensor(0.0).to(device)
    output_syn = []

    for c in range(num_classes):
        output_real = image_real[c].to(device)
        output_real = output_real.squeeze(0)

        if output_real.size(0) == 0:
            continue

        # Get synthetic images for this class (these have gradients enabled)
        image_syn_per_c = image_syn_generated[c * ipc:(c + 1) * ipc]

        # Use tqdm with leave=False to prevent multiple progress bars
        for slide_idx in tqdm(range(ipc), desc=f"DM Updating {c+1}/{num_classes}"):
            slide_syn = image_syn_per_c[slide_idx].unsqueeze(0)
            for patch_idx in range(nps):
                patch_syn = slide_syn[:, patch_idx, :, :, :]
                patch_syn = patch_syn.to(device)

                if 'ViT' in args.ft_model:
                    patch_syn = transforms.Resize(224)(patch_syn)

                if args.sn:
                    im_mean, im_std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
                    aug, aug_rand = diffaug(args, mean=im_mean, std=im_std)
                    patch_syn = aug(patch_syn)

                output_syn.append(net(patch_syn))
            pbar.set_postfix({'RAM': f"{torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB"})

    output_syn = torch.cat(output_syn, dim=0) #(nm, 1024/768)

    # Convert list of image_real to tensor
    image_real = [img_real.squeeze(0) for img_real in image_real]
    image_real = torch.cat(image_real, dim=0).to(device).detach()

    # Compute loss
    loss += torch.sum((torch.mean(image_real, dim=0) - torch.mean(output_syn, dim=0)) ** 2)

    # Backward pass - compute gradients for image_syn_generated
    optimizer_img.zero_grad()
    loss.backward()

    # Now manually assign gradients from image_syn_generated back to image_syn (latents)
    gan_backward(latents=image_syn, image_syn_grad=image_syn_generated, G=G, im_size=im_size)

    optimizer_img.step()

    # Check gradients
    if image_syn.grad is not None:
        print(
            f'Syn image gradient max: {torch.max(image_syn.grad)} and min: {torch.min(image_syn.grad)} and avg: {torch.mean(image_syn.grad)}')
    else:
        print("Warning: image_syn.grad is None!")

    return loss.item(), image_syn

def gan_backward(latents, image_syn_grad, G, im_size=224):
    """
    Manually propagate gradients from synthetic images back to latent vectors
    This is needed because latents -> G(latents) -> synthetic images
    PyTorch's autograd stops at synthetic images, so we manually compute:
    d_loss/d_latents = d_loss/d_images * d_images/d_latents
    """
    if not latents.requires_grad:
        latents.requires_grad_(True)

    if latents.grad is None:
        latents.grad = torch.zeros_like(latents)

    ns, np, nd = latents.size()

    # Process each slide individually to manage memory
    pbar = tqdm(range(ns))
    for slide_idx in pbar:
        pbar.set_description(f"Updating slides {slide_idx}/{ns}")
        # Get gradients for this slide
        slide_image_grad = image_syn_grad.grad[slide_idx]  # Shape: [np, C, H, W]
        slide_latents = latents[slide_idx]  # Shape: [np, nd]

        # Process patches in batches to manage memory
        batch_size = 4
        slide_latent_grad = torch.zeros_like(slide_latents)

        for batch_start in range(0, np, batch_size):
            batch_end = min(batch_start + batch_size, np)

            # Get batch of latents and corresponding image gradients
            batch_latents = slide_latents[batch_start:batch_end].clone().detach().requires_grad_(True)
            batch_image_grad = slide_image_grad[batch_start:batch_end]

            # Generate images for this batch
            batch_latents_reshaped = batch_latents.unsqueeze(1)  # Add channel dim
            generated_batch = G.gen_image(batch_latents_reshaped)
            generated_batch = F.interpolate(generated_batch, size=(im_size, im_size),
                                            mode='bilinear', align_corners=False)

            # Compute gradients: d_images/d_latents
            # Use torch.autograd.grad to compute Jacobian-vector product
            grad_outputs = batch_image_grad

            try:
                latent_grads = torch.autograd.grad(
                    outputs=generated_batch,
                    inputs=batch_latents,
                    grad_outputs=grad_outputs,
                    retain_graph=False,
                    create_graph=False
                )

                if latent_grads[0] is not None:
                    slide_latent_grad[batch_start:batch_end] = latent_grads[0]

            except RuntimeError as e:
                print(f"Warning: Could not compute gradients for batch {batch_start}-{batch_end}: {e}")
                continue

            # Clean up
            del batch_latents, generated_batch, grad_outputs
            torch.cuda.empty_cache()

        # Accumulate gradients for this slide
        latents.grad[slide_idx] = slide_latent_grad

        # Clean up slide-level variables
        del slide_image_grad, slide_latents, slide_latent_grad
        torch.cuda.empty_cache()
        pbar.set_postfix({'Slide': slide_idx + 1, 'RAM': f"{torch.cuda.memory_allocated(device) / (1024 ** 2):.2f} MB"})

class Normalize():
    def __init__(self, mean, std, device='cpu'):
        self.mean = torch.tensor(mean, device=device).reshape(1, len(mean), 1, 1)
        self.std = torch.tensor(std, device=device).reshape(1, len(mean), 1, 1)

    def __call__(self, x, seed=-1):
        return (x - self.mean) / self.std
def diffaug(args, mean, std, device='cuda'):
    """Differentiable augmentation for condensation
    """
    aug_type = args.aug_type

    normalize = Normalize(mean=mean, std=std, device=device)
    print("Augmentataion Matching: ", aug_type)
    augment = DiffAug(strategy=aug_type, batch=True)
    aug_batch = transforms.Compose([normalize, augment])

    print("Augmentataion Net update: ", aug_type)
    augment_rand = DiffAug(strategy=aug_type, batch=False)
    aug_rand = transforms.Compose([normalize, augment_rand])

    return aug_batch, aug_rand

def distribution_matching_latent_instance(image_real, image_syn, class_proto, optimizer_img, args=None, precomputed_cluster_ids=None):
    ''' update synthetic data '''
    # Clear cache before processing
    torch.cuda.empty_cache()
    
    loss = torch.tensor(0.0).to(device)
    output_real = image_real.to(device)  # real images are already compressed to features
    output_real = output_real.detach()
    if args.cluster:
        if args.faiss_cluster:
            # Use precomputed cluster IDs if available
            if precomputed_cluster_ids is not None:
                cluster_ids_real = precomputed_cluster_ids
                cluster_ids_syn, _ = faiss_kmeans_cluster(image_syn, n_clusters=16)
            else:
                cluster_ids_real, _ = faiss_kmeans_cluster(output_real, n_clusters=16)
                cluster_ids_syn, _ = faiss_kmeans_cluster(image_syn, n_clusters=16)
            loss = cluster_structure_loss(output_real, cluster_ids_real, image_syn, cluster_ids_syn, n_clusters=16, lambda_centroid=0.1)
        else:
            loss = cluster_mean_cov_loss(output_real, image_syn)
    elif args.dd_mix:
        strength = np.random.uniform(0, 1)
        # Fix: class_proto is a dict, need to get the right class
        proto_key = list(class_proto.keys())[0] if len(class_proto) > 0 else 0
        output_real = mix_aug(output_real, class_proto[proto_key].unsqueeze(0).to(device),
                            shift=None,
                            rate=[0.5], strength=strength, mode='joint').to(device)
        loss = torch.sum((torch.mean(output_real, dim=0) - torch.mean(image_syn, dim=0)) ** 2)
    else:
        if 'IDH' in args.task:
            loss = torch.sum((torch.mean(output_real, dim=0) - torch.mean(image_syn, dim=0)) ** 2)
        else:
            loss = mean_cov_loss(output_real, image_syn)
    # Skip update if loss exploded
    if not torch.isfinite(loss):
        return 0.0, image_syn.detach()
    optimizer_img.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_([image_syn], max_norm=1.0)
    optimizer_img.step()
    return loss.item(), image_syn.detach()
    # print the gradient of the synthetic image
    # print(
    #     f'Syn image gradient max: {torch.max(image_syn.grad)} and min: {torch.min(image_syn.grad)} and avg: {torch.mean(image_syn.grad)}')
    # return loss.item(), image_syn#,  image_syn_ft# , total_norm