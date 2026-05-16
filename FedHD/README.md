# FedHD: Federated Histopathology with Heterogeneous Models

Code for the ICML paper. Supports fully heterogeneous federation — each client uses a different feature extractor and MIL aggregator — via feature-space dataset distillation.

---

## Environment

```bash
source /g/data/iq24/mmcv_env/bin/activate
```

---

## Data

| Dataset | Path | Extractors |
|---|---|---|
| CAMELYON16 | `` | R50, UNI, PhikonV2, GPFM |
| CAMELYON17 | `` | R50, PhikonV2, GPFM (**no UNI**) |
| TCGA-IDH | `` | R50, UNI, PhikonV2, GPFM |

Each dataset follows the structure:
```
<root>/{train,test}/{class}/{extractor}/pt_files/*.pt
                                       /h5_files/*.h5
```

---

## Heterogeneous Client Configs (Table 1)

Assigned automatically when `--heter_model` is set.

| Dataset | Client | Extractor | MIL |
|---|---|---|---|
| CAMELYON16 | 0 | R50 | CLAM_SB |
| | 1 | UNI | TransMIL |
| CAMELYON17 | 0 | UNI | CLAM_SB |
| | 1 | R50 | TransMIL |
| | 2 | R50 | ACMIL |
| | 3 | PhikonV2 | TransMIL |
| | 4 | GPFM | CLAM_SB |
| TCGA-IDH | 0–7 | R50/GPFM/R50/UNI/PhikonV2/GPFM/R50/PhikonV2 | CLAM/TransMIL/ACMIL/ACMIL/CLAM/CLAM/ACMIL/CLAM |

---

## Running Experiments

### Two-stage pipeline

**Stage 1 — Synthetic data generation** (`syn_data` module):
Runs FDD + O2O distillation + GMA alignment, then CBF final training.

```bash
qsub scripts/run_cam16_syn.sh   # CAMELYON16
qsub scripts/run_cam17_syn.sh   # CAMELYON17
qsub scripts/run_idh_syn.sh     # TCGA-IDH
```

**Stage 2 — Local training on saved synthetic data** (`local_train` module):
Loads best synthetic data from Stage 1 and re-runs CBF final training.

```bash
qsub scripts/run_cam16_train.sh
qsub scripts/run_cam17_train.sh
qsub scripts/run_idh_train.sh
```

### Quick sanity check

Runs the full pipeline for CAMELYON16 with 1 iteration each (debug mode):

```bash
qsub scripts/sanity_check.sh
```

Log: `logs/CAMELYON16/CLAM_SB_R50_features_sanity_check_logs.txt`

---

## Key Arguments

| Argument | Description |
|---|---|
| `--module` | `syn_data` (full pipeline) or `local_train` (load pre-saved syn data) |
| `--heter_model` | Use Table 1 per-client heterogeneous configs |
| `--use_latent_prior` | **FDD**: distill in feature space (1024-dim tensors) |
| `--instance_learn` | **O2O**: one synthetic slide per real slide |
| `--cluster` | **GMA**: Gaussian mixture alignment (K=16 KMeans mean+cov loss) |
| `--load_syn_data` | Skip distillation, load from `--syn_data_dir` |
| `--syn_data_dir` | Path to pre-saved synthetic data directory |
| `--dc_iterations` | Number of distillation iterations (default: 1000) |
| `--ipc` | Synthetic slides per class (used when `--instance_learn` is off) |
| `--nps` | Patches per synthetic slide (default: 1000) |
| `--syn_size` | Feature dimension (default: 1024) |
| `--image_lr` | Synthetic data learning rate (default: 0.1) |
| `--local_epochs` | Training epochs per client (default: 50) |
| `--debug` | Override to 1 iteration / 1 epoch for quick checks |

---

## Paper Contributions → Code

| Contribution | Flag | Implementation |
|---|---|---|
| **FDD** (feature distillation) | `--use_latent_prior` | `distillation.py: distribution_matching_latent_instance` |
| **O2O** (one-to-one) | `--instance_learn` | `SynGenerator.py`: per-slide distillation loop |
| **GMA** (Gaussian mixture alignment) | `--cluster` | `distillation.py: cluster_mean_cov_loss` |
| **CBF** (curriculum federation) | always on | Stage 1: real data + CE; Stage 2: real + cross-client synthetic + GCE(q=0.7) |

CBF is implemented in the final training loop of both `SynGenerator` (`syn_data` module) and `LocalTrainer` (`local_train` module).

---

## Output Structure

```
exp/<TASK>/<MIL>_<extractor>_<exp_code>/
    client_<i>_<Model>_pretrain.pt        # pretrained model checkpoint
    0/client<i>/<iter>/synthetic_images.pt # best synthetic data per client
```

Results are logged to `logs/<TASK>/<MIL>_<extractor>_<exp_code>_logs.txt`.
