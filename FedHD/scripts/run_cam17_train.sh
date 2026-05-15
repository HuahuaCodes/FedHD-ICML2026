#!/bin/bash
#PBS -P iq24
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=64GB
#PBS -l storage=gdata/iq24+scratch/iq24
#PBS -l walltime=12:00:00
#PBS -l jobfs=20GB
#PBS -N fedhd_cam17_train

CODE_ROOT=/scratch/iq24/cc0395/FedHD
cd $CODE_ROOT
source /g/data/iq24/mmcv_env/bin/activate

python main.py \
  --task CAMELYON17 \
  --exp_code fedhd_full \
  --module local_train \
  --feature_type R50_features \
  --ft_model ResNet50 \
  --mil_method CLAM_SB \
  --heter_model \
  --n_classes 4 \
  --drop_out \
  --lr 3e-3 \
  --opt adamw \
  --bag_loss ce \
  --inst_loss svm \
  --B 8 \
  --ipc 10 \
  --nps 1000 \
  --syn_size 1024 \
  --local_epochs 50 \
  --data_root_dir /g/data/iq24/CAMELYON17_patches/centers \
  --results_dir $CODE_ROOT/exp \
  --use_latent_prior \
  --instance_learn \
  --cluster \
  --load_syn_data \
  --syn_data_dir $CODE_ROOT/exp/CAMELYON17/CLAM_SB_R50_features_fedhd_full
