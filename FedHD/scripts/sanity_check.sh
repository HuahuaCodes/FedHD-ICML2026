#!/bin/bash
#PBS -P iq24
#PBS -l ncpus=12
#PBS -l ngpus=1
#PBS -q gpuvolta
#PBS -l mem=64GB
#PBS -l storage=gdata/iq24+scratch/iq24
#PBS -l walltime=00:20:00
#PBS -l jobfs=10GB
#PBS -N fedhd_sanity

CODE_ROOT=/scratch/iq24/cc0395/FedHD
FT_ROOT=/g/data/iq24/CAMELYON16_patches
cd $CODE_ROOT
source /g/data/iq24/mmcv_env/bin/activate

python main.py \
  --task CAMELYON16 \
  --exp_code sanity_check \
  --module syn_data \
  --feature_type R50_features \
  --ft_model ResNet50 \
  --mil_method CLAM_SB \
  --heter_model \
  --n_classes 2 \
  --drop_out \
  --lr 3e-3 \
  --opt adamw \
  --bag_loss ce \
  --inst_loss svm \
  --B 8 \
  --ipc 10 \
  --nps 1000 \
  --dc_iterations 1000 \
  --image_lr 0.1 \
  --image_opt sgd \
  --syn_size 1024 \
  --local_epochs 50 \
  --test_iter 50 \
  --data_root_dir $FT_ROOT \
  --results_dir $CODE_ROOT/exp \
  --use_latent_prior \
  --instance_learn \
  --cluster \
  --debug
