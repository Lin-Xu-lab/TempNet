#!/bin/bash

python TempNet_train.py --data_dir /home/lexieh/spaunion/breast_cancer_data/proc/proc.er --temporal_key grade --temporal_labels 1,2,3 --predictor_key sample --filter_key cell_type --filter_labels Tumor --num_epochs 20 --mmd_loss_w 0.9 --cosine_loss_w 0.1 --enc_loss_w 0.8 --pred_loss_w 0.01 --disc_loss_w 0.9 --save_dir /home/lexieh/spaunion/test --do_preprocess --use_rev --rev_alpha 1

