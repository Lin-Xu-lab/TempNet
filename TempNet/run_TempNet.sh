#!/bin/bash

python TempNet_train.py --data_dir /home/lexieh/spaunion/YuanshengData_TempNet --temporal_key stage --temporal_labels E10.5,E11.5,E12.5,E13.5 --predictor_key cell_type --num_epochs 20 --save_dir /home/lexieh/spaunion/test

