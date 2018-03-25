#!/bin/bash


python3 train.py --line-suffix quasart \
                --data-type Quasart \
                --gpu 2 \
                --loss-curve quasart_loss \
                --acc-curve quasart_acc \
                --auc-curve quasrt_auc \
                --title quasart_match
