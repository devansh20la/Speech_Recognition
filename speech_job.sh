#!/bin/sh
CUDA_VISIBLE_DEVICES=2 python main.py --checkpoint=checkpoint_ep400.pth.tar --lr=0.01 
