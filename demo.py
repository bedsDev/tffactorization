#!/usr/bin/python
"""Demo script for running

Authors : Shun Nukui
License : GNU General Public License v2.0

Usage: python demo.py
"""
from __future__ import print_function
import time
import numpy as np
import tensorflow as tf
from tfnmf import TFNMF

def main():
    #user setting parameters
    V = np.random.rand(10000,10000)
    rank = 10
    num_core = 8

    tfnmf = TFNMF(V, rank,'mu') #'grad')
    config = tf.ConfigProto(inter_op_parallelism_threads=num_core,
                               intra_op_parallelism_threads=num_core)
    
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    
    # config = tf.ConfigProto()
    # config.gpu_options.allow_growth=True
    
    # sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    
    
    with tf.Session(config=config) as sess:
    # with tf.Session(config=tf.ConfigProto(config=config)) as sess:
        start = time.time()
        W, H = tfnmf.run(sess)
        print("Computational Time for TFNMF: ", time.time() - start)

    W = np.mat(W)
    H = np.mat(H)
    error = np.power(V - W * H, 2).sum()
    print("Reconstruction Error for TFNMF: ", error)
    print(W)
    print(H)

if __name__ == '__main__':
    main()
