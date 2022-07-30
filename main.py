import os
import random
import numpy as np
import tensorflow as tf
import argparse
import time
from train import train
from evaluation import *
from tfdeterminism import patch

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Training')
    parser.add_argument('--GPU', type=str, default='0')
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--hid_units', type=int, default=256, help='number of neurons in GAT')
    parser.add_argument('--dense0', type=int, default=256, help='number of neurons in BFN')
    parser.add_argument('--dense1', type=int, default=128, help='number of neurons in BFN')
    parser.add_argument('--layers', type=int, default=4, help='number of layer aggregator in GAT')
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--attention_drop', type=float, default=0.1)
    parser.add_argument('--feedforward_drop', type=float, default=0.1)
    parser.add_argument('--dataset', type=str, default='HMDD')
    args = parser.parse_args()

    patch()
    SEED = 1000
    tf.set_random_seed(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPU

    dataset = args.dataset
    times = 10
    total_KFOLD_test_labels, total_FOLD_test_scores = [], []
    for i in range(times):
        for fold in range(5):
            print("times: %d, fold: %d" % (int(i), int(fold)))
            train_arr = np.loadtxt(f'data/{dataset}/data_dir/{i}/{fold}/train_arr.txt')
            test_arr = np.loadtxt(f'data/{dataset}/data_dir/{i}/{fold}/test_arr.txt')
            train_arr = train_arr.astype(np.int64)
            test_arr = test_arr.astype(np.int64)
            test_labels, scores = train(args, train_arr, test_arr, dataset, i, fold)
            total_KFOLD_test_labels.append(test_labels)
            total_FOLD_test_scores.append(scores)

    statistic_total_AUC(args, total_KFOLD_test_labels, total_FOLD_test_scores)