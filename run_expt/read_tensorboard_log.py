# Copyright (C) 2020-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


import sys

sys.path.append("../")

import argparse
import os

import numpy as np
from tensorflow.python.summary.summary_iterator import summary_iterator

from preprocess import build_encoding, omnipush, omnipush_collate_fn


# get min or last test rmse for method m in folder m_path
def getrmse(m_path, num_samp_perbatch, value="test/RMSE", getmin=False):
    mse_list = []
    min_test_mse = None
    logs = os.listdir(m_path)
    if len(logs) == 1:
        try:
            logfile = os.path.join(m_path, logs[0])
            # collect all results for each batch
            for e in summary_iterator(logfile):
                stp = e.step
                for v in e.summary.value:
                    if v.tag == value:
                        mse_list.append(v.simple_value**2)
        except:
            pass
        # weigh by sample size per batch, sum for entire test data,
        # and average to get error per sample
        num_samp = np.tile(num_samp_perbatch, len(mse_list) // len(num_samp_perbatch))
        weighted_mse = np.asarray([mse_list[i] * n for i, n in enumerate(num_samp)])
        test_mse = weighted_mse.reshape((-1, len(num_samp_perbatch))).sum(axis=1) / sum(num_samp_perbatch)
        if getmin:
            eval_test_mse = min(np.sqrt(test_mse))
        else:
            eval_test_mse = np.sqrt(test_mse)[-1]
    return eval_test_mse


# get min or last test nlpd for method m in folder m_path
def getnlpd(m_path, num_samp_perbatch, value="test/nlogP", getmin=False):
    nlpd_list = []
    min_test_nlpd = None
    logs = os.listdir(m_path)
    if len(logs) == 1:
        try:
            logfile = os.path.join(m_path, logs[0])
            # collect all results for each batch
            for e in summary_iterator(logfile):
                stp = e.step
                for v in e.summary.value:
                    if v.tag == value:
                        nlpd_list.append(v.simple_value)
        except:
            pass
        # weigh by sample size per batch, sum for entire test data,
        # and average to get error per sample
        num_samp = np.tile(num_samp_perbatch, len(nlpd_list) // len(num_samp_perbatch))
        weighted_nlpd = np.asarray([nlpd_list[i] * n for i, n in enumerate(num_samp)])
        test_nlpd = weighted_nlpd.reshape((-1, len(num_samp_perbatch))).sum(axis=1) / sum(num_samp_perbatch)
        if getmin:
            eval_test_nlpd = min(test_nlpd)
        else:
            eval_test_nlpd = test_nlpd[-1]
    return eval_test_nlpd


# get min or last test std for method m in folder m_path
def getstd(m_path, num_samp_perbatch, value="test/lstd", getmin=False):
    std_list = []
    min_test_std = None
    logs = os.listdir(m_path)
    if len(logs) == 1:
        try:
            logfile = os.path.join(m_path, logs[0])
            # collect all results for each batch
            for e in summary_iterator(logfile):
                stp = e.step
                for v in e.summary.value:
                    if v.tag == value:
                        std_list.append(np.exp(v.simple_value))
        except:
            pass
        # weigh by sample size per batch, sum for entire test data,
        # and average to get error per sample
        num_samp = np.tile(num_samp_perbatch, len(std_list) // len(num_samp_perbatch))
        weighted_std = np.asarray([std_list[i] * n for i, n in enumerate(num_samp)])
        test_std = weighted_std.reshape((-1, len(num_samp_perbatch))).sum(axis=1) / sum(num_samp_perbatch)
        if getmin:
            eval_test_std = min(test_std)
        else:
            eval_test_std = test_std[-1]
    return eval_test_std


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Read Tensorboard event logs")
    parser.add_argument("--expt", help="experiment name")
    parser.add_argument("--results_dir", default="./runs", help="results folder")
    parser.add_argument("--batch_size", default=64, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("--data", default="../data/0,1,2_weights", help="data folder")
    parser.add_argument("--test_data", default=None, help="test data folder")
    parser.add_argument("--split", default="../split.json", help="data folder")
    parser.add_argument(
        "--methods",
        default=["baseline", "concat", "anp", "cazsl-noreg", "cazsl-l2reg", "cazsl-neuralreg"],
        nargs="*",
        help="list of methods to be retrieved",
    )

    args = parser.parse_args()

    # methods = ['baseline', 'concat', 'anp', 'cazsl-noreg', 'cazsl-l2reg', 'cazsl-neuralreg']
    methods = args.methods

    if "image" in args.expt:
        # not including baseline and anp which do not use context, therefore results are same as without 'image' in args.expt
        methods_expt = [args.expt + "-" + m for m in methods if m not in ["baseline", "anp"]]
    else:
        methods_expt = [args.expt + "-" + m for m in methods]

    # get corresponding folder names
    methods_expt_f = []
    for m in methods_expt:
        f_match = [os.path.join(args.results_dir, f) for f in os.listdir(args.results_dir) if m in f]
        if len(f_match) > 1:  # if multiple matches, take first one
            f_match = [sorted(f_match, key=lambda x: len(x))[0]]
        methods_expt_f += f_match

    print("These two should correspond:")
    print(methods_expt)
    print(methods_expt_f)

    # number of batches to calculate error for entire test data
    if args.test_data == None:
        test_names, test_X, test_y = omnipush(
            args.data, train=False, split_file=args.split, checked_pattern="X_3_norm", normalize=False
        )
    else:
        test_names, test_X, test_y = omnipush(
            args.test_data,
            train=False,
            split_file=args.split,
            checked_pattern="X_3_norm",
            normalize=False,
            return_all=True,
        )
    print(" Shape of testing data: ", test_X.shape)
    test_size = test_X.shape[0]
    num_batch = int(np.ceil(test_size * 1.0 / args.batch_size))

    # number of samples per batch
    if test_size % args.batch_size == 0:
        num_samp_perbatch = [args.batch_size] * num_batch
    else:
        num_samp_perbatch = [args.batch_size] * (num_batch - 1) + [test_size % args.batch_size]

    # find best performance for each method
    rmse_dict = dict.fromkeys(methods_expt, [])
    nlpd_dict = dict.fromkeys(methods_expt, [])
    std_dict = dict.fromkeys(methods_expt, [])
    for i, m in enumerate(methods_expt):
        rmse_dict[m] = getrmse(methods_expt_f[i], num_samp_perbatch, getmin=False)
        nlpd_dict[m] = getnlpd(methods_expt_f[i], num_samp_perbatch, getmin=False)
        # if 'anp' not in m:
        #     std_dict[m] = getstd(methods_expt_f[i], num_samp_perbatch, getmin = False)
        std_dict[m] = getstd(methods_expt_f[i], num_samp_perbatch, getmin=False)

    print("rmse")
    print(rmse_dict)
    print("nlpd")
    print(nlpd_dict)
    print("std")
    print(std_dict)

    """
    # example commands

    python3 read_tensorboard_log.py --expt omnipush
    python3 read_tensorboard_log.py --expt omnipush-image

    python3 read_tensorboard_log.py --expt plywood --test_data ../data/plywood --split ../plywood_split.json

    python3 read_tensorboard_log.py --expt weight-0,1 --split ../0,1_split.json
    python3 read_tensorboard_log.py --expt weight-0,1-image --split ../0,1_split.json
    (similarly for 0,2 and 1,2)
    """
