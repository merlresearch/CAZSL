# Copyright (C) 2020-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


"""
ANP baseline.

Adapted from https://github.com/MIT-Omnipush/omnipush-metalearning-baselines/
"""

import argparse
import os
import random

import numpy as np
import torch as t
import torchvision
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

from network import LatentModel
from preprocess import omnipush, omnipush_collate_fn


def adjust_learning_rate(optimizer, step_num, warmup_step=2000):
    lr = 0.001 * warmup_step**0.5 * min(step_num * warmup_step**-1.5, step_num**-0.5)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def evaluate(model, writer, global_step, loader, name):
    for i, data in enumerate(loader):
        context_x, context_y, target_x, target_y = data
        context_x = context_x.cuda()
        context_y = context_y.cuda()
        target_x = target_x.cuda()
        target_y = target_y.cuda()

        # pass through the latent model
        y_pred, lstd, kl, loss, mse, log_p = model(context_x, context_y, target_x, target_y)
        if mse is not None:
            writer.add_scalar(name + "/RMSE", t.sqrt(mse), global_step)
        writer.add_scalar(name + "/nlogP", -log_p * target_y.shape[-1], global_step)
        writer.add_scalar(name + "/loss", loss, global_step)
        writer.add_scalar(name + "/kl", kl, global_step)
        writer.add_scalar(name + "/lstd", lstd, global_step)


def main():
    # Load data
    shot = 50
    train_names, train_X, train_y = omnipush(
        args.data, train=True, split_file=args.split, checked_pattern="X_3_norm", normalize=False
    )
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
    train_dataset = t.utils.data.TensorDataset(train_X, train_y)
    test_dataset = t.utils.data.TensorDataset(test_X, test_y)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda x: omnipush_collate_fn(x, shot, meta_train=True),
        shuffle=True,
        num_workers=args.workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        collate_fn=lambda x: omnipush_collate_fn(x, shot, meta_train=False),
        shuffle=True,
        num_workers=args.workers,
    )

    boolean_pred = False
    x_dim = 3
    y_dim = 3
    epochs = 3000
    model = LatentModel(num_hidden=128, x_dim=x_dim, y_dim=y_dim, boolean_pred=boolean_pred).cuda()

    optim = t.optim.Adam(model.parameters(), lr=args.lr)
    writer = SummaryWriter(comment=args.job_name)
    global_step = 0
    ebar = tqdm(range(epochs))
    for epoch in ebar:
        model.train()
        pbar = train_loader
        mses = []
        logps = []
        kls = []
        for i, data in enumerate(pbar):
            global_step += 1
            adjust_learning_rate(optim, global_step, warmup_step=2000)
            context_x, context_y, target_x, target_y = data
            context_x = context_x.cuda()
            context_y = context_y.cuda()
            target_x = target_x.cuda()
            target_y = target_y.cuda()

            # pass through the latent model
            y_pred, lstd, kl, loss, mse, log_p = model(context_x, context_y, target_x, target_y)

            # Training step
            optim.zero_grad()
            loss.backward()
            optim.step()

            # Logging
            if mse is not None:
                writer.add_scalar("train/RMSE", t.sqrt(mse), global_step)
                if epoch % 50 == 0:
                    print(epoch, loss.item(), t.sqrt(mse).item())
            writer.add_scalar("train/nlogP", -log_p * target_y.shape[-1], global_step)
            writer.add_scalar("train/loss", loss, global_step)
            writer.add_scalar("train/kl", kl, global_step)
            writer.add_scalar("train/lstd", lstd, global_step)

        model.train(False)
        evaluate(model, writer, global_step, test_loader, "test")

        """
        # Save model by each epoch
        if epoch%500==0:
            t.save({'model':model.state_dict(), 'optimizer':optim.state_dict()},
                    os.path.join('./checkpoints','ANP_checkpoint_%d.pth.tar' % (epoch+1)))
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="PyTorch Omnipush Training")
    parser.add_argument("--workers", default=32, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("--batch_size", default=64, type=int, metavar="N", help="mini-batch size")
    parser.add_argument("--epochs", default=3000, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument(
        "--lr",
        "--learning-rate",
        default=2e-3,
        type=float,
        metavar="LR",
        help="initial learning rate for model parameters",
    )
    parser.add_argument("--seed", default=1, type=int, help="seed for initializing training")

    # to change depending on experiment:
    parser.add_argument("--job_name", default="anp", help="job save name for tensorboard")
    parser.add_argument("--data", default="../data/0,1,2_weights", help="data folder")
    parser.add_argument("--test_data", default=None, help="test data folder if different from data folder")
    parser.add_argument("--split", default="../split.json", help="train-test split")

    args = parser.parse_args()
    args.device = "cuda" if t.cuda.is_available() else "cpu"

    random.seed(args.seed)
    np.random.seed(args.seed)
    t.manual_seed(args.seed)
    t.cuda.manual_seed(args.seed)
    t.backends.cudnn.deterministic = True
    t.backends.cudnn.benchmark = False

    main()
