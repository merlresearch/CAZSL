# Copyright (C) 2020-2023 Mitsubishi Electric Research Laboratories (MERL)
#
# SPDX-License-Identifier: AGPL-3.0-or-later


"""
Run this script with relevant arguments for baseline models (FCN and FCN + CC), and CAZSL models (FCN + CM, FCN + CM + L2Reg, FCN + CM + NeuralReg),
Context is either indicator context or visual context.
"""

import argparse
import json
import os
import random
import sys
import warnings

import numpy as np
import torch
from tensorboardX import SummaryWriter
from torch import nn
from torch.distributions.normal import Normal
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm as Tqdm

from preprocess import build_encoding, omnipush, omnipush_collate_fn

global_step = 0

# helper function to add context to original dataset
class ObjectsDataset(Dataset):
    def __init__(self, context_mat, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.context_mat = context_mat

    def __len__(self):
        return len(self.context_mat)

    def __getitem__(self, idx):
        X, y = self.orig_dataset.__getitem__(idx)
        return X, y, self.context_mat[idx, :]


# helper function to concatenate context to original dataset
class ConcatDataset(Dataset):
    def __init__(self, context_mat, orig_dataset, transform=None):
        self.orig_dataset = orig_dataset
        self.context_mat = context_mat

    def __len__(self):
        return len(self.context_mat)

    def __getitem__(self, idx):
        X, y = self.orig_dataset.__getitem__(idx)
        repeat_index_context_mat = self.context_mat[idx, :].repeat(250, 1)  # each object has 250 samples
        X = torch.cat([X, repeat_index_context_mat], axis=1)
        return X, y


# defining baseline FCNs and CAZSL models
class NeuralNet(nn.Module):
    def __init__(self, context_type="visual", concat=False, reg_type="l2", len_context=32):
        super(NeuralNet, self).__init__()
        self.context_type = context_type
        self.concat = concat
        self.reg_type = reg_type
        self.len_context = len_context

        # network parameters for FCN
        if context_type == "indicator" and concat:
            self.fc1 = nn.Linear(3 + len_context, 256)
        else:
            self.fc1 = nn.Linear(3, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.fc4 = nn.Linear(256, 6)
        self.relu = nn.ReLU()

        # network parameters for emedding context mask (and concatenated visual context) and regularizer
        if context_type == "visual":
            # embedding context for mask
            self.cnnlayer = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(4, 4, kernel_size=3, stride=1, padding=1),
                nn.BatchNorm2d(4),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.cfc1 = torch.nn.Linear(int(4 * (len_context / 4) * (len_context / 4)), 256)
            # embedding context for neural regularization
            self.avg_pool = nn.AvgPool2d(8, stride=4)
            self.dist_cfc1 = nn.Sequential(
                nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1), self.relu, self.avg_pool
            )
            self.dist_cfc2 = nn.Linear(int(4 * ((len_context - 8) / 4 + 1) * ((len_context - 8) / 4 + 1)), 10)
        elif context_type == "indicator":
            # embedding contect for mask
            self.mfc1 = nn.Linear(len_context, 256)
            # embedding context for neural regularization
            self.dist_cfc = nn.Sequential(nn.Linear(len_context, 10), self.relu, nn.Linear(10, 5))

    def forward(self, x, context=None):

        out_c = 0  # placeholder
        if self.context_type == "nocontext":
            out = self.relu(self.fc1(x))
            out = self.relu(self.fc2(out))
            out_mask = torch.ones(out.shape)  # placeholder mask
        elif self.context_type == "indicator":
            if self.concat:
                out = self.relu(self.fc1(x))
                out = self.relu(self.fc2(out))
                out_mask = torch.ones(out.shape)  # placeholder mask
            else:
                out_c = self.dist_cfc(context)
                out_mask = self.relu(self.mfc1(context.repeat(1, 250).view(-1, self.len_context)))
                out = self.relu(self.fc1(x))
                out = self.relu(self.fc2(out * out_mask))
        else:
            if self.concat:
                out_mask = self.cnnlayer(context).view(-1, int(4 * (self.len_context / 4) * (self.len_context / 4)))
                out_mask = out_mask.repeat(1, 250).view(-1, int(4 * (self.len_context / 4) * (self.len_context / 4)))
                out_mask = self.cfc1(self.relu(out_mask))
                out = self.relu(self.fc1(x) + out_mask)  # same effect as concatenating
                out = self.relu(self.fc2(out))
            else:
                out_mask = self.cnnlayer(context).view(-1, int(4 * (self.len_context / 4) * (self.len_context / 4)))
                out_mask = out_mask.repeat(1, 250).view(-1, int(4 * (self.len_context / 4) * (self.len_context / 4)))
                out_mask = self.cfc1(self.relu(out_mask))
                out_mask = self.relu(out_mask)
                out_c = self.dist_cfc1(context).view(
                    -1, int(4 * ((self.len_context - 8) / 4 + 1) * ((self.len_context - 8) / 4 + 1))
                )
                out_c = self.dist_cfc2(out_c)
                out = self.relu(self.fc1(x))
                out = self.relu(self.fc2(out * out_mask))

        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out, out_mask, out_c


def evaluate(model, loader, name):
    global global_step
    full_loss = 0.0
    count = 0
    if name == "train":
        model.train()
    else:
        model.eval()

    for i, data in enumerate(loader):
        # extract data
        if (model.context_type == "nocontext") or (model.context_type == "indicator" and model.concat):
            X, y = data
        else:
            X, y, context = data

        # split into equal parts
        batch_size_each = int(X.shape[0] / 2)

        X1 = X[:batch_size_each].to(args.device)
        y1 = y[:batch_size_each].to(args.device)

        X2 = X[batch_size_each:].to(args.device)
        y2 = y[batch_size_each:].to(args.device)

        X1 = X1.reshape(-1, X1.shape[2])
        y1 = y1.reshape(-1, y1.shape[2])

        X2 = X2.reshape(-1, X2.shape[2])
        y2 = y2.reshape(-1, y2.shape[2])

        # run model
        if (model.context_type == "nocontext") or (model.context_type == "indicator" and model.concat):
            pred1, mask1, embedc1 = model(X1)
            pred2, mask2, embedc2 = model(X2)
        else:
            context1 = context[:batch_size_each].to(args.device)
            context2 = context[batch_size_each:].to(args.device)
            pred1, mask1, embedc1 = model(X1, context1)
            pred2, mask2, embedc2 = model(X2, context2)

        # evaluate fit
        mu1, log_std1 = torch.chunk(pred1, 2, dim=-1)
        mu2, log_std2 = torch.chunk(pred2, 2, dim=-1)

        # mean squared error
        mse = (nn.MSELoss()(y1, mu1) + nn.MSELoss()(y2, mu2)) / 2

        # negative loglikelihood
        log_p1 = Normal(loc=mu1, scale=torch.exp(log_std1)).log_prob(y1)
        log_p2 = Normal(loc=mu2, scale=torch.exp(log_std2)).log_prob(y2)
        log_p1 = torch.mean(log_p1)
        log_p2 = torch.mean(log_p2)
        loss_p = (-log_p1 - log_p2) / 2

        # siamese loss
        dist_encode = torch.zeros([batch_size_each])  # placeholder
        if model.reg_type == "l2":
            if model.context_type == "visual":
                dist_encode = torch.sqrt(torch.sum((context1 - context2) ** 2, dim=(1, 2, 3)))
            else:
                dist_encode = torch.sqrt(torch.sum((context1 - context2) ** 2, dim=1))
        elif model.reg_type == "neural":
            dist_encode = torch.sum(embedc1 * embedc2, dim=1)

        mask_diff250 = torch.sqrt(torch.sum((mask1 - mask2) ** 2, dim=1))
        mask_diff = mask_diff250[range(0, batch_size_each * 250, 250)]
        loss = loss_p + args.lambda1 * torch.mean(
            (mask_diff.to(args.device) - args.lambda2 * dist_encode.to(args.device)) ** 2
        )

        if name == "train":
            opt.zero_grad()
            loss.backward()
            opt.step()

        full_loss += loss.item()
        count += 1

        writer.add_scalar(name + "/loss", loss, global_step)
        writer.add_scalar(name + "/RMSE", torch.sqrt(mse), global_step)
        writer.add_scalar(name + "/nlogP", loss_p * y1.shape[-1], global_step)
        writer.add_scalar(name + "/lstd", torch.mean((log_std1 + log_std2) / 2), global_step)
        global_step += 1

    print("Epoch # {}: {} Loss: {}".format(epoch, name, full_loss))


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
    parser.add_argument("--job_name", default="fcn_cm_l2reg_visual", help="job save name for tensorboard")
    parser.add_argument("--data", default="../data/0,1,2_weights", help="data folder")
    parser.add_argument("--test_data", default=None, help="test data folder if different from data folder")
    parser.add_argument("--split", default="../split.json", help="train-test split")
    parser.add_argument(
        "--context_type",
        default="visual",
        choices=["nocontext", "indicator", "visual"],
        help="type of context for learning",
    )
    parser.add_argument("--image_dir", default="../data/top-down_view32", help="directory of images for visual context")
    parser.add_argument(
        "--concat",
        default=False,
        const=True,
        action="store_const",
        help="concatenate context, instead of CAZSL masking",
    )
    parser.add_argument(
        "--reg_type", default="l2", choices=["noreg", "l2", "neural"], help="type of CAZSL context regularization"
    )
    parser.add_argument("--lambda1", default=0.01, type=float, help="CAZSL regularizaiton coefficient")
    parser.add_argument("--lambda2", default=0.01, type=float, help="CAZSL inner regularization coefficient")

    args = parser.parse_args()
    args.device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.context_type == "nocontext":
        if args.lambda1 != 0:
            args.lambda1 = 0
            warnings.warn("lambda1 will be set to 0 since model is not CAZSL")
    else:
        if args.concat:
            if args.lambda1 != 0:
                args.lambda1 = 0
                warnings.warn("lambda1 will be set to 0 since model is not CAZSL")
        else:
            if args.reg_type == "noreg":
                if args.lambda1 != 0:
                    args.lambda1 = 0
                    warnings.warn("lambda1 will be set to 0 since specified CAZSL model uses no regularization")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # object attributes according to naming conventions
    char_id = ["a", "b", "c", "B", "C"]
    num_id = [1, 2, 3, 4]

    # create model
    if args.context_type == "indicator":
        len_context = 4 * (len(char_id) + len(num_id))  # 9 options per side
    elif args.context_type == "visual":
        len_context = 32  # 32*32 image
    else:
        len_context = 0

    model = NeuralNet(
        context_type=args.context_type, concat=args.concat, reg_type=args.reg_type, len_context=len_context
    ).to(args.device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    # load training data
    if args.context_type == "visual":
        train_names, train_X, train_y, train_images = omnipush(
            args.data,
            train=True,
            split_file=args.split,
            checked_pattern="X_3_norm",
            normalize=False,
            get_image=True,
            image_dir=args.image_dir,
        )
    else:
        train_names, train_X, train_y = omnipush(
            args.data,
            train=True,
            split_file=args.split,
            checked_pattern="X_3_norm",
            normalize=False,
            get_image=False,
            image_dir=args.image_dir,
        )

    print(" Shape of training data: ", train_X.shape)

    # load test data
    if args.test_data == None:
        if args.context_type == "visual":
            test_names, test_X, test_y, test_images = omnipush(
                args.data,
                train=False,
                split_file=args.split,
                checked_pattern="X_3_norm",
                normalize=False,
                get_image=True,
                image_dir=args.image_dir,
            )
        else:
            test_names, test_X, test_y = omnipush(
                args.data,
                train=False,
                split_file=args.split,
                checked_pattern="X_3_norm",
                normalize=False,
                get_image=False,
                image_dir=args.image_dir,
            )
    else:
        if args.context_type == "visual":
            test_names, test_X, test_y, test_images = omnipush(
                args.test_data,
                train=False,
                split_file=args.split,
                checked_pattern="X_3_norm",
                normalize=False,
                return_all=True,
                get_image=True,
                image_dir=args.image_dir,
            )
        else:
            test_names, test_X, test_y = omnipush(
                args.test_data,
                train=False,
                split_file=args.split,
                checked_pattern="X_3_norm",
                normalize=False,
                return_all=True,
                get_image=False,
                image_dir=args.image_dir,
            )

    print(" Shape of testing data: ", test_X.shape)

    train_dataset = torch.utils.data.TensorDataset(train_X, train_y)
    test_dataset = torch.utils.data.TensorDataset(test_X, test_y)

    # create data loader
    if args.context_type == "nocontext":
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers)
    else:
        # update data loader to include context
        if args.context_type == "indicator":
            # context is indicator of object descriptors
            train_context_mat = np.zeros((len(train_names), len_context))
            for row in range(len(train_context_mat)):
                train_context_mat[row, :] = build_encoding(train_names[row][0])
            test_context_mat = np.zeros((len(test_names), len_context))
            for row in range(len(test_context_mat)):
                test_context_mat[row, :] = build_encoding(test_names[row][0])
        else:
            # context is images
            train_context_mat = train_images
            test_context_mat = test_images
        train_context_mat = torch.Tensor(train_context_mat)
        test_context_mat = torch.Tensor(test_context_mat)
        if args.context_type == "indicator" and args.concat:
            train_context_dataset = ConcatDataset(train_context_mat, train_dataset)
            test_context_dataset = ConcatDataset(test_context_mat, test_dataset)
        else:
            train_context_dataset = ObjectsDataset(train_context_mat, train_dataset)
            test_context_dataset = ObjectsDataset(test_context_mat, test_dataset)
        train_loader = DataLoader(
            train_context_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )
        test_loader = DataLoader(
            test_context_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers
        )

    # training and testing
    writer = SummaryWriter(comment=args.job_name)
    for epoch in Tqdm(range(args.epochs)):
        evaluate(model, train_loader, "train")
        evaluate(model, test_loader, "test")
