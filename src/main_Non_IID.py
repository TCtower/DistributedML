import os
import torch
import torchvision
import argparse
import numpy as np
# import pandas as pd
import math
import sys

import torch.optim as optim
from cjltest.utils_model import MySGD
from cjltest.divide_data import partition_dataset, select_dataset
from cjltest.models import MnistCNN, AlexNetForCIFAR, LeNetForMNIST

from PIL import Image
import torch.nn.functional as F

from torchvision import transforms, datasets
from torch.utils.data import Dataset, DataLoader

import alexnet
import random

import data_utils
import aggregator_utils
import eval_utils
import sklearn.metrics.pairwise as smp


parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--chunks', type=int, default=5)
parser.add_argument('--train-bsz', type=int, default=512)
parser.add_argument('--aggregator', type=str, default='mean')
parser.add_argument('--dynamic', type=str, default='off')
parser.add_argument('--compression', type=str, default='topk')
parser.add_argument('--similarity', type=str, default='cosine')
parser.add_argument('--ratio', type=float, default=0.01)
parser.add_argument('--shuffle-worker', type=int, default=1)
parser.add_argument('--random-assign-list', type=int, default=1)
parser.add_argument('--local-config', type=int, default=1)

args = parser.parse_args()


def param_adjust(args, param_level, increase_overlap):

    if increase_overlap:
        # iteration_loss > prev_loss
        param_level += 1
        if param_level > worker_number:
            param_level = worker_number
    else:
        # iteration_loss < prev_loss
        param_level -= 1
        if param_level < 1:
            param_level = 1
    return param_level


# chunk level distributed list
# def chunk_adjust(args, param_level, distribution_list, increase_overlap):
#     worker_number = len(distribution_list)
#
#     # transfer distribution list into chunk worker list
#     chunk_worker_list = [[] for i in range(args.chunks)]
#     for i in range(worker_number):
#         for j in range(len(distribution_list[i])):
#             core = distribution_list[i][j]
#             chunk_worker_list[core].append(i)
#
#
#     if param_level == 1:
#         if increase_overlap:
#             # increase overlap
#             # add one more copy for each worker
#             param_level += 1
#
#             for i in range(worker_number):
#                 core_i = distribution_list[i][0]
#                 core_j = (core_i + 1) % worker_number
#                 # add chunk
#                 distribution_list[i].append(core_j)
#         else:
#             # decrease overlap
#             # do nothing
#             param_level = 1
#
#     elif param_level == 2:
#         # calculate the similarity
#         for chunk_no in range(len(chunk_worker_list)):
#             chunk_g_list = []
#             for i in range(worker_number):
#                 worker_g_list = []
#                 for
#
#         if increase_overlap:
#
#
#         else:
#             # do nothing
#
#
#     return param_level


def train(workers, train_data_list, test_data):
    iterations = 1000

    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    print(device)

    # get some random training images
    # plot_random_figure(test_data)
    # checks
    print("got")

    train_data_iter_list = []
    for i in workers:
        train_data_iter_list.append(iter(train_data_list[i - 1]))


    torch.manual_seed(10)
    if device == "cuda":
        torch.cuda.manual_seed(10)
        print("init cuda seed")

    global_model = alexnet.fasion_mnist_alexnet().to(device)

    # for p_idx, param in enumerate(global_model.parameters()):
    #     print(param.data)
    # global_model = MnistCNN().to(device)
    criterion = F.nll_loss
    model_list = []
    optimizer_list = []
    for i in workers:
        model = alexnet.fasion_mnist_alexnet().to(device)
        # model = MnistCNN().to(device)
        model_list.append(model)
        optimizer = MySGD(model.parameters(), lr=0.1)
        optimizer_list.append(optimizer)
        model.train()

    print("start training")
    # iteration_epoch = int(len())
    # model_list[i-1].train()

    # init assign list
    assign_list = []
    for g_idx, g_param in enumerate(global_model.parameters()):
        assign_list.append(torch.randint(worker_number, g_param.shape).to(device))

    # init distribution list
    distribution_list = []
    for i in range(len(workers)):
        distribution_list.append([i])
    # print(assign_list)

    g_remain_list = []
    for i in workers:
        g_remain = []
        for g_idx, g_param in enumerate(global_model.parameters()):
            g_remain_param = torch.zeros_like(g_param).to(device)
            g_remain.append(g_remain_param)

        g_remain_list.append(g_remain)

    accum_g_list = [] # accumulated g for all iterations
    accum_iter_g_list = [] # accumulated g for every 10 iterations
    cores_workers_list = []

    # number of overlap
    param_level = 2
    prev_loss = 10.0



    for iteration in range(1, iterations + 1):

        # print memory information first
        cuda_t = torch.cuda.get_device_properties(0).total_memory
        cuda_r = torch.cuda.memory_reserved(0)
        cuda_a = torch.cuda.memory_allocated(0)
        cuda_f = cuda_r - cuda_a  # free inside reserved
        print("CUDA:", cuda_t, cuda_r, cuda_a, cuda_f)

        # gradient list
        g_list = []
        iteration_loss = 0

        # distributed training
        for i in workers:

            try:
                data, target = next(train_data_iter_list[i - 1])
            except StopIteration:
                train_data_iter_list[i - 1] = iter(train_data_list[i - 1])
                data, target = next(train_data_iter_list[i - 1])
            data, target = data.to(device), target.to(device)

            # strange
            # target = target.type(torch.LongTensor)
            # data, target = data.to(device), target.to(device)
            optimizer_list[i - 1].zero_grad()
            output = model_list[i - 1](data)
            loss = criterion(output, target)
            loss.backward()

            # optimizer.step()

            delta_ws = optimizer_list[i - 1].get_delta_w()
            g_list.append(delta_ws)

            # calculate accumulated list
            if iteration == 1:
                accum_g_list.append(delta_ws)
            else:
                for p_idx, param in enumerate(global_model.parameters()):
                    accum_g_list[i - 1][p_idx] += delta_ws[p_idx]

            iteration_loss += loss.data.item() / worker_number

        # flip flop choice
        global_g_list = []
        if args.aggregator == "mean":
            global_g_list = aggregator_utils.mean(args, workers, g_list)
        elif args.aggregator == "layer":
            global_g_list = aggregator_utils.layer_flip_flop(args, workers, g_list)
        elif args.aggregator == "param":
            global_g_list, cores_workers_list = aggregator_utils.param_flip_flop(args, workers, g_list, assign_list, param_level, device)
        elif args.aggregator == "chunk":
            # chunk level flip flop
            global_g_list, cores_workers_list = aggregator_utils.chunk_flip_flop(args, workers, g_list, assign_list,
                                                                                 distribution_list, device)

        # compression choice
        if args.compression == 'topk':
            ratio = args.ratio

            if args.aggregator == "param":
                ratio = (ratio / worker_number) * param_level

            for i in workers:
                g_remain, g_upload = aggregator_utils.get_topk(args, g_remain_list[i-1], global_g_list[i-1], ratio, device)
                g_remain_list[i-1] = g_remain
                global_g_list[i-1] = g_upload

        # aggregate gradients
        # cannot directly add them together - use partial
        global_g = []
        for p_idx, param in enumerate(global_model.parameters()):
            global_g_param = torch.zeros_like(param).to(device)
            for i in workers:
                global_g_param += global_g_list[i-1][p_idx]

            global_g.append(global_g_param)

        for p_idx, param in enumerate(global_model.parameters()):
            # global_g
            # if not (("conv2" in name) or ("conv4" in name) or ("fc3" in name)):
            #     continue

            param.data -= global_g[p_idx].data
            for i in workers:
                list(model_list[i - 1].parameters())[p_idx].data = param.data + torch.zeros_like(param.data)

        if (iteration + 1) % 10 == 0:
            print("Train Iter {}:\tLoss: {:.6f}".format(
                iteration, iteration_loss))
            eval_utils.check_sparsification_ratio(global_g_list)
            eval_utils.check_accuracy(test_data, global_model, device)
            sys.stdout.flush()

            # check param level cosine similarity
            if args.similarity == "cosine":
                print("Iter", iteration, "Cosine")
                # check_cosine_similarity(accum_g_list, device)

                eval_utils.check_cosine_similarity_chunk(global_g_list, assign_list, cores_workers_list, device)

            # dynamic flip flop
            # if args.dynamic == "on":
            #     if args.aggregator == "param":
            #         param_level = param_adjust(args, param_level, iteration_loss > prev_loss)
            #
            #     elif args.aggregator == "chunk":
            #         param_level = chunk_adjust(args, param_level,
            #                                    distribution_list, iteration_loss > prev_loss)
            #
            #     print(prev_loss, iteration_loss, param_level)
            #     sys.stdout.flush()
            #     prev_loss = iteration_loss


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)

    print(args)
    batch_size = args.train_bsz

    # local configuration
    if args.local_config == 1:
        batch_size = 10

    # create a list of workers
    worker_number = args.workers
    workers = [v + 1 for v in range(worker_number)]

    train_data_list, test_loader = data_utils.get_loader(batch_size, worker_number, 2)
    print("finish")
    # # train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    # test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=True)
    #
    # # print("finish loading data")
    #
    train(workers, train_data_list, test_loader)

    # t = torch.randint(3, (1, 5))
    # print(t)
    # print(t[(t == 1)])

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
