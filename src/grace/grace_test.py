import torch
import argparse
import numpy as np
# import pandas as pd
import sys
import random

from cjltest.utils_model import MySGD
from cjltest.utils_data import get_data_transform

import torch.nn.functional as F
from torchvision import transforms

sys.path.append("..")
print(sys.path)
from src.models import alexnet, lr
from src.utils import aggregator_utils, data_utils, eval_utils, compressor_helper

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--chunks', type=int, default=5)
parser.add_argument('--train-bsz', type=int, default=512)
parser.add_argument('--aggregator', type=str, default='flipflop')
parser.add_argument('--dynamic', type=str, default='off')
parser.add_argument('--compression', type=str, default='terngrad')
parser.add_argument('--similarity', type=str, default='cosine')
parser.add_argument('--ratio', type=float, default=0.01)
parser.add_argument('--shuffle-worker', type=int, default=1)
parser.add_argument('--random-assign-list', type=int, default=1)
parser.add_argument('--local-config', type=int, default=1)

args = parser.parse_args()


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

    if args.model == "alexnet":
        global_model = alexnet.fasion_mnist_alexnet().to(device)
    else:
        global_model = lr.LROnMnist().to(device)

    # for p_idx, param in enumerate(global_model.parameters()):
    #     print(param.data)
    # global_model = MnistCNN().to(device)
    if args.model == "alexnet":
        criterion = F.nll_loss
    else:
        criterion = torch.nn.CrossEntropyLoss()

    model_list = []
    optimizer_list = []
    for i in workers:
        if args.model == "alexnet":
            model = alexnet.fasion_mnist_alexnet().to(device)
        else:
            model = lr.LROnMnist().to(device)
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

    compressor = compressor_helper.compressor_init(args)
    params_name = []
    for name, param in global_model.named_parameters():
        params_name.append(name)
    print(params_name)

    for iteration in range(1, iterations + 1):

        # print memory information first
        # cuda_t = torch.cuda.get_device_properties(0).total_memory
        # cuda_r = torch.cuda.memory_reserved(0)
        # cuda_a = torch.cuda.memory_allocated(0)
        # cuda_f = cuda_r - cuda_a  # free inside reserved
        # print("CUDA:", cuda_t, cuda_r, cuda_a, cuda_f)

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
                    # accum_g_list[i - 1][p_idx] += delta_ws[p_idx]
                    accum_g_list[i - 1][p_idx] = param.data - delta_ws[p_idx]

            iteration_loss += loss.data.item() / worker_number

        # flip flop choice
        global_g_list = []
        cores_worker_list = []

        if args.aggregator == "mean":
            global_g_list = aggregator_utils.mean(args, workers, g_list)
        elif args.aggregator == "flipflop":
            global_g_list, cores_worker_list = aggregator_utils.param_tensor_flip_flop(
                args, workers, g_list, param_level, device)
        # print(global_g_list[0][0])
        # client: compression
        for i in workers:
            # print(type(global_g_list[i - 1][0]))
            for p_idx, param in enumerate(global_model.parameters()):
                # print(type(global_g_list[i - 1][p_idx]))
                global_g_list[i - 1][p_idx], ctx_tmp = compressor.compress(
                    global_g_list[i - 1][p_idx], params_name[p_idx])

                global_g_list[i - 1][p_idx] = compressor.decompress(global_g_list[i - 1][p_idx], ctx_tmp)

        # send & receive

        # server: decompression
        # for i in workers:
        #     global_g_list[i - 1] = compressor.decompress(global_g_list[i - 1], ctx_g_list[i - 1])

        # aggregate gradients
        global_g = []
        if args.aggregator == "mean":
            for p_idx, param in enumerate(global_model.parameters()):
                global_g_param = torch.zeros_like(param).to(device)
                for i in workers:
                    global_g_param += global_g_list[i - 1][p_idx]

                global_g.append(global_g_param)

        elif args.aggregator == "flipflop":
            for p_idx, param in enumerate(global_model.parameters()):

                tensor_split_list = [0 for i in workers]

                for i in workers:
                    cores_i = cores_worker_list[i-1]
                    tensor_split_list[cores_i] = global_g_list[i-1][p_idx]

                global_g_param = torch.cat(tensor_split_list).to(device)
                global_g.append(global_g_param)

        for p_idx, param in enumerate(global_model.parameters()):

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
                eval_utils.check_similarity(accum_g_list, device, "cosine")
                print("Iter", iteration, "Euc")
                eval_utils.check_similarity(accum_g_list, device, "euc")


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

    # train_data_list, test_loader = data_utils.get_loader(batch_size, 2, 10)
    data_transform = 0
    if args.model == "alexnet":
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])
    else:
        data_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.ToTensor(),
        ])

    train_data_list, test_loader = data_utils.get_loader(data_transform, batch_size, worker_number, 10)

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
