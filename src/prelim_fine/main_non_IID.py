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
from src.utils import aggregator_utils, data_utils, eval_utils

parser = argparse.ArgumentParser()


# training parameters
parser.add_argument('--data', type=str, default='MNIST')
parser.add_argument('--data-setting', type=str, default='none')
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--train-bsz', type=int, default=64)
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--aggregator', type=str, default='mean')
parser.add_argument('--local-updates', type=int, default=10)

# flip flop settings
parser.add_argument('--compression', type=str, default='none')
parser.add_argument('--compress-ratio', type=float, default=0.1)
parser.add_argument('--similarity', type=str, default='Euc')
parser.add_argument('--warmup-similarity', type=str, default='NA')


# local config
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


def train_iter(w_id, criterion, w_optimizer, w_model, global_model, train_data, train_data_iter, device):
    # print(w_id)

    iteration_loss = 0
    batch_cnt = 0

    print(w_id)

    for j in range(args.local_updates):
        # for batch_id, (data, target) in enumerate(train_data):
        try:
            data, target = next(train_data_iter)
        except StopIteration:
            train_data_iter = iter(train_data)
            data, target = next(train_data_iter)

        data, target = data.to(device), target.to(device)

        w_optimizer.zero_grad()
        output = w_model(data)
        loss = criterion(output, target)
        loss.backward()

        # print(batch_id, data.shape, loss.data.item())
        iteration_loss += loss.data.item()

        # if delta_ws == 0:
        #     delta_ws = w_optimizer.get_delta_w()
        # else:
        #     delta_ws += w_optimizer.get_delta_w()

        w_optimizer.step()
        batch_cnt += 1

    delta_ws = []
    for p_idx, param in enumerate(global_model.parameters()):
        delta_ws.append(param - list(w_model.parameters())[p_idx].data)

    return delta_ws, (iteration_loss / batch_cnt)


def train(workers, train_data_list, test_data):
    iterations = args.iterations

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
        # criterion = F.nll_loss
        criterion = torch.nn.CrossEntropyLoss()

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
        optimizer = MySGD(model.parameters(), lr=0.01)
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

    accum_g_list = [[] for i in workers] # accumulated g for all iterations
    accum_iter_g_list = [] # accumulated g for every 10 iterations
    cores_workers_list = []

    # number of overlap
    param_level = 2
    prev_loss = 10.0
    params_name = []

    from src.utils import compressor_helper, flipflop_utils
    compressor = compressor_helper.compressor_init(args)

    for name, param in global_model.named_parameters():
        params_name.append(name)
        
    for p_idx, param in enumerate(global_model.parameters()):
        for i in workers:
            list(model_list[i - 1].parameters())[p_idx].data = param.data + torch.zeros_like(param.data)

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

            w_g, w_loss = train_iter(i, criterion,
                                     optimizer_list[i - 1],
                                     model_list[i - 1], global_model,
                                     train_data_list[i - 1],
                                     train_data_iter_list[i - 1],
                                     device)

            g_list.append(w_g)

            iteration_loss += (w_loss / len(workers))

            # calculate accumulated list
            for p_idx, param in enumerate(global_model.parameters()):
                # accum_g_list[i - 1][p_idx] += delta_ws[p_idx]
                # print(params_name[p_idx])
                if iteration == 1:
                    accum_g_list[i - 1].append(w_g[p_idx])
                else:
                    accum_g_list[i - 1][p_idx] = w_g[p_idx]

        # flip flop choice
        global_g_list = []
        layer_cluster = []
        if args.aggregator == "mean":
            global_g_list = aggregator_utils.mean(args, workers, g_list)

        elif args.aggregator == "flipflop_layer":

            for p_idx, param in enumerate(global_model.parameters()):
                if "bias" in params_name[p_idx]:
                    layer_cluster.append(layer_cluster[-1])
                    continue

                temp_g_list = []
                for i in workers:
                    temp_g_list.append([accum_g_list[i - 1][p_idx], accum_g_list[i - 1][p_idx + 1]])

                similarity_matrix = eval_utils.check_similarity(temp_g_list, device, "Euc")

                layer_cluster.append(flipflop_utils.cluster_generation_client(similarity_matrix, 3))

            global_g_list = aggregator_utils.warmup_layer_flipflop(
                args, workers, g_list, params_name, layer_cluster, 3, device)

        elif args.aggregator == "flipflop_client":

            similarity_matrix = eval_utils.check_similarity(accum_g_list, device, "Euc")
            layer_cluster.append(flipflop_utils.cluster_generation_client(similarity_matrix, 3))

            global_g_list = aggregator_utils.warmup_layer_flipflop_client(
                args, workers, g_list, params_name, layer_cluster, 3, device)

        elif args.aggregator == "random_client":

            global_g_list = aggregator_utils.random_client_flipflop(
                args, workers, g_list, params_name, 3, device)


        # compression choice
        # --- compression & decompression ---
        for i in workers:
            # print(type(global_g_list[i - 1][0]))
            for p_idx, param in enumerate(global_model.parameters()):
                global_g_list[i - 1][p_idx], ctx_tmp = compressor.compress(
                    global_g_list[i - 1][p_idx], params_name[p_idx])

                global_g_list[i - 1][p_idx] = compressor.decompress(global_g_list[i - 1][p_idx], ctx_tmp)

        # aggregate gradients
        # cannot directly add them together - use partial
        global_g = []
        if args.aggregator == "mean":
            for p_idx, param in enumerate(global_model.parameters()):
                global_g_param = torch.zeros_like(param).to(device)
                for i in workers:
                    global_g_param += global_g_list[i - 1][p_idx]

                global_g.append(global_g_param)

        else:
            for p_idx, param in enumerate(global_model.parameters()):
                global_g_param = torch.zeros_like(param).to(device)
                for i in workers:
                    global_g_param += global_g_list[i - 1][p_idx]
                global_g_param /= 3

                global_g.append(global_g_param)

        for p_idx, param in enumerate(global_model.parameters()):

            param.data -= global_g[p_idx].data
            for i in workers:
                list(model_list[i - 1].parameters())[p_idx].data = param.data + torch.zeros_like(param.data)

        if (iteration + 1) % 1 == 0:
            print("Train Iter {}:\tLoss: {:.6f}".format(
                iteration, iteration_loss))
            eval_utils.check_sparsification_ratio(global_g_list)
            eval_utils.check_accuracy(test_data, global_model, device)
            sys.stdout.flush()

            # check param level cosine similarity
            if args.aggregator == "mean" and args.similarity == "Euc":
                # print("Iter", iteration, "Euc")
                similarity_matrix = eval_utils.check_similarity(accum_g_list, device, "Euc")
                layer_cluster = flipflop_utils.cluster_generation_client(similarity_matrix, 3)

                print("Iter", iteration, "Euc", layer_cluster)

                for p_idx, param in enumerate(global_model.parameters()):
                    if "bias" in params_name[p_idx]:
                        continue

                    temp_g_list = []
                    for i in workers:
                        temp_g_list.append([accum_g_list[i - 1][p_idx], accum_g_list[i - 1][p_idx + 1]])

                    similarity_matrix = eval_utils.check_similarity(temp_g_list, device, "Euc")

                    layer_cluster = flipflop_utils.cluster_generation_client(similarity_matrix, 3)
                    print("Layer", p_idx, "Euc", layer_cluster)
            else:
                for i in range(len(layer_cluster)):
                    print(i, layer_cluster[i])

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
    args.workers = 10
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

    import json

    data_f = open("prelim_fine/" + args.data_setting + ".json", )
    data_setting = json.load(data_f)

    train_data_list, test_loader = data_utils.get_loader_customize(args.data,
        data_transform, batch_size, args.workers, data_setting["data-setting"])

    data_f.close()

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
