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
from src.utils import aggregator_utils, data_utils, eval_utils, compressor_helper, flipflop_utils

parser = argparse.ArgumentParser()

# training parameters
parser.add_argument('--model', type=str, default='alexnet')
parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--train-bsz', type=int, default=512)
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--aggregator', type=str, default='flipflop')

# flip flop settings
parser.add_argument('--warmup-iterations', type=int, default=100)
parser.add_argument('--overlap-level', type=int, default=2)
parser.add_argument('--partition-size', type=int, default=0)
parser.add_argument('--warmup-similarity', type=str, default='none')
parser.add_argument('--compression', type=str, default='none')
parser.add_argument('--similarity', type=str, default='none')

# local config
parser.add_argument('--local-config', type=int, default=1)

args = parser.parse_args()


def train_iter(w_id, criterion, w_optimizer, w_model, global_model, w_dataloader, device):
    # print(w_id)

    iteration_loss = 0
    batch_cnt = 0
    for batch_id, (data, target) in enumerate(w_dataloader):

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

    # --- initialize training parameter ---
    iterations = args.iterations
    warmup_iterations = args.warmup_iterations
    overlap_level = args.overlap_level

    # --- initialize device and seed ---
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    print(device)

    torch.manual_seed(10)
    if device == "cuda":
        torch.cuda.manual_seed(10)
        print("init cuda seed")

    # --- initialize training data on each worker ----
    train_data_iter_list = []
    for i in workers:
        train_data_iter_list.append(iter(train_data_list[i - 1]))

    # --- initialize model on each worker ---
    if args.model == "alexnet":
        global_model = alexnet.fasion_mnist_alexnet().to(device)
        criterion = F.nll_loss
    else:
        global_model = lr.LROnMnist().to(device)
        criterion = torch.nn.CrossEntropyLoss()

    model_list = []
    optimizer_list = []
    for i in workers:
        if args.model == "alexnet":
            model = alexnet.fasion_mnist_alexnet().to(device)
        else:
            model = lr.LROnMnist().to(device)

        model_list.append(model)
        optimizer = MySGD(model.parameters(), lr=0.01)
        optimizer_list.append(optimizer)
        model.train()

    # --- initialize compressor ---
    compressor = compressor_helper.compressor_init(args)
    params_name = []
    for name, param in global_model.named_parameters():
        params_name.append(name)
    print(params_name)
    # for name, param in model_list[4].named_parameters():
    #     print(param)
    # print(params_name)

    # --- initialize warmup parameters ---
    # warmup_g_list = []
    average_similarity_list = []
    layer_cluster = []
    for g_idx, g_param in enumerate(global_model.parameters()):
        # if "bias" in params_name[g_idx]:
        #     continue

        average_similarity_list.append([[0 for x in range(worker_number)] for y in range(worker_number)])
    # print(average_similarity_list)

    for p_idx, param in enumerate(global_model.parameters()):
        for i in workers:
            list(model_list[i - 1].parameters())[p_idx].data = param.data + torch.zeros_like(param.data)

    # --- flip-flop training ---
    print("start flip-flop")
    for iteration in range(1, iterations + 1):

        # gradient list
        g_list = []
        iteration_loss = 0

        # distributed training
        for i in workers:

            w_g, w_loss = train_iter(i, criterion,
                       optimizer_list[i - 1],
                       model_list[i - 1], global_model,
                       train_data_list[i - 1], device)

            g_list.append(w_g)
            iteration_loss += (w_loss / len(workers))

        if iteration == warmup_iterations:
            print("average", average_similarity_list)
            layer_cluster = flipflop_utils.cluster_generation(
                g_list, average_similarity_list, params_name,
                args.warmup_similarity, overlap_level, device)

            print(layer_cluster)

        global_g_list = []
        cores_worker_list = []

        # --- aggregator ---
        if (iteration <= warmup_iterations) or (args.aggregator == "mean"):
            global_g_list = aggregator_utils.mean(args, workers, g_list)
        elif args.aggregator == "flipflop":
            global_g_list = aggregator_utils.warmup_layer_flipflop(
                args, workers, g_list, params_name, layer_cluster, overlap_level, device)
        elif args.aggregator == "flipflop_client":
            global_g_list = aggregator_utils.warmup_layer_flipflop_client(
                args, workers, g_list, params_name, layer_cluster, overlap_level, device)

        # --- compression & decompression ---
        for i in workers:
            # print(type(global_g_list[i - 1][0]))
            for p_idx, param in enumerate(global_model.parameters()):
                global_g_list[i - 1][p_idx], ctx_tmp = compressor.compress(
                    global_g_list[i - 1][p_idx], params_name[p_idx])

                global_g_list[i - 1][p_idx] = compressor.decompress(global_g_list[i - 1][p_idx], ctx_tmp)

        # --- warmup generation ---
        if (1 < iteration) and (iteration <= warmup_iterations):
            for g_idx, g_param in enumerate(global_g_list[0]):
                # print(params_name[g_idx])
                # print(params_name[g_idx], "bias" in params_name[g_idx])
                if "bias" in params_name[g_idx]:
                    continue

                tensor_list = []
                for i in range(worker_number):
                    tensor_list.append(global_g_list[i][g_idx])
                # print(tensor_list)
                similarity_matrix = eval_utils.check_similarity(tensor_list, device, 'ecu')

                # if "fc3" in params_name[g_idx]:
                #     print(params_name[g_idx], similarity_matrix)

                for x in range(worker_number):
                    for y in range(worker_number):
                        average_similarity_list[g_idx][x][y] += (similarity_matrix[x][y])

            # print(eval_utils.check_similarity(warmup_g_list, device, 'ecu'))

        # --- aggregate gradients ---
        global_g = []
        if (iteration <= warmup_iterations) or (args.aggregator == "mean"):
            for p_idx, param in enumerate(global_model.parameters()):
                global_g_param = torch.zeros_like(param).to(device)
                for i in workers:
                    global_g_param += global_g_list[i - 1][p_idx]

                global_g.append(global_g_param)

        elif args.aggregator == "flipflop":
            for p_idx, param in enumerate(global_model.parameters()):
                global_g_param = torch.zeros_like(param).to(device)
                for i in workers:
                    global_g_param += global_g_list[i - 1][p_idx]
                global_g_param /= overlap_level

                global_g.append(global_g_param)

        # --- update parameters ---
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
            if args.similarity == "cosine":
                print("Iter", iteration, "Cosine")
                print("Iter", iteration, "Euc")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)

    print(args)
    batch_size = args.train_bsz
    worker_number = args.workers

    # local configuration
    if args.local_config == 1:
        batch_size = 10
        worker_number = 2
        # worker_number = 10

    # create a list of workers
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
    train_data_list, test_loader = data_utils.get_loader_customize(
        data_transform, batch_size, worker_number,
        [
            [100, 100, 100, 100, 5, 5, 5, 5, 5, 5],
            [100, 100, 100, 100, 5, 5, 5, 5, 5, 5],
            [100, 100, 100, 100, 5, 5, 5, 5, 5, 5],
            [100, 100, 100, 100, 5, 5, 5, 5, 5, 5],
            [100, 100, 100, 100, 5, 5, 5, 5, 5, 5],
            [100, 100, 100, 100, 5, 5, 5, 5, 5, 5],
            [100, 100, 100, 100, 5, 5, 5, 5, 5, 5],
            [100, 100, 100, 100, 5, 5, 5, 5, 5, 5],
            [5, 5, 5, 5, 100, 100, 100, 100, 100, 100],
            [5, 5, 5, 5, 100, 100, 100, 100, 100, 100],
        ])

    print("finish")

    train(workers, train_data_list, test_loader)


