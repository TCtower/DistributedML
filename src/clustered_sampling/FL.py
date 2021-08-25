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
parser.add_argument('--data', type=str, default='MNIST_shard')
parser.add_argument('--train-bsz', type=int, default=64)
parser.add_argument('--iterations', type=int, default=100)
parser.add_argument('--aggregator', type=str, default='flipflop')
parser.add_argument('--local-updates', type=int, default=1)

# flip flop settings
parser.add_argument('--warmup-iterations', type=int, default=100)
parser.add_argument('--n-sampled', type=int, default=5)
parser.add_argument('--partition-size', type=int, default=0)
parser.add_argument('--warmup-similarity', type=str, default='none')
parser.add_argument('--compression', type=str, default='none')
parser.add_argument('--compress-ratio', type=float, default=0.01)
parser.add_argument('--similarity', type=str, default='none')

# local config
parser.add_argument('--local-config', type=int, default=1)

args = parser.parse_args()


def train_iter(w_id, criterion, w_optimizer, w_model, global_model, w_data, device):
    # print(w_id)

    iteration_loss = 0
    batch_cnt = 0

    # do local updates here
    for j in range(args.local_updates):

        data, target = next(iter(w_data))
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


def train(train_data_list, test_data):
    # default 100 works
    workers = [i for i in range(0, 100)]
    n_sampled = args.n_sampled # number of sampled workers
    K = len(train_data_list)  # number of clients
    n_samples = np.array([len(db.dataset) for db in train_data_list])
    weights = n_samples / np.sum(n_samples)
    print(weights)
    # --- initialize training parameter ---
    iterations = args.iterations

    # --- initialize device and seed ---
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    print(torch.__version__)
    print(device)

    torch.manual_seed(10)
    if device == "cuda":
        torch.cuda.manual_seed(10)
        print("init cuda seed")

    # --- initialize training data on each worker ----
    # train_data_iter_list = []
    # for i in workers:
    #     train_data_iter_list.append(iter(train_data_list[i - 1]))

    # --- initialize model on each worker ---
    import create_model
    global_model = create_model.NN(50, 10).to(device)
    criterion = F.nll_loss
    import torch.nn as nn
    criterion = nn.CrossEntropyLoss()

    model_list = []
    optimizer_list = []

    for i in range(0, n_sampled):
        model = create_model.NN(50, 10).to(device)

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

    # --- initialize warmup parameters ---
    # warmup_g_list = []
    similarity_list = [[0 for x in range(worker_number)] for y in range(worker_number)]
    from scipy.cluster.hierarchy import linkage
    linkage_matrix = linkage(np.array(similarity_list), "ward")
    last_gradient_list = [[] for x in range(worker_number)]
    for g_idx, g_param in enumerate(global_model.parameters()):
        for i in range(worker_number):
            last_gradient_list[i].append(torch.zeros_like(g_param))

    for p_idx, param in enumerate(global_model.parameters()):
        for i in range(0, n_sampled):
            list(model_list[i - 1].parameters())[p_idx].data = param.data + torch.zeros_like(param.data)

    # --- flip-flop training ---
    print("start flip-flop")
    for iteration in range(1, iterations + 1):

        # gradient list
        g_list = []
        iteration_loss = 0

        import clustering
        # print(similarity_list)

        sample_workers =[]
        if args.aggregator == "random":
            sample_workers = aggregator_utils.random_client(n_sampled)

        elif args.aggregator == "flipflop":
            if iteration > 50:
                n_sampled = 3

            if iteration > 75:
                n_sampled = 2

            if iteration > 100:
                n_sampled = 1

            client_cluster = flipflop_utils.cluster_generation_client(
                similarity_list, n_sampled)

            # print(client_cluster)

            sample_workers = aggregator_utils.warmup_client_flipflop(
                client_cluster, n_sampled)


        elif args.aggregator == "cluster":
            distri_clusters = clustering.get_clusters_with_alg2(
                linkage_matrix, n_sampled, weights
            )
            sample_workers = clustering.sample_clients(distri_clusters)

        print(sample_workers)
        # distributed training
        for i in range(n_sampled):
            k = sample_workers[i]

            w_g, w_loss = train_iter(i, criterion,
                                     optimizer_list[i],
                                     model_list[i], global_model,
                                     train_data_list[k], device)

            g_list.append(w_g)
            iteration_loss += (w_loss / n_sampled)

            # print(layer_cluster)

        global_g_list = []
        cores_worker_list = []

        # --- aggregator ---
        global_g_list = aggregator_utils.mean(args, workers, g_list)

        # --- compression & decompression ---
        for i in range(n_sampled):
            # print(type(global_g_list[i - 1][0]))
            for p_idx, param in enumerate(global_model.parameters()):
                global_g_list[i][p_idx], ctx_tmp = compressor.compress(
                    global_g_list[i][p_idx], params_name[p_idx])

                global_g_list[i][p_idx] = compressor.decompress(global_g_list[i][p_idx], ctx_tmp)

        # --- update last gradient ---

        for g_idx, g_param in enumerate(global_g_list[0]):

            if not ("bias" in params_name[g_idx]):
                for i in range(n_sampled):
                    k = sample_workers[i]
                    last_gradient_list[k] = global_g_list[i]

        similarity_matrix = eval_utils.check_similarity(last_gradient_list, device, 'ecu')

        for x in range(worker_number):
            for y in range(worker_number):
                similarity_list[x][y] = (similarity_matrix[x][y])
        # print(similarity_list)
        linkage_matrix = linkage(np.array(similarity_list), "ward")
        # print(eval_utils.check_similarity(warmup_g_list, device, 'ecu'))

        # --- aggregate gradients ---
        global_g = []
        for p_idx, param in enumerate(global_model.parameters()):
            global_g_param = torch.zeros_like(param).to(device)
            for i in range(n_sampled):
                global_g_param += global_g_list[i][p_idx]

            global_g.append(global_g_param)

        # --- update parameters ---
        for p_idx, param in enumerate(global_model.parameters()):

            param.data -= global_g[p_idx].data
            for i in range(n_sampled):
                list(model_list[i].parameters())[p_idx].data = param.data + torch.zeros_like(param.data)

        if (iteration + 1) % 1 == 0:
            print("Train Iter {}:\tLoss: {:.6f}".format(
                iteration, iteration_loss))
            # eval_utils.check_sparsification_ratio(global_g_list)
            # eval_utils.check_accuracy(test_data, global_model, device)
            sys.stdout.flush()

            # check param level cosine similarity
            # if args.similarity == "cosine":
            #     print("Iter", iteration, "Cosine")
            #     print("Iter", iteration, "Euc")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    random.seed(10)
    np.random.seed(10)

    print(args)
    batch_size = args.train_bsz
    worker_number = 100

    # local configuration
    if args.local_config == 1:
        batch_size = 10

    from read_db import get_dataloaders
    list_dls_train, list_dls_test = get_dataloaders(args.data, batch_size)

    print("finish")

    train(list_dls_train, list_dls_test)


