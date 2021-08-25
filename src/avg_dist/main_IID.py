import torch
import argparse
import numpy as np
# import pandas as pd
import sys
from scipy import spatial
import random

from cjltest.utils_model import MySGD

import torch.nn.functional as F

sys.path.append("..")
print(sys.path)
from src.models import alexnet
from src.utils import aggregator_utils, data_utils

parser = argparse.ArgumentParser()
# 集群信息
parser.add_argument('--workers', type=int, default=5)
parser.add_argument('--train-bsz', type=int, default=512)
parser.add_argument('--aggregator', type=str, default='mean')
parser.add_argument('--compression', type=str, default='topk')
parser.add_argument('--similarity', type=str, default='cosine')
parser.add_argument('--ratio', type=float, default=0.01)
parser.add_argument('--shuffle-worker', type=int, default=1)
parser.add_argument('--random-assign-list', type=int, default=1)
parser.add_argument('--local-config', type=int, default=1)

args = parser.parse_args()


def check_sparsification_ratio(global_g_list):
    worker_number = len(global_g_list)
    spar_ratio = 0.

    total_param = 0
    for g_idx, g_param in enumerate(global_g_list[0]):
        total_param += len(torch.flatten(global_g_list[0][g_idx]))

    for i in range(worker_number):
        non_zero_param = 0
        for g_idx, g_param in enumerate(global_g_list[i]):
            mask = g_param != 0.
            # print(mask)
            non_zero_param += float(torch.sum(mask))
        # print(i, non_zero_param, total_param)
        # print(non_zero_param / total_param)
        # print((non_zero_param / total_param) / worker_number)
        spar_ratio += (non_zero_param / total_param) / worker_number

    print('Ratio: {:.6f}'.format(spar_ratio))
    return spar_ratio


def check_accuracy(loader, model, device):
    num_correct = 0
    num_samples = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device=device)
            y = y.to(device=device)

            # print("sd")

            scores = model(x)
            _, predictions = scores.max(1)
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f'Got {num_correct} / {num_samples} with accuracy {float(num_correct) / float(num_samples) * 100:.2f}')

    model.train()


def cosine_similarity_pair_list(v):
    num = len(v)
    res = []
    for i in range(num):
        tmp = []
        for j in range(num):
            tmp.append(cosine_similarity(v[i], v[j]))
        res.append(tmp)
    return res


def cosine_similarity(v1,v2):
    return 1 - spatial.distance.cosine(v1, v2)


def check_cosine_similarity(global_g_list, dev):
    n_clients = len(global_g_list)
    # tt = np.array(global_g_list)
    grad_len = []
    for param in global_g_list[0]:
        grad_len.append(np.array(param.shape).prod())

    flat_g_list = []
    for i in range(n_clients):
        tmp_flat_list = []
        for j in range(len(global_g_list[0])):
            tmp_flat_list.append(np.reshape(global_g_list[i][j].cpu().data.numpy(), grad_len[j]))
        # print(tmp_tensor)
        flat_g_list.append(np.concatenate(tmp_flat_list))

    # print(flat_g_list)
    # print(np.array(global_g_list[0]).shape)
    # grad_len = np.array(np.array(global_g_list[0]).shape).prod()
    # print(grad_len)
    # cs = smp.cosine_similarity(flat_g_list)
    cs = cosine_similarity_pair_list(flat_g_list)
    # cs = 0
    for i in cs:
        print(i)
    # print(cs)
    # cs = smp.cosine_similarity(flat_g_list)
    # print(cs)
    sys.stdout.flush()


def check_cosine_similarity_chunk(upload_g_list, assign_list, cores_worker_list, device):
    worker_number = len(upload_g_list)
    chunk_g_list = [[] for i in range(worker_number)]

    for i in range(worker_number):
        cores_i = cores_worker_list[i]
        cores_j = cores_worker_list[(i + 1) % worker_number]

        chunk_i_list = []
        chunk_j_list = []
        for g_idx, g_param in enumerate(upload_g_list[0]):

            mask_i = assign_list[g_idx] == cores_i
            mask_j = assign_list[g_idx] == cores_j

            chunk_i_list.append(upload_g_list[i][g_idx][mask_i])
            chunk_j_list.append(upload_g_list[i][g_idx][mask_j])

        chunk_g_list[cores_i].append(chunk_i_list)
        chunk_g_list[cores_j].append(chunk_j_list)


    chunk_similarity_list = []
    for chunk_idx in range(worker_number):

        grad_len = []
        for param in chunk_g_list[chunk_idx][0]:
            grad_len.append(np.array(param.shape).prod())

        print(grad_len)

        flat_g_list = []
        for i in range(2):
            tmp_flat_list = []
            for j in range(len(chunk_g_list[chunk_idx][i])):
                tmp_flat_list.append(np.reshape(chunk_g_list[chunk_idx][i][j].cpu().data.numpy(), grad_len[j]))
            # print(tmp_tensor)
            flat_g_list.append(np.concatenate(tmp_flat_list))

        chunk_similarity_list.append(cosine_similarity(flat_g_list[0], flat_g_list[1]))

    print("chunk similarity")
    print(chunk_similarity_list)


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
    assign_list = []
    for g_idx, g_param in enumerate(global_model.parameters()):
        assign_list.append(torch.randint(worker_number, g_param.shape).to(device))

    # print(assign_list)

    g_remain_list = []
    for i in workers:
        g_remain = []
        for g_idx, g_param in enumerate(global_model.parameters()):
            g_remain_param = torch.zeros_like(g_param).to(device)
            g_remain.append(g_remain_param)

        g_remain_list.append(g_remain)

    accum_g_list = []
    cores_workers_list = []

    # number of overlap
    param_level = 1
    prev_loss = 10.0
    for iteration in range(1, iterations + 1):

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
            # print(len(delta_ws))
            # delta_ws = []
            # with torch.no_grad():
            #     for p_idx, param in enumerate(model.parameters()):
            #         # if params.requires_grad:
            #         param.data -= 0.1 * param.grad.data.clone()
            #
            #         delta_ws.append(param.grad.clone())
            # print(delta_ws)

        # dynamic
        if (args.aggregator == "param") and (iteration % 10 == 0):
            # if iteration_loss > prev_loss:
            #     # iteration_loss > prev_loss
            #     param_level += 1
            #     if param_level > worker_number:
            #         param_level = worker_number
            # else:
            #     # iteration_loss < prev_loss
            #     param_level -= 1
            #     if param_level < 1:
            #         param_level = 1
            print(prev_loss, iteration_loss, param_level)
            sys.stdout.flush()
            prev_loss = iteration_loss

        # flip flop choice
        global_g_list = []
        if args.aggregator == "mean":
            global_g_list = aggregator_utils.mean(args, workers, g_list)
        elif args.aggregator == "layer":
            global_g_list = aggregator_utils.layer_flip_flop(args, workers, g_list)
        elif args.aggregator == "param":
            global_g_list, cores_workers_list = aggregator_utils.param_flip_flop(args, workers, g_list, assign_list, param_level, device)

        # compression choice
        if args.compression == 'topk':
            ratio = args.ratio

            if args.aggregator == "param":
                ratio = (ratio / worker_number) * param_level

            for i in workers:
                g_remain, g_upload = aggregator_utils.get_topk(args, g_remain_list[i - 1], global_g_list[i - 1], ratio, device)
                g_remain_list[i-1] = g_remain
                global_g_list[i-1] = g_upload

        # aggregate gradients
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
            check_sparsification_ratio(global_g_list)
            check_accuracy(test_data, global_model, device)
            sys.stdout.flush()

            # check cosine similarity
            if args.similarity == "cosine":
                print("Iter", iteration, "Cosine")
                check_cosine_similarity(accum_g_list, device)
                # check_cosine_similarity_chunk(global_g_list, assign_list, cores_workers_list, device)


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

    train_data_list, test_loader = data_utils.get_loader(batch_size, worker_number, 10)
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
