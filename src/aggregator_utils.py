import torch
import random

def mean(args, workers, g_list):
    worker_number = len(workers)
    # g_res = [torch.zeros_like(param.data) for param in g_list[0]]
    for i in workers:
        for g_idx, g_param in enumerate(g_list[0]):
            g_list[i - 1][g_idx].data /= worker_number
    return g_list


def param_flip_flop(args, workers, g_list, assign_list, device):
    worker_number = len(workers)
    g_res = []

    cores_worker_list = [v for v in range(worker_number)]

    if args.shuffle_worker:
        random.shuffle(cores_worker_list)

    print(cores_worker_list)
    # get total parameter number
    # total_param = 0
    # for g_idx, g_param in enumerate(g_list[0]):
    #     total_param += len(torch.flatten(g_list[0][g_idx]))
    #
    # seg_size = int(total_param / worker_number) + 1
    # print(seg_size)

    for g_idx, g_param in enumerate(g_list[0]):

        if args.random_assign_list:
            assign_list[g_idx] = torch.randint(worker_number, g_list[0][g_idx].shape).to(device)

        # g_param_tmp = torch.zeros_like(g_list[0][g_idx]).to(device) + g_list[0][g_idx]
        for i in range(worker_number):
            cores_i = cores_worker_list[i]
            mask = assign_list[g_idx] != cores_i
            # print(mask)
            g_list[i][g_idx][mask] = 0.
            # g_param_tmp[mask] = g_list[i][g_idx][mask]

    return g_list


def layer_flip_flop(args, workers, g_list):
    worker_number = len(workers)
    cores_worker_list = [v + 1 for v in range(worker_number)]

    if args.shuffle_worker:
        random.shuffle(cores_worker_list)
    # print(cores_worker_list)
    g_res = []

    # get total layer number
    total_layer = 0
    for g_idx, g_param in enumerate(g_list[0]):
        total_layer += 1

    seg_size = int(total_layer / worker_number) + 1
    # print(seg_size)

    for g_idx, g_param in enumerate(g_list[0]):
        cores_worker = cores_worker_list[int(g_idx / seg_size)]
        # print(cores_worker)
        g_res_param = torch.zeros_like(g_list[cores_worker - 1][g_idx]) + g_list[cores_worker - 1][g_idx]

        g_res.append(g_res_param)

    # return g_res
    return g_list


def get_topk(args, g_remain, g_new, ratio, dev):
    for idx, g_layer in enumerate(g_new):
        g_remain[idx] += g_layer

    # print(dev)
    g_remain_abs_vector = torch.empty(0).to(dev)
    g_remain_abs = []
    for idx, g_layer in enumerate(g_remain):
        g_remain_layer_abs = torch.abs(g_remain[idx])
        g_remain_abs.append(g_remain_layer_abs)
        g_remain_layer_abs_reshape = g_remain_layer_abs.reshape(torch.numel(g_remain_layer_abs))
        g_remain_abs_vector = torch.cat((g_remain_abs_vector, g_remain_layer_abs_reshape),dim=0)  # merge two vectors into one vector

    param_num = torch.numel(g_remain_abs_vector)
    k = int(param_num * ratio)
    k = k if k>0 else 1
    top_k = torch.topk(g_remain_abs_vector, k)
    threshold = top_k[0][k-1].item()
    # print(threshold)
    g_upload = []
    for idx, g_layer in enumerate(g_remain_abs):
        mask = g_layer >= threshold
        g_upload_layer = torch.zeros_like(g_layer).to(dev)
        g_upload_layer[mask] += g_remain[idx][mask]
        g_remain[idx][mask] = 0.
        g_upload.append(g_upload_layer)

    return g_remain, g_upload
