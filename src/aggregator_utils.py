import torch
import random

def mean(args, workers, g_list):
    worker_number = len(workers)
    # g_res = [torch.zeros_like(param.data) for param in g_list[0]]
    for i in workers:
        for g_idx, g_param in enumerate(g_list[0]):
            g_list[i - 1][g_idx].data /= worker_number
    return g_list


# distribution list - assign by worker
# [[0, 1], [1, 2]] - worker 1 chooses chunk 0 and chunk 1
def chunk_flip_flop(args, workers, g_list, assign_list, distribution_list, device):
    worker_number = len(workers)
    chunk_number = args.chunks
    chunk_count = [0 for i in range(chunk_number)]
    # count the number in each chunk
    for i in range(worker_number):
        worker_chunk_list = distribution_list[i]
        for j in range(len(worker_chunk_list)):
            chunk_count[worker_chunk_list[j]] += 1

    print("worker: ", distribution_list)
    print("chunk count: ", chunk_count)

    for g_idx, g_param in enumerate(g_list[0]):

        for i in range(worker_number):
            mask_not = 0
            for j in range(len(distribution_list[i])):
                core = distribution_list[i][j]
                mask_is = assign_list[g_idx] == core
                if mask_not == 0:
                    mask_not = assign_list[g_idx] != core
                else:
                    mask_not = mask_not & (assign_list[g_idx] != core)
                g_list[i][g_idx][mask_is] /= chunk_count[core]

            g_list[i][g_idx][mask_not] = 0.

    return g_list


def param_flip_flop(args, workers, g_list, assign_list, param_level, device):
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
            for j in range(1, param_level):
                cores_j = cores_worker_list[(i + j) % worker_number]
                mask = mask & (assign_list[g_idx] != cores_j)
                # [1, 0, 2, 1, 0]
                # i = 1: false, true, true, false, true
                # j = 2: true, true, false, true, true
            g_list[i][g_idx][mask] = 0.
            g_list[i][g_idx] /= param_level
            # g_param_tmp[mask] = g_list[i][g_idx][mask]

    return g_list, cores_worker_list


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
