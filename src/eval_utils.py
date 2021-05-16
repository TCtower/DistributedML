import torch
import numpy as np
from scipy import spatial

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