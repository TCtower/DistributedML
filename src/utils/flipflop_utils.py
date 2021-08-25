
from src.utils import eval_utils
import numpy as np
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster

def hierarchy_cluster(data, method='average', threshold=2):

    data = np.array(data)

    Z = linkage(data, method=method)
    cluster_assignments = fcluster(Z, threshold, criterion='maxclust')
    # print(type(cluster_assignments))
    num_clusters = cluster_assignments.max()
    indices = get_cluster_indices(cluster_assignments)

    return num_clusters, indices


def get_cluster_indices(cluster_assignments):
    n = cluster_assignments.max()
    indices = []
    for cluster_number in range(1, n + 1):
        indices.append(np.where(cluster_assignments == cluster_number)[0])

    return indices


def cluster_generation(warmup_g_list, average_similarity_list, params_name, warmup_similarity, overlap_level, device):
    worker_num = len(warmup_g_list)

    if warmup_similarity == "none":
        layer_cluster = []
        for g_idx, g_param in enumerate(warmup_g_list[0]):
            layer_cluster.append([[i for i in range(worker_num)] for j in range(overlap_level)])
        return layer_cluster

    layer_cluster = []
    for g_idx, g_param in enumerate(warmup_g_list[0]):
        if "bias" in params_name[g_idx]:
            layer_cluster.append(layer_cluster[-1])
            continue

        similarity_matrix = average_similarity_list[g_idx]

        # print(similarity_matrix)

        num_clusters, indices = hierarchy_cluster(similarity_matrix, threshold=overlap_level)
        # print(g_idx, params_name[g_idx], num_clusters, indices)

        layer_cluster.append(indices)

    return layer_cluster


def cluster_generation_client(similarity_list, overlap_level):

    num_clusters, indices = hierarchy_cluster(similarity_list, threshold=overlap_level)

    return indices
