import torch
import torchvision
from torchvision import transforms, datasets
from torch.utils.data.dataset import Dataset

import numpy as np
import matplotlib.pyplot as plt

# class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
#                'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
class_names = ['0', '1', '2', '3', '4',
                '5', '6', '7', '8', '9']
global_data_name = "none"

def plot_random_figure(train_loader):
    dataiter = iter(train_loader)
    images, labels = dataiter.next()

    # creat grid of images
    img_grid = torchvision.utils.make_grid(images[0])

    # show images & labels
    matplotlib_imshow(img_grid)
    print(class_names[labels[0]])


def matplotlib_imshow(img):
    img = img.mean(dim=0)
    img = img / 2 + 0.5  # unnormalize
    npimg = img.numpy()
    plt.imshow(npimg, cmap="Greys")
    plt.show()


def load_dataset(data_name):

    if data_name == "MNIST":
        print("MNIST")
        train_set = torchvision.datasets.MNIST(
            root='../../data/MNIST',
            train=True,
            download=True
        )

        test_set = torchvision.datasets.MNIST(
            root='../../data/MNIST',
            train=False,
            download=True,
        )
    else:
        print("FashionMNIST")
        train_set = torchvision.datasets.FashionMNIST(
            root='../../data/FashionMNIST',
            train=True,
            download=True
        )

        test_set = torchvision.datasets.FashionMNIST(
            root='../../data/FashionMNIST',
            train=False,
            download=True,
        )

    x_train, y_train = train_set.train_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(train_set.train_labels)
    x_test, y_test = test_set.test_data.numpy().reshape(-1, 1, 28, 28) / 255, np.array(test_set.test_labels)

    return x_train, y_train, x_test, y_test


def split_image_data(data, labels, n_clients=10, classes_per_client=10, shuffle=True, verbose=True, balancedness=None):
    '''
    Splits (data, labels) evenly among 'n_clients s.t. every client holds 'classes_per_client
    different labels
    data : [n_data x shape]
    labels : [n_data (x 1)] from 0 to n_labels
    '''

    # constants
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    # if balancedness >= 1.0:
    data_per_client = [n_data // n_clients] * n_clients
    data_per_client_per_class = [data_per_client[0] // classes_per_client] * n_clients
    # data_per_client = [n_data] * n_clients
    print(data_per_client, data_per_client_per_class)

    # else:
    #     fracs = balancedness ** np.linspace(0, n_clients - 1, n_clients)
    #     fracs /= np.sum(fracs)
    #     fracs = 0.1 / n_clients + (1 - 0.1) * fracs
    #     data_per_client = [np.floor(frac * n_data).astype('int') for frac in fracs]
    #
    #     data_per_client = data_per_client[::-1]
    #
    #     data_per_client_per_class = [np.maximum(1, nd // classes_per_client) for nd in data_per_client]

    # if sum(data_per_client) > n_data:
    #     print("Impossible Split")
    #     exit()

    # sort for labels
    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    # split data among clients
    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []
        budget = data_per_client[i]
        c = np.random.randint(n_labels)
        while budget > 0:
            take = min(data_per_client_per_class[i], len(data_idcs[c]), budget)

            client_idcs += data_idcs[c][:take]
            data_idcs[c] = data_idcs[c][take:]

            budget -= take
            c = (c + 1) % n_labels

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split(clients_split)

    return clients_split


def split_image_data_customize(data, labels, n_clients=10, class_distribution=None, shuffle=True, verbose=True):
    if len(class_distribution) != n_clients:
        print("Invalid distribution")
        exit()

    # constants
    n_data = data.shape[0]
    n_labels = np.max(labels) + 1

    data_idcs = [[] for i in range(n_labels)]
    for j, label in enumerate(labels):
        data_idcs[label] += [j]
    if shuffle:
        for idcs in data_idcs:
            np.random.shuffle(idcs)

    clients_split = []
    c = 0
    for i in range(n_clients):
        client_idcs = []

        for c in range(len(class_distribution[i])):
            take = class_distribution[i][c]
            client_idcs += data_idcs[c][:take]

        clients_split += [(data[client_idcs], labels[client_idcs])]

    def print_split(clients_split):
        print("Data split:")
        for i, client in enumerate(clients_split):
            split = np.sum(client[1].reshape(1, -1) == np.arange(n_labels).reshape(-1, 1), axis=1)
            print(" - Client {}: {}".format(i, split))
        print()

    if verbose:
        print_split(clients_split)

    return clients_split


class CustomImageDataset(Dataset):
    '''
    A custom Dataset class for images
    inputs : numpy array [n_data x shape]
    labels : numpy array [n_data (x 1)]
    '''

    def __init__(self, inputs, labels, transforms=None):
        assert inputs.shape[0] == labels.shape[0]
        self.inputs = torch.Tensor(inputs)
        self.labels = torch.Tensor(labels).long()
        self.transforms = transforms

    def __getitem__(self, index):
        img, label = self.inputs[index], self.labels[index]

        if self.transforms is not None:
            img = self.transforms(img)

        return (img, label)

    def __len__(self):
        return self.inputs.shape[0]


def get_loader(transform, batch_size, worker_number, class_per):

    x_train, y_train, x_test, y_test = load_dataset()

    split = split_image_data(x_train, y_train, n_clients=worker_number,
                             classes_per_client=class_per, verbose=True)

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transform),
                                                  batch_size=batch_size, shuffle=True) for x, y in split]

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transform), batch_size=batch_size,
                                              shuffle=True)
    plot_random_figure(client_loaders[0])
    return client_loaders, test_loader


def get_loader_customize(data_name, transform, batch_size, worker_number, class_dist):

    x_train, y_train, x_test, y_test = load_dataset(data_name)

    split = split_image_data_customize(x_train, y_train, n_clients=worker_number,
                                       class_distribution=class_dist,
                                       shuffle=True,
                                       verbose=True)

    client_loaders = [torch.utils.data.DataLoader(CustomImageDataset(x, y, transform),
                                                  batch_size=batch_size, shuffle=True) for x, y in split]

    test_loader = torch.utils.data.DataLoader(CustomImageDataset(x_test, y_test, transform), batch_size=batch_size,
                                              shuffle=True)
    # plot_random_figure(client_loaders[0])
    return client_loaders, test_loader