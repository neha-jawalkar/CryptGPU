import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import crypten
import crypten.mpc as mpc
import crypten.communicator as comm

from crypten.mpc import MPCTensor
from crypten.mpc.ptype import ptype as Ptype


import torchvision
import torchvision.models as models
import torch.autograd.profiler as profiler

import logging
import time
import timeit
import argparse

from network import *

def inference(model, input_size, batch_size=1, device="cuda"):
    comm.get().set_verbosity(True)

    bs = batch_size
    c, w, h = input_size
    x = crypten.cryptensor(torch.rand((bs, c, w, h)), device=device, requires_grad=False)

    model = crypten.nn.from_pytorch(model, dummy_input=torch.empty(bs, c, w, h))
    model = model.encrypt()
    model = model.to(device) 

    model.eval()
    model.replicate_parameters()

    total_time = 0
    comm_time = 0
    conv_time, pool_time, relu_time, matmul_time = 0, 0, 0, 0
    for i in range(6):
        comm.get().reset_communication_stats()
        
        tic = time.perf_counter()
        model(x)
        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic
            comm_time += comm.get().comm_time
            conv_time += comm.get().time_conv
            relu_time += comm.get().time_relu
            pool_time += comm.get().time_pool
            matmul_time += comm.get().time_matmul

            # if comm.get().get_rank() == 0:
            #     print(f"Iteration {i} runtime: {toc - tic}")

        comm.get().print_total_communication()

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Total Communication: {comm.get().total_comm_bytes}")
        print(f"Avg Runtime: {total_time / 5}")
        print(f"Avg Comm: {comm_time / 5}")
        print(f"Avg Linear: {conv_time + matmul_time/ 5}")
        print(f"Avg ReLU: {relu_time / 5}")
        print(f"Avg Pool: {pool_time / 5}")

def to_mb(num_bytes):
    return f"{num_bytes / (float(1024)**2)} MB"


def training(model, input_size, batch_size, num_classes, device="cuda"):
    comm.get().set_verbosity(True)
    rank = comm.get().get_rank()

    c, h, w = input_size
    bs = batch_size

    criterion = crypten.nn.CrossEntropyLoss()
    model = crypten.nn.from_pytorch(model, dummy_input=torch.empty(bs, c, h, w))
    model = model.to(device)
    model.encrypt() 
    model.train()

    labels =  torch.ones(bs, requires_grad=False).long().to(device)
    labels = F.one_hot(labels, num_classes=num_classes)
    labels = crypten.cryptensor(labels, src=0)

    input = torch.randn([bs,c,w,h], requires_grad=False)
    input = crypten.cryptensor(input, src=0).to(device)

    total_time = 0
    comm_time, comm_time_relu, comm_time_conv, comm_time_matmul, comm_time_pool, comm_time_softmax = [0]*6
    conv_time, pool_time, relu_time, matmul_time, softmax_time = 0, 0, 0, 0, 0
    num_iterations = 6
    assert num_iterations > 1
    for i in range(num_iterations):
        if comm.get().get_rank() == 0:
            print("Iteration", i)
        comm.get().reset_communication_stats()
        tic = time.perf_counter()

        output = model(input)

        loss = criterion(output, labels)
        loss.backward()
        model.update_parameters(learning_rate=0.1)

        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic
            comm_time += comm.get().comm_time
            conv_time += comm.get().time_conv
            relu_time += comm.get().time_relu
            pool_time += comm.get().time_pool
            matmul_time += comm.get().time_matmul
            softmax_time += comm.get().time_softmax
            if comm.get().get_rank() == 0:
                comm_time_relu += comm.get().comm_time_relu
                comm_time_conv += comm.get().comm_time_conv
                comm_time_matmul += comm.get().comm_time_matmul
                comm_time_pool += comm.get().comm_time_pool
                comm_time_softmax += comm.get().comm_time_softmax
            # if comm.get().get_rank() == 0:
            #     print(f"Iteration {i} runtime: {toc - tic}")

        comm.get().print_total_communication()

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Avg Runtime: {total_time / (num_iterations-1)}\n")

        print(f"Avg Runtime Conv: {conv_time/ (num_iterations-1)}")
        print(f"Avg Runtime Matmul: {matmul_time/ (num_iterations-1)}")
        print(f"Avg Runtime Pool: {pool_time / (num_iterations-1)}")
        print(f"Avg Runtime ReLU: {relu_time / (num_iterations-1)}")
        print(f"Avg Runtime Softmax: {softmax_time / (num_iterations-1)}\n")
        
        print(f"Avg Comm: {comm_time / (num_iterations-1)}\n")

        print(f"Avg Comm time Conv: {comm_time_conv / (num_iterations-1)}")
        print(f"Avg Comm time Matmul: {comm_time_matmul / (num_iterations-1)}")
        print(f"Avg Comm time Pool: {comm_time_pool / (num_iterations-1)}")
        print(f"Avg Comm time ReLU: {comm_time_relu / (num_iterations-1)}")
        print(f"Avg Comm time Softmax: {comm_time_softmax / (num_iterations-1)}\n")
        
        print(f"Total rounds: {comm.get().comm_rounds}\n")

        print(f"Comm rounds Conv: {comm.get().comm_rounds_conv}")
        print(f"Comm rounds Matmul: {comm.get().comm_rounds_matmul}")
        print(f"Comm rounds Pool: {comm.get().comm_rounds_pool}")
        print(f"Comm rounds ReLU: {comm.get().comm_rounds_relu}")
        print(f"Comm rounds Softmax: {comm.get().comm_rounds_softmax}\n")

        print(f"Total Communication: {to_mb(comm.get().total_comm_bytes)}")
        print(f"Total communication for one party: {to_mb(comm.get().comm_bytes)}\n")

        print(f"Comm size Conv: {to_mb(comm.get().comm_bytes_conv)}")
        print(f"Comm size Matmul: {to_mb(comm.get().comm_bytes_matmul)}")
        print(f"Comm size Pool: {to_mb(comm.get().comm_bytes_pool)}")
        print(f"Comm size ReLU: {to_mb(comm.get().comm_bytes_relu)}")
        print(f"Comm size Softmax: {to_mb(comm.get().comm_bytes_softmax)}")




def inference_plaintext(model, input_size, device="cuda"):

    c, w, h = input_size
    x = torch.rand((1, c, w, h), device=device, requires_grad=False)

    model = model.to(device) 
    model.eval()

    total_time = 0
    for i in range(101):
        comm.get().reset_communication_stats()
        
        tic = time.perf_counter()
        model(x)
        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic

        comm.get().print_total_communication()

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Avg Runtime: {total_time / 100}")


def training_plaintext(model, input_size, batch_size, num_classes, device="cuda"):

    c, h, w = input_size
    bs = batch_size

    model = model.to(device)
    print(model)
    model.train()

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    input = torch.randn([bs,c,w,h], requires_grad=True).to(device)
    labels =  torch.ones(bs, requires_grad=False).long().to(device)

    total_time = 0
    comm_time = 0
    conv_time, pool_time, relu_time, matmul_time = 0, 0, 0, 0
    for i in range(101):
        tic = time.perf_counter()

        output = model(input)

        optimizer.zero_grad()
        loss = criterion(output, labels)
        loss.backward()
        optimizer.step()

        toc = time.perf_counter()

        if i != 0:
            total_time += toc - tic

            # if comm.get().get_rank() == 0:
            #     print(f"Iteration {i} runtime: {toc - tic}")

    if comm.get().get_rank() == 0:
        print("----------- Statistics ----------------")
        print(f"Avg Runtime: {total_time / 100}")


def select_model(dataset, network):
    if dataset == "mnist":
        input_size = (1,28,28)
        num_classes = 10
        if network == "lenet":
            model = LeNet()
    elif dataset == "cifar10":
        input_size = (3,32,32)
        num_classes = 10
        if network == "alexnet":
            model = AlexNet(num_classes=10)
        elif network == "vgg16":
            model = VGG16(num_classes=10)
    elif dataset == 'tinyin':
        input_size = (3,64,64)
        num_classes = 200
        if network == 'alexnet':
            model = AlexNet(num_classes=200)
        elif network == "vgg16":
            model = VGG16(num_classes=200)
    elif dataset == 'imagenet':
        input_size = (3, 224, 224)
        num_classes = 1000
        if network == 'alexnet':
            model = AlexNet(num_classes=1000)
        elif network == "vgg16":
            model = VGG16(num_classes=1000)
        elif network == "resnet34":
            model = models.resnet34()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)
        elif network == "resnet50":
            model = models.resnet50()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)
        elif network == "resnet101":
            model = models.resnet101()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)
        elif network == "resnet152":
            model = models.resnet152()
            model.maxpool = nn.AvgPool2d(kernel_size=3, stride=2)

    return model, input_size, num_classes

def train_all():
    train_config = [
        # ["mnist", "lenet", 128],
        # ["cifar10", "alexnet", 128],
        # ["cifar10", "vgg16", 32],
        ["tinyin", "alexnet", 128],
        # ["tinyin", "vgg16", 8],
    ]
    for dataset, network, bs in train_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Training on {dataset} dataset with {network} network")
        training(model, input_size, bs, num_classes, device="cpu") # cuda or cpu


def inference_all():
    inference_config = [
        # ["mnist", "lenet"],
        # ["cifar10", "alexnet"],
        # ["cifar10", "vgg16"],
        # ["tinyin", "alexnet"],
        # ["tinyin", "vgg16"],
        # ["imagenet", "alexnet"],
        # ["imagenet", "vgg16"],
        ["imagenet", "resnet50"],
        ["imagenet", "resnet101"],
        ["imagenet", "resnet152"]
    ]
    for dataset, network in inference_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Running inference on {dataset} dataset with {network} network")
        inference(model, input_size, device="cpu") # can be either cpu or cuda


def train_all_plaintext():
    train_config = [
        ["mnist", "lenet", 128],
        ["cifar10", "alexnet", 128],
        ["cifar10", "vgg16", 32],
        ["tinyin", "alexnet", 128],
        ["tinyin", "vgg16", 8],
    ]
    for dataset, network, bs in train_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Training on {dataset} dataset with {network} network")
        training_plaintext(model, input_size, bs, num_classes, device="cuda")


def inference_all_plaintext():
    inference_config = [
        # ["mnist", "lenet"],
        # ["cifar10", "alexnet"],
        # ["cifar10", "vgg16"],
        # ["tinyin", "alexnet"],
        # ["tinyin", "vgg16"],
        # ["imagenet", "alexnet"],
        # ["imagenet", "vgg16"],
        ["imagenet", "resnet50"],
        ["imagenet", "resnet101"],
        ["imagenet", "resnet152"]
    ]
    for dataset, network in inference_config:
        model, input_size, num_classes = select_model(dataset, network)
        if comm.get().get_rank() == 0:
            print(f"Running inference on {dataset} dataset with {network} network")
        inference_plaintext(model, input_size, device="cpu") # can be cpu or cuda


def batch_inference():
    inference_config = [
        ["cifar10", "alexnet", 64],
        ["cifar10", "vgg16", 64],
        ["imagenet", "resnet50", 8],
        ["imagenet", "resnet101", 8],
        ["imagenet", "resnet152", 8]
    ]

    for dataset, network, bs in inference_config:
        model, input_size, num_classes = select_model(dataset, network)
        inference(model, input_size, bs, device='cpu') # can be cpu or cuda

def measure_comm_time():
    # bytes_list = 
    # [
    #     33849344,
    #     8650752,
    #     336134144,
    #     170131456,
    #     47737856,
    #     2162688,
    #     864768,
    #     4325376,
    #     1729536,
    #     4194304,
    #     1048576,
    #     20973568,
    #     150994944,
    #     75497472,
    #     37748736,
    #     18874368,
    #     4718592,
    #     294912,
    #     1884872,
    #     38408]
    bytes_list = [
        44712192,
        20643840,
        305127936,
        151781376,
        10485760,
        2891776,
        20971520,
        5783552,
        6553600,
        1048576,
        62921472,
        5251072,
        286654464,
        88473600,
        9437184,
        254803968,
        78643200,
        8388608,
        30157832,
        614408
    ]
    num_iterations = 6
    for num_bytes in bytes_list:
        comm_time = 0
        for i in range(num_iterations):
            tensor_to_send = torch.zeros_like(torch.empty(num_bytes), dtype=torch.uint8).data
            tensor_to_recv = torch.empty(num_bytes, dtype=torch.int8)
            start_time = time.perf_counter()
            if comm.get().get_rank() == 0:
                send_req = comm.get().isend(tensor=tensor_to_send, dst=1)
                send_req.wait()
            elif comm.get().get_rank() == 1:
                recv_req = comm.get().irecv(tensor=tensor_to_recv, src=0)
                recv_req.wait()
            end_time = time.perf_counter()
            if i > 0:
                comm_time += (end_time - start_time)
        if comm.get().get_rank() == 0:
            print(f"Time taken to send {num_bytes} bytes:", comm_time / (num_iterations - 1))
    

# A playground to test different network and dataset combinations
def test():
    dataset = "cifar10"
    network = "alexnet"
    device = "cuda"
    train = True
    batch_size = 1

    model, input_size, num_classes = select_model(dataset, network)

    if train:
        training(model, input_size, batch_size, num_classes, device)
    else:
        inference(model, input_size, device)


parser = argparse.ArgumentParser()
experiments = ['test', 'train_all', 'inference_all', 'train_all_plaintext', 'inference_all_plaintext', 'batch_inference', 'comm_time']
parser.add_argument(
    "--exp",
    "-e",
    required=False,
    default="test",
    help="Experiment to run",
)

if __name__ == '__main__':
    import multiprocess_launcher

    args = parser.parse_args()
    assert args.exp in experiments
    func = globals()[args.exp]

    launcher = multiprocess_launcher.MultiProcessLauncher(
        3, func,
    )
    launcher.start()
    launcher.join()
    launcher.terminate()