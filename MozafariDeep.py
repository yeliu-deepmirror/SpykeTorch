#################################################################################
# Reimplementation of the 10-Class Digit Recognition Experiment Performed in:   #
# https://arxiv.org/abs/1804.00227                                              #
#                                                                               #
# Reference:                                                                    #
# Mozafari, Milad, et al.,                                                      #
# "Combining STDP and Reward-Modulated STDP in                                  #
# Deep Convolutional Spiking Neural Networks for Digit Recognition."            #
# arXiv preprint arXiv:1804.00227 (2018).                                       #
#                                                                               #
# Original implementation (in C++/CUDA) is available upon request.              #
#################################################################################

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import ImageFolder
from torch.nn.parameter import Parameter
import torchvision
import numpy as np
from SpykeTorch import snn
from SpykeTorch import functional as sf
from SpykeTorch import visualization as vis
from SpykeTorch import utils
from torchvision import transforms

use_cuda = True

class MozafariMNIST2018(nn.Module):
    def __init__(self, max_classes = 10):
        super(MozafariMNIST2018, self).__init__()

        self.conv1 = snn.Convolution(6, 30, 5, 0.8, 0.05)
        self.conv1_t = 15
        self.k1 = 5
        self.r1 = 3

        self.conv2 = snn.Convolution(30, 250, 3, 0.8, 0.05)
        self.conv2_t = 10
        self.k2 = 8
        self.r2 = 1

        self.conv3 = snn.Convolution(250, 200, 5, 0.8, 0.05)

        self.stdp1 = snn.STDP(self.conv1, (0.004, -0.003))
        self.stdp2 = snn.STDP(self.conv2, (0.004, -0.003))
        self.stdp3 = snn.STDP(self.conv3, (0.004, -0.003), False, 0.2, 0.8)
        self.anti_stdp3 = snn.STDP(self.conv3, (-0.004, 0.0005), False, 0.2, 0.8)
        self.max_ap = Parameter(torch.Tensor([0.15]))

        self.decision_map = []
        num_each_class = int(200 / max_classes)
        for i in range(max_classes):
            self.decision_map.extend([i]*num_each_class)

        self.ctx = {"input_spikes":None, "potentials":None, "output_spikes":None, "winners":None}
        self.spk_cnt1 = 0
        self.spk_cnt2 = 0

    def forward(self, input, max_layer):
        input = sf.pad(input.float(), (2,2,2,2), 0)
        if self.training:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                self.spk_cnt1 += 1
                if self.spk_cnt1 >= 500:
                    self.spk_cnt1 = 0
                    ap = torch.tensor(self.stdp1.learning_rate[0][0].item(), device=self.stdp1.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp1.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k1, self.r1, spk)
                self.ctx["input_spikes"] = input
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1))
            pot = self.conv2(spk_in)
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                self.spk_cnt2 += 1
                if self.spk_cnt2 >= 500:
                    self.spk_cnt2 = 0
                    ap = torch.tensor(self.stdp2.learning_rate[0][0].item(), device=self.stdp2.learning_rate[0][0].device) * 2
                    ap = torch.min(ap, self.max_ap)
                    an = ap * -0.75
                    self.stdp2.update_all_learning_rate(ap.item(), an.item())
                pot = sf.pointwise_inhibition(pot)
                spk = pot.sign()
                winners = sf.get_k_winners(pot, self.k2, self.r2, spk)
                self.ctx["input_spikes"] = spk_in
                self.ctx["potentials"] = pot
                self.ctx["output_spikes"] = spk
                self.ctx["winners"] = winners
                return spk, pot
            spk_in = sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2))
            pot = self.conv3(spk_in)
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            self.ctx["input_spikes"] = spk_in
            self.ctx["potentials"] = pot
            self.ctx["output_spikes"] = spk
            self.ctx["winners"] = winners
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output
        else:
            pot = self.conv1(input)
            spk, pot = sf.fire(pot, self.conv1_t, True)
            if max_layer == 1:
                return spk, pot
            pot = self.conv2(sf.pad(sf.pooling(spk, 2, 2), (1,1,1,1)))
            spk, pot = sf.fire(pot, self.conv2_t, True)
            if max_layer == 2:
                return spk, pot
            pot = self.conv3(sf.pad(sf.pooling(spk, 3, 3), (2,2,2,2)))
            spk = sf.fire(pot)
            winners = sf.get_k_winners(pot, 1, 0, spk)
            output = -1
            if len(winners) != 0:
                output = self.decision_map[winners[0][0]]
            return output

    def stdp(self, layer_idx):
        if layer_idx == 1:
            self.stdp1(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])
        if layer_idx == 2:
            self.stdp2(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def update_learning_rates(self, stdp_ap, stdp_an, anti_stdp_ap, anti_stdp_an):
        self.stdp3.update_all_learning_rate(stdp_ap, stdp_an)
        self.anti_stdp3.update_all_learning_rate(anti_stdp_an, anti_stdp_ap)

    def reward(self):
        self.stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

    def punish(self):
        self.anti_stdp3(self.ctx["input_spikes"], self.ctx["potentials"], self.ctx["output_spikes"], self.ctx["winners"])

def train_unsupervise(network, data, layer_idx):
    network.train()
    for i in range(len(data)):
        data_in = data[i]
        if use_cuda:
            data_in = data_in.cuda()
        network(data_in, layer_idx)
        network.stdp(layer_idx)


def pass_test(d, target, loose):
    if loose :
        return abs(d - target) < 2
    return d == target


def train_rl(network, data, target, loose=False):
    network.train()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if pass_test(d, target_in, loose):
                perf[0]+=1
                network.reward()
            else:
                perf[1]+=1
                network.punish()
        else:
            perf[2]+=1
    return perf/len(data)


def test(network, data, target, loose=False):
    network.eval()
    perf = np.array([0,0,0]) # correct, wrong, silence
    for i in range(len(data)):
        data_in = data[i]
        target_in = target[i]
        if use_cuda:
            data_in = data_in.cuda()
            target_in = target_in.cuda()
        d = network(data_in, 3)
        if d != -1:
            if pass_test(d, target_in, loose):
                perf[0]+=1
            else:
                perf[1]+=1
        else:
            perf[2]+=1
    return perf/len(data)


class S1C1Transform:
    def __init__(self, filter, gray = False, timesteps = 15):
        self.to_tensor = transforms.ToTensor()
        self.gray = gray
        self.gray_scale = transforms.Grayscale()
        self.filter = filter
        self.temporal_transform = utils.Intensity2Latency(timesteps)


    def __call__(self, image):
        if self.gray:
            image = self.gray_scale(image)
        image = self.to_tensor(image) * 255
        image.unsqueeze_(0)
        image = self.filter(image)
        image = sf.local_normalization(image, 8)
        temporal_image = self.temporal_transform(image)
        return temporal_image.sign().byte()


def get_transforms(gray = False):
    kernels = [ utils.DoGKernel(3,3/9,6/9),
                utils.DoGKernel(3,6/9,3/9),
                utils.DoGKernel(7,7/9,14/9),
                utils.DoGKernel(7,14/9,7/9),
                utils.DoGKernel(13,13/9,26/9),
                utils.DoGKernel(13,26/9,13/9)]
    filter_feature = utils.Filter(kernels, padding = 6, thresholds = 50)
    s1c1 = S1C1Transform(filter_feature, gray)
    return s1c1


def train_heading():
    s1c1 = get_transforms(True)

    model_root = "model/"
    dataset = ImageFolder("dataset/heading", s1c1) # adding transform to the dataset
    print("Load", len(dataset), "data")

    # splitting training and testing sets
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    split_point = int(0.75*len(indices))
    train_indices = indices[:split_point]
    test_indices = indices[split_point:]
    print("Size of the training set:", len(train_indices))
    print("Size of the  testing set:", len(test_indices))

    dataset = utils.CacheDataset(dataset)
    train_loader = DataLoader(dataset, sampler=SubsetRandomSampler(train_indices))
    test_loader = DataLoader(dataset, sampler=SubsetRandomSampler(test_indices))

    mozafari = MozafariMNIST2018(20)
    if use_cuda:
        mozafari.cuda()
    print(mozafari)
    print(mozafari.decision_map)

    # Training The First Layer
    layer_1_path = model_root + "/saved_l1.net"
    layer_2_path = model_root + "/saved_l2.net"
    layer_3_path = model_root + "/saved_l3.net"
    layer_3l_path = model_root + "/saved_l3l.net"
    if os.path.isfile(layer_1_path):
        print("Loading the first layer")
        mozafari.load_state_dict(torch.load(layer_1_path))
    else:
        print("Training the first layer")
        for epoch in range(20):
            print("Epoch", epoch)
            for data,targets in train_loader:
                train_unsupervise(mozafari, data, 1)
        torch.save(mozafari.state_dict(), layer_1_path)
    # Training The Second Layer
    if os.path.isfile(layer_2_path):
        print("Loading the second layer")
        mozafari.load_state_dict(torch.load(layer_2_path))
    else:
        print("Training the second layer")
        for epoch in range(40):
            print("Epoch", epoch)
            for data,targets in train_loader:
                train_unsupervise(mozafari, data, 2)
        torch.save(mozafari.state_dict(), layer_2_path)

    # initial adaptive learning rates
    apr = mozafari.stdp3.learning_rate[0][0].item()
    anr = mozafari.stdp3.learning_rate[0][1].item()
    app = mozafari.anti_stdp3.learning_rate[0][1].item()
    anp = mozafari.anti_stdp3.learning_rate[0][0].item()

    adaptive_min = 0
    adaptive_int = 1
    apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
    anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
    app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
    anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

    # perf
    best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
    best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
    best_test_loose = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch

    # Training The Third Layer
    if os.path.isfile(layer_3_path):
        print("Loading the third layer")
        mozafari.load_state_dict(torch.load(layer_3_path))

    print("Training the third layer")
    for epoch in range(680):
        # train one loose and one tight
        loose = (epoch%2 == 0)
        print("Epoch #:", epoch, loose)
        perf_train = np.array([0.0,0.0,0.0])
        for data,targets in train_loader:
            perf_train += train_rl(mozafari, data, targets, loose)
        perf_train /= len(train_loader)
        if best_train[0] <= perf_train[0]:
            best_train = np.append(perf_train, epoch)
        print("Current Train:", perf_train)
        print("   Best Train:", best_train)

        perf_test = np.array([0.0,0.0,0.0])
        perf_test_loose = np.array([0.0,0.0,0.0])
        for data,targets in test_loader:
            perf_test += test(mozafari, data, targets, False)
            perf_test_loose += test(mozafari, data, targets, True)
        perf_test /= len(test_loader)
        perf_test_loose /= len(test_loader)

        if best_test[0] <= perf_test[0]:
            best_test = np.append(perf_test, epoch)
            torch.save(mozafari.state_dict(), layer_3_path)
        if best_test_loose[0] <= perf_test_loose[0]:
            best_test_loose = np.append(perf_test_loose, epoch)
            torch.save(mozafari.state_dict(), layer_3l_path)

        print(" Current Test:", perf_test, perf_test_loose)
        print("    Best Test:", best_test, best_test_loose)

        # update adaptive learning rates
        apr_adapt = apr * (perf_train[1] * adaptive_int + adaptive_min)
        anr_adapt = anr * (perf_train[1] * adaptive_int + adaptive_min)
        app_adapt = app * (perf_train[0] * adaptive_int + adaptive_min)
        anp_adapt = anp * (perf_train[0] * adaptive_int + adaptive_min)
        print("  Update learing rates :", apr_adapt, anr_adapt, app_adapt, anp_adapt)
        mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)


def train_mnist():
    s1c1 = get_transforms()

    data_root = "data"
    MNIST_train = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=True, download=True, transform = s1c1))
    MNIST_test = utils.CacheDataset(torchvision.datasets.MNIST(root=data_root, train=False, download=True, transform = s1c1))
    MNIST_loader = DataLoader(MNIST_train, batch_size=1000, shuffle=False)
    MNIST_testLoader = DataLoader(MNIST_test, batch_size=len(MNIST_test), shuffle=False)

    mozafari = MozafariMNIST2018()
    if use_cuda:
        mozafari.cuda()

    # Training The First Layer
    print("Training the first layer")
    if os.path.isfile("saved_l1.net"):
        mozafari.load_state_dict(torch.load("saved_l1.net"))
    else:
        for epoch in range(2):
            print("Epoch", epoch)
            iter = 0
            for data,targets in MNIST_loader:
                print("Iteration", iter)
                train_unsupervise(mozafari, data, 1)
                print("Done!")
                iter+=1
        torch.save(mozafari.state_dict(), "saved_l1.net")
    # Training The Second Layer
    print("Training the second layer")
    if os.path.isfile("saved_l2.net"):
        mozafari.load_state_dict(torch.load("saved_l2.net"))
    else:
        for epoch in range(4):
            print("Epoch", epoch)
            iter = 0
            for data,targets in MNIST_loader:
                print("Iteration", iter)
                train_unsupervise(mozafari, data, 2)
                print("Done!")
                iter+=1
        torch.save(mozafari.state_dict(), "saved_l2.net")

    # initial adaptive learning rates
    apr = mozafari.stdp3.learning_rate[0][0].item()
    anr = mozafari.stdp3.learning_rate[0][1].item()
    app = mozafari.anti_stdp3.learning_rate[0][1].item()
    anp = mozafari.anti_stdp3.learning_rate[0][0].item()

    adaptive_min = 0
    adaptive_int = 1
    apr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * apr
    anr_adapt = ((1.0 - 1.0 / 10) * adaptive_int + adaptive_min) * anr
    app_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * app
    anp_adapt = ((1.0 / 10) * adaptive_int + adaptive_min) * anp

    # perf
    best_train = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch
    best_test = np.array([0.0,0.0,0.0,0.0]) # correct, wrong, silence, epoch

    # Training The Third Layer
    print("Training the third layer")
    for epoch in range(680):
        print("Epoch #:", epoch)
        perf_train = np.array([0.0,0.0,0.0])
        for data,targets in MNIST_loader:
            perf_train_batch = train_rl(mozafari, data, targets)
            print(perf_train_batch)
            #update adaptive learning rates
            apr_adapt = apr * (perf_train_batch[1] * adaptive_int + adaptive_min)
            anr_adapt = anr * (perf_train_batch[1] * adaptive_int + adaptive_min)
            app_adapt = app * (perf_train_batch[0] * adaptive_int + adaptive_min)
            anp_adapt = anp * (perf_train_batch[0] * adaptive_int + adaptive_min)
            mozafari.update_learning_rates(apr_adapt, anr_adapt, app_adapt, anp_adapt)
            perf_train += perf_train_batch
        perf_train /= len(MNIST_loader)
        if best_train[0] <= perf_train[0]:
            best_train = np.append(perf_train, epoch)
        print("Current Train:", perf_train)
        print("   Best Train:", best_train)

        for data,targets in MNIST_testLoader:
            perf_test = test(mozafari, data, targets)
            if best_test[0] <= perf_test[0]:
                best_test = np.append(perf_test, epoch)
                torch.save(mozafari.state_dict(), "saved.net")
            print(" Current Test:", perf_test)
            print("    Best Test:", best_test)


if __name__ == "__main__":
    train_heading()
    # train_mnist()
