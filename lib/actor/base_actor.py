"""
-*- coding: utf-8 -*-
@Author : Leezed
@Time : 2024/4/12 16:31
"""
from lib.common.tensor import TensorDict


class BaseActor:
    def __init__(self, net, objective):
        self.net = net
        self.objective = objective

    def __call__(self, data: TensorDict):
        """
        Called in each training iteration. Should pass in input data through the network, calculate the loss, and return the training stats for the input data
        :param data: A TensorDict containing all the necessary data blocks.
        :return:
            loss -loss for the input data
            stats - a dict containing detailed losses
        """
        raise NotImplementedError

    def to(self, device):
        self.net.to(device)

    def train(self, mode=True):
        self.net.train(mode)

    def eval(self):
        self.train(False)
