# coding=utf-8

import numpy as np
import torch


def Variable(tensor):
    if isinstance(tensor, np.ndarray):
        tensor = torch.from_numpy(tensor)
    if torch.cuda.is_available():
        return torch.autograd.Variable(tensor).cuda()
    return torch.autograd.Variable(tensor)


def NLLLoss(inputs, targets):
    """
    Custom Negative Log Likelihood loss that returns loss per example,
        rather than for the entire batch.

    :param inputs: (batch_size, num_classes) *Log probabilities of each class*.
    :param targets: (batch_size) *Target class index*.
    :return: loss : (batch_size) *Loss for each example*.
    """

    if torch.cuda.is_available():
        target_expanded = torch.zeros(inputs.size()).cuda()
    else:
        target_expanded = torch.zeros(inputs.size())

    target_expanded.scatter_(1, targets.contiguous().view(-1, 1).data, 1.0)
    loss = Variable(target_expanded) * inputs
    loss = torch.sum(loss, 1)
    return loss


def read_smi_file(file_path):
    """
    Reads a SMILES file.
    :param file_path: Path to a SMILES file.
    :return: A list with all the SMILES.
    """
    with open(file_path, "r") as smi_file:
        return [smi.rstrip().split()[0] for smi in smi_file]
