import torch
import torch.nn as nn
import numpy as np
from model import Generator, ContentDiscriminator, DomainDiscriminator


def train():
    generator = Generator()
    generator.cuda()
    generator.train()

    