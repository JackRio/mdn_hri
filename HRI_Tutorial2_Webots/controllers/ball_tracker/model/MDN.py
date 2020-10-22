from abc import ABC

import torch
import torch.nn as nn


class MDN(nn.Module, ABC):
    def __init__(self, n_input, n_hidden, n_output, n_gaussian):
        super(MDN, self).__init__()

        self.hidden = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Tanh()
        )

        self.pi = nn.Sequential(
            nn.Linear(n_hidden, n_gaussian),
            nn.Softmax(dim=1)
        )

        self.sigma = nn.Linear(n_hidden, n_gaussian)
        self.mu = nn.Linear(n_hidden, n_output * n_gaussian)

    def forward(self, x):
        hidden = self.hidden(x)
        pi = self.pi(hidden)
        mu = self.mu(hidden)
        sigma = torch.exp(self.sigma(hidden))

        return pi, sigma, mu
