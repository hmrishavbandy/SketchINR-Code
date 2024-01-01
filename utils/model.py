import torch
from torch import nn
import math
import torch.nn.functional as F

class DeepSDF(nn.Module):
    def __init__(
        self,
        latent_size =  512,
        dims = [ 512, 512, 512, 512, 512],
        dropout=[0, 1, 2, 3],
        dropout_prob=0.0,
        norm_layers=[0, 1, 2, 3],
        latent_in=(),
        weight_norm=True,
        xyz_in_all=True,
        use_tanh=False,
        latent_dropout=False,
        positional_encoding = False,
        fourier_degree = 1,
        num_tensors=2,
        th=False,
    ):
        super(DeepSDF, self).__init__()

        def make_sequence():
            return []
        if positional_encoding is True:
            dims = [latent_size + 2*fourier_degree*3] + dims + [1]
        else:
            dims = [latent_size + 8*num_tensors] + dims + [2]

        self.positional_encoding = positional_encoding
        self.fourier_degree = fourier_degree
        self.num_layers = len(dims)
        self.norm_layers = norm_layers
        self.latent_in = latent_in
        self.latent_dropout = latent_dropout
        if self.latent_dropout:
            self.lat_dp = nn.Dropout(0.2)

        self.xyz_in_all = xyz_in_all
        self.weight_norm = weight_norm

        for layer in range(0, self.num_layers - 1):
            if layer + 1 in latent_in:
                out_dim = dims[layer + 1] - dims[0]
            else:
                out_dim = dims[layer + 1]
                if self.xyz_in_all and layer != self.num_layers - 2:
                    out_dim -= 8*num_tensors

            if weight_norm and layer in self.norm_layers:
                setattr(
                    self,
                    "lin" + str(layer),
                    nn.utils.weight_norm(nn.Linear(dims[layer], out_dim)),
                )
            else:
                setattr(self, "lin" + str(layer), nn.Linear(dims[layer], out_dim))

            if (
                (not weight_norm)
                and self.norm_layers is not None
                and layer in self.norm_layers
            ):
                setattr(self, "bn" + str(layer), nn.LayerNorm(out_dim))

        self.use_tanh = use_tanh
        if use_tanh:
            self.tanh = nn.Tanh()
        self.relu = nn.ReLU()

        self.dropout_prob = dropout_prob
        self.dropout = dropout
        if th: 
            self.th = nn.Tanh()
        else:
            self.th = nn.Sigmoid()
        self.num_tensors = num_tensors

 
    def fourier_transform(self,x, L=5):
        cosines = torch.cat([torch.cos(2**l*3.1415*x) for l in range(L)], -1)
        sines = torch.cat([torch.sin(2**l*3.1415*x) for l in range(L)], -1)
        transformed_x = torch.cat((cosines,sines),-1)
        return transformed_x
    
    def forward(self, input):
        x = input
        xyz = input[:, :8*self.num_tensors]
        for layer in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(layer))
            if layer in self.latent_in:
                x = torch.cat([x, input], 1)
            elif layer != 0 and self.xyz_in_all:
                x = torch.cat([x, xyz], 1)
            x = lin(x)
            if layer == self.num_layers - 2 and self.use_tanh:
                x = self.tanh(x)
            if layer < self.num_layers - 2:
                if (
                    self.norm_layers is not None
                    and layer in self.norm_layers
                    and not self.weight_norm
                ):
                    bn = getattr(self, "bn" + str(layer))
                    x = bn(x)
                x = self.relu(x)
                if self.dropout is not None and layer in self.dropout:
                    x = F.dropout(x, p=self.dropout_prob, training=self.training)

        if hasattr(self, "th"):
            x = self.th(x)

        return x

