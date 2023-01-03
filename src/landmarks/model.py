import numpy as np
import torch
from monai.networks.layers.factories import Act, Conv, Norm, Pool
from torch import nn


class HardParameterSharingModel(nn.Module):
    def __init__(
        self,
        agents,
        state_size,
        conv_filters=[32, 32, 64, 64],
        conv_strides=[2, 2, 2, 2],
        conv_kernels=[3, 3, 3, 3],
        fc_layers=[512, 256, 128, 6],
        norm="BATCH",
        use_norm=True,
        pooling="MAX",
        pooling_kernels=[2, 2, 2, 2],
        pooling_strides=[1, 1, 1, 1],
        activation="PRELU",
        activation_args=None,
        location_sharing=False,
        gpu=None,
        bias_init=0.01,
        input_channels=4,
        scale_result=1,
    ):
        super(HardParameterSharingModel, self).__init__()

        # get the device
        self.device = "cpu"
        if gpu:
            self.device = f"cuda:{gpu}"

        self.agents = agents
        self.location_sharing = location_sharing
        self.location_vector_size = 0
        self.scale_result = scale_result
        if self.location_sharing:
            self.location_vector_size = (len(agents) - 1) * 3
        # print("location_vector_size", self.location_vector_size)

        if activation_args:
            activation_args = dict(activation_args)

        self.use_norm = use_norm
        self.input_channels = input_channels
        self.conv_type = Conv[Conv.CONV, 3]
        self.act_type = Act[activation]
        self.norm1d_type = Norm[norm, 1]
        self.norm3d_type = Norm[norm, 3]
        self.pool_type = Pool[pooling, 3]

        # define shared convolutions
        self.conv_layers = nn.ModuleList()
        for index, filter_size in enumerate(conv_filters):
            if index == 0:
                self.conv_layers.append(
                    self.conv_type(
                        in_channels=self.input_channels,
                        out_channels=filter_size,
                        kernel_size=conv_kernels[index],
                        stride=1,
                        bias=True,
                    )
                )
            else:
                self.conv_layers.append(
                    self.conv_type(
                        in_channels=conv_filters[index - 1],
                        out_channels=filter_size,
                        kernel_size=conv_kernels[index],
                        stride=conv_strides[index],
                        bias=True,
                    )
                )

            if self.use_norm:
                self.conv_layers.append(self.norm3d_type(filter_size))

            if activation_args:
                self.conv_layers.append(self.act_type(**activation_args))
            else:
                self.conv_layers.append(self.act_type())

            self.conv_layers.append(
                self.pool_type(
                    kernel_size=pooling_kernels[index], stride=pooling_strides[index]
                )
            )

        # compute output size of convolutions
        arr = (
            torch.zeros((self.input_channels, *state_size))
            .unsqueeze(0)
            .to(self.conv_layers[0].weight.device)
        )
        # apply all conv layers
        for l in self.conv_layers:
            arr = l(arr)
        fc_input_size = np.prod(np.array(arr.shape)) + self.location_vector_size

        # define agent specific layers
        self.fc_layers = nn.ModuleList([nn.ModuleList() for _ in range(len(agents))])
        for agent_idx in range(len(agents)):
            for index, n_neurons in enumerate(fc_layers):
                if index == 0:
                    # first conv has input dimension 1
                    self.fc_layers[agent_idx].append(
                        nn.Linear(fc_input_size, n_neurons)
                    )
                else:
                    self.fc_layers[agent_idx].append(
                        nn.Linear(
                            fc_layers[index - 1] + self.location_vector_size,
                            n_neurons,
                        )
                    )

                if self.use_norm:
                    self.fc_layers[agent_idx].append(self.norm1d_type(n_neurons))

                if activation_args:
                    self.fc_layers[agent_idx].append(self.act_type(**activation_args))
                else:
                    self.fc_layers[agent_idx].append(self.act_type())

        # define flatten op
        self.flatten = nn.Flatten()

        # initialize weight and bias values
        for p in self.parameters():
            if type(p) in [self.conv_type, nn.Linear]:
                nn.init.xavier_uniform_(p.weight)
                # nn.init.constant_(p.bias, bias_init)

    def forward(self, x):
        location_vector_batch = None
        if self.location_sharing:
            assert len(x) == 2, "must pass a tuple to do location sharing"
            x, location_vector_batch = x
            location_vector_batch = location_vector_batch.to(self.device)

        x = x.to(self.device)
        # normalize network input 0-255 to 0-1
        x = x / 255.0

        agent_outputs = []
        for agent_idx in range(len(self.agents)):

            tmp = x[:, agent_idx, :, :, :, :]
            for conv_l in self.conv_layers:
                tmp = conv_l(tmp)

            tmp = self.flatten(tmp)

            for fc_l in self.fc_layers[agent_idx]:
                if type(fc_l) == nn.Linear and self.location_sharing:
                    tmp = torch.cat(
                        (tmp, location_vector_batch[:, agent_idx, :]), dim=1
                    )
                tmp = fc_l(tmp)
            agent_outputs.append(tmp)

        return torch.stack(agent_outputs, dim=1) * self.scale_result
