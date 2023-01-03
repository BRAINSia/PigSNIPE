from abc import ABC
from monai.networks.nets import UNet
from monai.networks.layers import Norm
import pytorch_lightning as pl


class LowResBrainmaskModel(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        # initialize network architecture
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 128),
            strides=(2, 2, 2, 2),
            num_res_units=2,
            norm=Norm.BATCH,
        )

    def forward(self, x):
        return self.model(x)


class HighResBrainmaskModel(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        # initialize network architecture
        self.model = UNet(
            spatial_dims=3,
            in_channels=1,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            norm=Norm.BATCH,
        )

    def forward(self, x):
        return self.model(x)


class ICVModel(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        # initialize network architecture
        self.model = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=2,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            norm=Norm.BATCH,
        )

    def forward(self, x):
        return self.model(x)


class GWCModel(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        # initialize network architecture
        self.model = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=4,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            norm=Norm.BATCH,
        )

    def forward(self, x):
        return self.model(x)


class SegModel(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()
        # initialize network architecture
        self.model = UNet(
            spatial_dims=3,
            in_channels=2,
            out_channels=5,
            channels=(16, 32, 64, 128, 256),
            strides=(2, 2, 2, 2),
            num_res_units=3,
            norm=Norm.BATCH,
        )

    def forward(self, x):
        return self.model(x)
