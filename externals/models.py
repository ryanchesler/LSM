from pytorch3dunet.unet3d.model import get_model
from torch import nn
    
class Unet3D_full3d_shallow(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        self.model = get_model({"name": "UNet3D", "in_channels": 1, "out_channels": 1,
                               "f_maps": 32, "num_groups": 4, "is_segmentation": False, "num_levels":5})

    def forward(self, volume):
        output = self.model(volume)
        return output