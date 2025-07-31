import torch.nn as nn
from timm.models.swin_transformer import SwinTransformer

class SwinBackbone(nn.Module):
    def __init__(self, img_size=640, in_chans=3, embed_dim=96, depths=(2, 2, 6, 2), num_heads=(3, 6, 12, 24)):
        super().__init__()
        self.model = SwinTransformer(
            img_size=img_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            depths=depths,
            num_heads=num_heads,
            window_size=10,
            drop_path_rate=0.2,
            num_classes=0  # no classification head
        )

    def forward(self, x):
        return self.model.forward_features(x)  # return last-stage feature map
