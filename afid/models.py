"""AFID models: the shared frozen ConvNeXtV2-T backbone and the heads on top of it."""
import timm
import torch
import torch.nn.functional as F
from torch import nn
from timm.layers import LayerNorm2d

DIMS = [96, 192, 384, 768]
EMBEDDING_DIM = DIMS[-1]
MINUTIAE_CH = 12     # angular bins, 30 deg each
PROJ_DIM = 128
NUM_EXPERTS = 10


class ConvNeXtV2(nn.Module):
    """Backbone: returns the four stage maps."""

    def __init__(self):
        super().__init__()
        m = timm.create_model("convnextv2_tiny", pretrained=False, num_classes=0)
        assert [m.feature_info[i]["num_chs"] for i in range(4)] == DIMS
        self.stem = m.stem
        self.stages = m.stages

    def forward(self, x):
        x = self.stem(x)
        stages = []
        for stage in self.stages:
            x = stage(x)
            stages.append(x)
        return stages


class GeM(nn.Module):
    """Generalized-mean pooling."""

    def __init__(self, p=3.0, eps=1e-6):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p), x.shape[-2:]).pow(1. / self.p)


class GeMHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.global_pool = GeM()
        self.bn = nn.BatchNorm1d(EMBEDDING_DIM)

    def forward(self, x):
        return self.bn(self.global_pool(x).flatten(1))


class AFIDEncoder(nn.Module):
    """Backbone + GeM + BatchNorm -> raw 768-d embedding."""

    def __init__(self):
        super().__init__()
        self.encoder = ConvNeXtV2()
        self.head = GeMHead()

    def embed(self, stages):
        """stages: backbone output (list of stage maps) -> [B, 768]."""
        return self.head(stages[-1])

    def forward(self, x):
        return self.embed(self.encoder(x))


class DecoderHead(nn.Module):
    """U-Net-like decoder over all four backbone stages, output at input resolution."""

    def __init__(self, out_channels):
        super().__init__()
        self.upsample_layers = nn.ModuleList()
        for i in reversed(range(len(DIMS))):
            c = DIMS[i] // 2
            self.upsample_layers.append(nn.Sequential(
                nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
                nn.Conv2d(DIMS[i] * 2, c, kernel_size=3, padding=1),
                LayerNorm2d(c, eps=1e-6),
                nn.GELU(),
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                LayerNorm2d(c, eps=1e-6),
                nn.GELU(),
            ))
        self.out_conv = nn.Conv2d(DIMS[0] // 2, out_channels, kernel_size=1)
        self.final_up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)

    def forward(self, stages):
        x = stages[-1]
        for i, layer in enumerate(self.upsample_layers):
            x = layer(torch.cat([x, stages[-(i + 1)]], dim=1))
        return self.final_up(self.out_conv(x))


class AFIDX(nn.Module):
    """AFID-X: orientation (2 ch), minutiae (12 ch) and segmentation (1 ch) decoders."""

    def __init__(self):
        super().__init__()
        self.orientation_head = DecoderHead(2)
        self.minutiae_head = DecoderHead(MINUTIAE_CH)
        self.segmentation_head = DecoderHead(1)

    def forward(self, stages):
        return (self.orientation_head(stages),
                self.minutiae_head(stages),
                self.segmentation_head(stages))


class AFIDQ(nn.Module):
    """AFID-Q: per-stage GeM + projection -> MLP -> 10 expert scores in [0, 1]."""

    def __init__(self):
        super().__init__()
        self.gem_pools = nn.ModuleList([GeM() for _ in DIMS])
        self.projections = nn.ModuleList([nn.Linear(d, PROJ_DIM, bias=False) for d in DIMS])
        self.quality_head = nn.Sequential(
            nn.Linear(len(DIMS) * PROJ_DIM, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.PReLU(),
            nn.Linear(256, NUM_EXPERTS),
            nn.Sigmoid(),
        )

    def forward(self, stages):
        pooled = [proj(pool(f).flatten(1))
                  for pool, proj, f in zip(self.gem_pools, self.projections, stages)]
        return self.quality_head(torch.cat(pooled, dim=1))

