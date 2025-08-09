import torch
import torch.nn as nn

# Reference: https://github.com/moskomule/senet.pytorch


class SEBlock(nn.Module):
    """Squeeze and Excitation block"""

    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y))
        y = y.view(b, c, 1, 1, 1)
        return x * y


class DHWABlock(nn.Module):
    """Depth-Height-Weight 3D Attention Block with Bottleneck"""

    def __init__(self, channels: int, reduction: int = 16, init_scale: float = 5.0):
        super().__init__()
        self.depth_mlp = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
        )
        self.height_mlp = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
        )
        self.width_mlp = nn.Sequential(
            nn.Conv1d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels // reduction, channels, 1, bias=False),
        )
        self.gamma = nn.Parameter(torch.full([], init_scale))

    @staticmethod
    def _pool_sum(x: torch.Tensor, dim: int) -> torch.Tensor:
        return x.mean(dim=dim) + x.amax(dim=dim)

    def _depth_attn(self, x: torch.Tensor) -> torch.Tensor:
        y = self._pool_sum(x, dim=2)
        y = self.depth_mlp(y.view(y.shape[0], y.shape[1], -1))
        y = torch.sigmoid(y).view(x.shape[0], x.shape[1], 1, x.shape[3], x.shape[4])
        return y

    def _height_attn(self, x: torch.Tensor) -> torch.Tensor:
        y = self._pool_sum(x, dim=3)
        y = self.height_mlp(y.view(y.shape[0], y.shape[1], -1))
        y = torch.sigmoid(y).view(x.shape[0], x.shape[1], x.shape[2], 1, x.shape[4])
        return y

    def _width_attn(self, x: torch.Tensor) -> torch.Tensor:
        y = self._pool_sum(x, dim=4)
        y = self.width_mlp(y.view(y.shape[0], y.shape[1], -1))
        y = torch.sigmoid(y).view(x.shape[0], x.shape[1], x.shape[2], x.shape[3], 1)
        return y

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        aD = self._depth_attn(x)
        aH = self._height_attn(x)
        aW = self._width_attn(x)
        attn = aD * aH * aW * self.gamma
        return x * attn


class ResidualSEBlock(nn.Module):
    """Block for ResNet3D with SE attention"""

    expansion = 1

    def __init__(
        self,
        in_channels,
        out_channels,
        stride=1,
        downsample=None,
        reduction=16,
        drop_rate=0.0,
    ):
        super(ResidualSEBlock, self).__init__()
        self.conv1 = nn.Conv3d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SEBlock(out_channels, reduction=reduction)
        self.dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else None
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        if self.dropout is not None:
            out = self.dropout(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        out = self.relu(out)
        return out


class GeM(nn.Module):
    def __init__(
        self,
        p: float = 3.0,
        eps: float = 1e-6,
        learn_p: bool = True,
        p_min: float = 1.0,
        p_max: float = 6.0,
    ):
        super().__init__()
        if learn_p:
            self.p = nn.Parameter(torch.ones(1) * p)
        else:
            self.register_buffer("p", torch.tensor([p]))
        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p_clamped = torch.clamp(self.p, self.p_min, self.p_max)
        x = x.clamp(min=self.eps)
        x = x.pow(p_clamped)
        x = x.mean(dim=(-1, -2, -3))
        x = x.pow(1.0 / p_clamped)
        return x


class MultiPoolHead(nn.Module):
    def __init__(
        self,
        in_channels: int,
        proj_hidden: int = 512,
        emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool3d(1)
        self.gmp = nn.AdaptiveMaxPool3d(1)
        self.gem = GeM(p=3.0, learn_p=True)

        proj_layers = [
            nn.Linear(in_channels * 3, proj_hidden, bias=False),
            nn.BatchNorm1d(proj_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(proj_hidden, emb_dim, bias=False),
            nn.LayerNorm(emb_dim),
        ]
        if dropout > 0:
            proj_layers.insert(3, nn.Dropout(dropout))
        self.fc = nn.Sequential(*proj_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        g_avg = self.gap(x).flatten(1)
        g_max = self.gmp(x).flatten(1)
        g_gem = self.gem(x)
        feats = torch.cat([g_avg, g_max, g_gem], dim=1)
        z = self.fc(feats)
        return z


class ResidualSEEncoder(nn.Module):
    """3D ResNet encoder with SE attention"""

    def __init__(
        self,
        block=ResidualSEBlock,
        layers=(1, 2, 2, 2, 1, 1),
        num_channels=6,
        channels=(64, 128, 192, 256, 384, 512),
        layer_strides=(1, 2, 2, 2, 1, 1),
        proj_hidden_dim=4096,
        emb_dim=128,
        reduction=16,
        drop_rate=0.1,
    ):
        super().__init__()
        self.in_channels = channels[0]
        # 7x7x7 conv -> BN -> ReLU -> 3x3x3 max pool
        self.conv1 = nn.Conv3d(
            num_channels,
            self.in_channels,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=1, padding=1)
        # Residual layers
        # layer1: output channels = channels[0], stride=1
        self.layer1 = self._make_layer(
            block,
            channels[0],
            layers[0],
            layer_strides[0],
            reduction=reduction,
            drop_rate=drop_rate,
        )
        # layer2: output channels = channels[1], stride=2 (downsample)
        self.layer2 = self._make_layer(
            block,
            channels[1],
            layers[1],
            layer_strides[1],
            reduction=reduction,
            drop_rate=drop_rate,
        )
        # layer3: output channels = channels[2], stride=2
        self.layer3 = self._make_layer(
            block,
            channels[2],
            layers[2],
            layer_strides[2],
            reduction=reduction,
            drop_rate=drop_rate,
        )
        self.attn3 = DHWABlock(channels[2], reduction)
        # layer4: output channels = channels[3], stride=2
        self.layer4 = self._make_layer(
            block,
            channels[3],
            layers[3],
            layer_strides[3],
            reduction=reduction,
            drop_rate=drop_rate,
        )
        self.layer5 = self._make_layer(
            block,
            channels[4],
            layers[4],
            layer_strides[4],
            reduction=reduction,
            drop_rate=drop_rate,
        )
        self.layer6 = self._make_layer(
            block,
            channels[5],
            layers[5],
            layer_strides[5],
            reduction=reduction,
            drop_rate=drop_rate,
        )
        self.attn6 = DHWABlock(channels[5] * block.expansion, reduction)
        # self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.emb_dropout = nn.Dropout(drop_rate) if drop_rate > 0.0 else None
        # self.fc = nn.Sequential(
        #     nn.Linear(channels[3] * block.expansion, proj_hidden_dim),
        #     nn.ReLU(inplace=True),
        #     nn.Linear(proj_hidden_dim, emb_dim),
        #     nn.LayerNorm(emb_dim),
        # )
        self.fc = MultiPoolHead(channels[5] * block.expansion, proj_hidden_dim, emb_dim)

    def _make_layer(
        self, block, out_channels, blocks, stride=1, reduction=16, drop_rate=0.0
    ):
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv3d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm3d(out_channels * block.expansion),
            )
        layers = []
        layers.append(
            block(
                self.in_channels,
                out_channels,
                stride=stride,
                downsample=downsample,
                reduction=reduction,
                drop_rate=drop_rate,
            )
        )
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.in_channels,
                    out_channels,
                    stride=1,
                    downsample=None,
                    reduction=reduction,
                    drop_rate=drop_rate,
                )
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        # Input shape: (batch, 4, D, H, W)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.attn3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.attn6(x)
        # # Global average pool to (batch, channels, 1, 1, 1)
        # x = self.global_pool(x)
        # # flatten to (batch, channels)
        # x = torch.flatten(x, 1)
        if self.emb_dropout is not None:
            x = self.emb_dropout(x)
        x = self.fc(x)
        return x
