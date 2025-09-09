from timm.models.layers import to_2tuple, trunc_normal_
import math
from kan import KANLinear
import torch
import torch.nn as nn


class KANLayer(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., no_kan=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.dim = in_features

        grid_size = 5
        spline_order = 3
        scale_noise = 0.1
        scale_base = 1.0
        scale_spline = 1.0
        base_activation = torch.nn.SiLU
        grid_eps = 0.02
        grid_range = [-1, 1]

        if not no_kan:
            self.fc1 = KANLinear(
                in_features,
                hidden_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc2 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )
            self.fc3 = KANLinear(
                hidden_features,
                out_features,
                grid_size=grid_size,
                spline_order=spline_order,
                scale_noise=scale_noise,
                scale_base=scale_base,
                scale_spline=scale_spline,
                base_activation=base_activation,
                grid_eps=grid_eps,
                grid_range=grid_range,
            )

        else:
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.fc2 = nn.Linear(hidden_features, out_features)
            self.fc3 = nn.Linear(hidden_features, out_features)

        self.dwconv_1 = DW_bn_relu(hidden_features)
        self.dwconv_2 = DW_bn_relu(hidden_features)
        self.dwconv_3 = DW_bn_relu(hidden_features)

        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # pdb.set_trace()
        B, N, C = x.shape

        x = self.fc1(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_1(x, H, W)
        x = self.fc2(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_2(x, H, W)
        x = self.fc3(x.reshape(B * N, C))
        x = x.reshape(B, N, C).contiguous()
        x = self.dwconv_3(x, H, W)

        return x


class KANBlock(nn.Module):
    def __init__(self, dim, drop=0., act_layer=nn.GELU, no_kan=False):
        super().__init__()
        mlp_hidden_dim = int(dim)
        self.layer = KANLayer(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop,
                              no_kan=no_kan)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.layer(x, H, W)
        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class DW_bn_relu(nn.Module):
    def __init__(self, dim=768):
        super(DW_bn_relu, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)
        self.relu = nn.ReLU()

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = self.relu(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=7, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.DWT = DWT()  # 下采样
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        x = self.DWT(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class PatchEmbed0(nn.Module):
    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, H, W


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetConvBlock, self).__init__()
        block = [
            nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False),
            nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=False)
        ]
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, slope=0.2):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, slope)

    def forward(self, x, skip):
        up = self.up(x)
        out = torch.cat([up, skip], 1)
        out = self.conv_block(out)
        return out


def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer


class UWKAN(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, slope=0.2, img_size=256, drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, depths=[1, 1, 1]):
        super(UWKAN, self).__init__()
        filters = [16, 32, 64, 128, 256, 512]
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        # Encoder Path
        self.encoder1 = UNetConvBlock(in_channels, filters[0], slope)
        self.avg_pool0 = nn.AvgPool2d(kernel_size=2)
        self.encoder2 = UNetConvBlock(filters[0], filters[1], slope)
        self.avg_pool1 = nn.AvgPool2d(kernel_size=2)
        self.encoder3 = UNetConvBlock(filters[1], filters[2], slope)
        # KAN Stage
        self.block1 = nn.ModuleList([
            KANBlock(dim=filters[3], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        ])

        self.block2 = nn.ModuleList([
            KANBlock(dim=filters[4], drop=drop_rate, drop_path=dpr[1], norm_layer=norm_layer)
        ])

        self.dblock1 = nn.ModuleList([
            KANBlock(dim=filters[3], drop=drop_rate, drop_path=dpr[0], norm_layer=norm_layer)
        ])
        self.patch_embed3 = PatchEmbed0(img_size=img_size // 4, patch_size=3, stride=2, in_chans=filters[2],
                                        embed_dim=filters[3])

        self.patch_embed4 = PatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=filters[3],
                                       embed_dim=filters[4])

        self.decoder1 = IWT()
        self.channel_adjust = nn.Conv2d(filters[4], filters[3], kernel_size=1)
        self.up_concat1 = UNetUpBlock(filters[3], filters[2], slope)
        self.up_concat2 = UNetUpBlock(filters[2], filters[1], slope)
        self.up_concat3 = UNetUpBlock(filters[1], filters[0], slope)

        self.last = nn.Conv2d(filters[0], out_channels, kernel_size=1, bias=True)

    def forward(self, x):
        B = x.size(0)

        # Encoder path
        X_00 = self.encoder1(x)
        maxpool00 = self.avg_pool0(X_00)
        X_10 = self.encoder2(maxpool00)
        maxpool10 = self.avg_pool1(X_10)
        X_20 = self.encoder3(maxpool10)

        # KAN stage
        t4, H, W = self.patch_embed3(X_20)
        for i, blk in enumerate(self.block1):
            t4 = blk(t4, H, W)
        t4 = t4.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out, H, W = self.patch_embed4(t4)
        for i, blk in enumerate(self.block2):
            out = blk(out, H, W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = self.channel_adjust(out)
        out = self.decoder1(out)

        out = torch.add(out, t4)
        _, _, H, W = out.shape
        out = out.flatten(2).transpose(1, 2)
        for i, blk in enumerate(self.dblock1):
            out = blk(out, H, W)
        out = out.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        out = self.up_concat1(out, X_20)
        out = self.up_concat2(out, X_10)
        out = self.up_concat3(out, X_00)

        final = self.last(out)
        return final
