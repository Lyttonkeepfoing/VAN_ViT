from torch.nn import Module, ModuleList
from torch.nn import Conv2d, InstanceNorm2d, Dropout, Dropout2d
from torch.nn import ReLU
from torch.nn.functional import pad
import random


class DepthSepConv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, activation=None, padding=True, stride=(1, 1), dilation=(1, 1)):
        super(DepthSepConv2D, self).__init__()

        self.padding = None

        if padding:
            if padding is True:
                padding = [int((k - 1) / 2) for k in kernel_size]
                if kernel_size[0] % 2 == 0 or kernel_size[1] % 2 == 0:
                    padding_h = kernel_size[1] - 1
                    padding_w = kernel_size[0] - 1
                    self.padding = [padding_h//2, padding_h-padding_h//2, padding_w//2, padding_w-padding_w//2]
                    padding = (0, 0)

        else:
            padding = (0, 0)
        self.depth_conv = Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, dilation=dilation, stride=stride, padding=padding, groups=in_channels)
        self.point_conv = Conv2d(in_channels=in_channels, out_channels=out_channels, dilation=dilation, kernel_size=(1, 1))
        self.activation = activation

    def forward(self, x):
        x = self.depth_conv(x)
        if self.padding:
            x = pad(x, self.padding)
        if self.activation:
            x = self.activation(x)
        x = self.point_conv(x)
        return x


class MixDropout(Module):
    def __init__(self, dropout_proba=0.4, dropout2d_proba=0.2):
        super(MixDropout, self).__init__()

        self.dropout = Dropout(dropout_proba)
        self.dropout2d = Dropout2d(dropout2d_proba)

    def forward(self, x):
        if random.random() < 0.5:
            return self.dropout(x)
        return self.dropout2d(x)




class ConvBlock(Module):

    def __init__(self, in_, out_, stride=(1, 1), k=3, activation=ReLU, dropout=0.4):
        super(ConvBlock, self).__init__()

        self.activation = activation()
        self.conv1 = Conv2d(in_channels=in_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv2 = Conv2d(in_channels=out_, out_channels=out_, kernel_size=k, padding=k // 2)
        self.conv3 = Conv2d(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=stride)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout / 2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)
        x = self.activation(x)

        if pos == 3:
            x = self.dropout(x)
        return x


class DSCBlock(Module):

    def __init__(self, in_, out_, pool=(2, 1), activation=ReLU, dropout=0.4):
        super(DSCBlock, self).__init__()

        self.activation = activation()
        self.conv1 = DepthSepConv2D(in_, out_, kernel_size=(3, 3))
        self.conv2 = DepthSepConv2D(out_, out_, kernel_size=(3, 3))
        self.conv3 = DepthSepConv2D(out_, out_, kernel_size=(3, 3), padding=(1, 1), stride=pool)
        self.norm_layer = InstanceNorm2d(out_, eps=0.001, momentum=0.99, track_running_stats=False)
        self.dropout = MixDropout(dropout_proba=dropout, dropout2d_proba=dropout/2)

    def forward(self, x):
        pos = random.randint(1, 3)
        x = self.conv1(x)
        x = self.activation(x)

        if pos == 1:
            x = self.dropout(x)

        x = self.conv2(x)
        x = self.activation(x)

        if pos == 2:
            x = self.dropout(x)

        x = self.norm_layer(x)
        x = self.conv3(x)

        if pos == 3:
            x = self.dropout(x)
        return x
################# vit ###################


import basic.vit as vit
import torch.nn as nn
from functools import partial
import torch
import torch.nn.functional as F


class VisionTransformer(nn.Module):
     """ Vision Transformer """

     def __init__(self,params):
         super().__init__()
         img_size = [224]
         patch_size = 8
         in_chans = 3
         num_classes = 1
         embed_dim = 768
         depth = 12
         num_heads = 12
         mlp_ratio = 4.
         drop_rate = 0.0
         attn_drop_rate = 0.1
         drop_path_rate = 0.1
         norm_layer = nn.LayerNorm
         self.num_features = self.embed_dim = embed_dim

         self.patch_embed = vit.PatchEmbed(img_size=img_size[0],
                                           patch_size=patch_size,
                                           in_chans=in_chans,
                                           embed_dim=embed_dim)
         num_patches = self.patch_embed.num_patches

         self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
         self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=False)
         self.pos_drop = nn.Dropout(p=drop_rate)
         self.patch_size = patch_size
         self.num_classes = num_classes
         self.convex_kernel = 3
         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
         self.blocks = nn.ModuleList([vit.Block(dim=embed_dim,
                                                num_heads=num_heads,
                                                mlp_ratio=mlp_ratio,
                                                qkv_bias=False,
                                                qk_scale=None,
                                                drop=drop_rate,
                                                attn_drop=attn_drop_rate,
                                                drop_path=dpr[i],
                                                norm_layer=norm_layer) for i in range(depth)])
         self.norm = norm_layer(embed_dim)

         # Classifier head
         self.head = nn.Linear(embed_dim, 1) if num_classes > 0 else nn.Identity()
         self.upsampler = nn.Sequential(nn.Conv2d(embed_dim, embed_dim * 2, kernel_size=3, padding=1),
                                        nn.ReLU(inplace=True),
                                        nn.Conv2d(embed_dim * 2,
                                                  self.convex_kernel * self.convex_kernel * self.patch_size * self.patch_size,
                                                  kernel_size=1, padding=0))

         self.final = nn.Sequential(nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
                                    nn.ReLU(inplace=True),
                                    nn.Conv2d(embed_dim // 2, 256, kernel_size=1, padding=0))

         vit.trunc_normal_(self.pos_embed, std=.02)
         vit.trunc_normal_(self.cls_token, std=.02)
         self.apply(vit._init_weights)

     def prepare_tokens(self, x):
         B, nc, w, h = x.shape
         x = self.patch_embed(x)  # patch linear embedding

         # add the [CLS] token to the embed patch tokens
         cls_tokens = self.cls_token.expand(B, -1, -1)
         x = torch.cat((cls_tokens, x), dim=1)

         # add positional encoding to each token
         x = x + vit.interpolate_pos_encoding(self.pos_embed, self.patch_size, x, w, h)

         return self.pos_drop(x)

     def upsample_mask(self, mask, weight):
         """ Upsample mask field [H/8, W/8, 1] -> [H, W, 1] using convex combination """
         N, C, H, W = mask.shape
         weight = weight.view(N, 1, self.convex_kernel * self.convex_kernel, self.patch_size, self.patch_size, H,
                              W).contiguous()
         weight = torch.softmax(weight, dim=2)

         up_mask = F.unfold(mask, [self.convex_kernel, self.convex_kernel], padding=self.convex_kernel // 2)
         up_mask = up_mask.view(N, C, self.convex_kernel * self.convex_kernel, 1, 1, H, W).contiguous()

         up_mask = torch.sum(weight * up_mask, dim=2)
         up_mask = up_mask.permute(0, 1, 4, 2, 5, 3).contiguous()
         return up_mask.reshape(N, C, self.patch_size * H, self.patch_size * W).contiguous()

     def forward(self, x):
         B, nc, w, h = x.shape
         x = self.prepare_tokens(x)
         for blk in self.blocks:
             x = blk(x)
         x = self.norm(x)
         x = x[:, 1:].permute(0, 2, 1).contiguous().reshape(B, self.num_features, w // self.patch_size,
                                                             h // self.patch_size).contiguous()
         # weight = self.upsampler(x)
         # x = self.upsample_mask(x, weight)
         x = self.final(x)
         return x



class FCN_Encoder(Module):
    def __init__(self, params):
        super(FCN_Encoder, self).__init__()

        self.dropout = params["dropout"]

        self.init_blocks = ModuleList([
            ConvBlock(params["input_channels"], 16, stride=(1, 1), dropout=self.dropout),
            ConvBlock(16, 32, stride=(2, 2), dropout=self.dropout),
            ConvBlock(32, 64, stride=(2, 2), dropout=self.dropout),
            ConvBlock(64, 128, stride=(2, 2), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
            ConvBlock(128, 128, stride=(2, 1), dropout=self.dropout),
        ])
        self.blocks = ModuleList([
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 128, pool=(1, 1), dropout=self.dropout),
            DSCBlock(128, 256, pool=(1, 1), dropout=self.dropout),
        ])

    def forward(self, x):
        for b in self.init_blocks:
            x = b(x)
        for b in self.blocks:
            xt = b(x)
            x = x + xt if x.size() == xt.size() else xt
        print(x.shape, '====================')
        return x
