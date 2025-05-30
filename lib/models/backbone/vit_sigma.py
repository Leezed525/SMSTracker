from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import Mlp, DropPath
from lib.utils.backbone_utils import combine_tokens
from lib.models.layer.patch_embed import PatchEmbed
from .vit import VisionTransformer
from lib.models.layer.vmamba import CrossMambaFusionBlock, ConcatMambaFusionBlock
from lib.models.layer.score import ScoreAttention


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.drop_key_ratio = 0.1

    def forward(self, x, return_attention=False):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # drop key
        if self.training:
            m_r = torch.ones_like(attn) * self.drop_key_ratio
            attn = attn + torch.bernoulli(m_r) * -1e12

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False):
        if return_attention:
            feat, attn = self.attn(self.norm1(x), True)
            x = x + self.drop_path(feat)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, attn
        else:
            x = x + self.drop_path(self.attn(self.norm1(x)))
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x


class VisionTransformerP(VisionTransformer):
    """ Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929
    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', search_size=None, template_size=None, new_patch_size=None):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search = new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template = new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        """ sigma blocks   """

        self.cross_mamba = nn.ModuleList(
            CrossMambaFusionBlock(
                hidden_dim=self.embed_dim,
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(1)
        )
        self.channel_attn_mamba = nn.ModuleList(
            ConcatMambaFusionBlock(
                hidden_dim=self.embed_dim,
                mlp_ratio=0.0,
                d_state=4,
            ) for i in range(1)
        )
        self.sigma_norm = norm_layer(embed_dim)

        # score layer
        # self.score = nn.ModuleList(
        #     ScoreAttention(embed_dim=self.embed_dim) for i in range(13)
        # )
        self.score = ScoreAttention(embed_dim=self.embed_dim, num_heads=num_heads // 2)

        self.init_weights(weight_init)

    def forward_features(self, z, x, mask_z=None, mask_x=None,
                         return_last_attn=False
                         ):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        # rgb_img
        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        # depth_img
        x_modal = x[:, 3:, :, :]
        z_modal = z[:, 3:, :, :]
        # overwrite x & z
        x, z = x_rgb, z_rgb

        x, _ = self.patch_embed(x)  # (B, 16*16,768)
        z, _ = self.patch_embed(z)  # (B, 8*8,768)

        x_modal, _ = self.patch_embed(x_modal)
        z_modal, _ = self.patch_embed(z_modal)

        zw = zh = int(z.shape[1] ** 0.5)
        xw = xh = int(x.shape[1] ** 0.5)

        # attention mask handling
        # B, H, W
        if mask_z is not None and mask_x is not None:
            mask_z = F.interpolate(mask_z[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_z = mask_z.flatten(1).unsqueeze(-1)

            mask_x = F.interpolate(mask_x[None].float(), scale_factor=1. / self.patch_size).to(torch.bool)[0]
            mask_x = mask_x.flatten(1).unsqueeze(-1)

            mask_x = combine_tokens(mask_z, mask_x, mode=self.cat_mode)
            mask_x = mask_x.squeeze(-1)

        if self.add_cls_token:
            cls_tokens = self.cls_token.expand(B, -1, -1)
            cls_tokens = cls_tokens + self.cls_pos_embed

        z = z + self.pos_embed_z
        x = x + self.pos_embed_x
        z_modal = z_modal + self.pos_embed_z
        x_modal = x_modal + self.pos_embed_x

        # score function start

        mx = self.score(x, x_modal)  # (B, 16*16,768)
        mz = self.score(z, z_modal)  # (B, 8*8,768)

        # score function end

        if self.add_sep_seg:
            x += self.search_segment_pos_embed
            z += self.template_segment_pos_embed

        x = combine_tokens(z, x, mode=self.cat_mode)
        x_modal = combine_tokens(z_modal, x_modal, mode=self.cat_mode)
        mx = combine_tokens(mz, mx, mode=self.cat_mode)
        # mx = torch.zeros_like(x)

        if self.add_cls_token:
            x = torch.cat([cls_tokens, x], dim=1)
            x_modal = torch.cat([cls_tokens, x_modal], dim=1)

        x = self.pos_drop(x)
        x_modal = self.pos_drop(x_modal)
        mx = self.pos_drop(mx)

        lens_z = self.pos_embed_z.shape[1]
        lens_x = self.pos_embed_x.shape[1]

        for i, blk in enumerate(self.blocks):
            x = blk(x)
            x_modal = blk(x_modal)
            # score fusion
            mx = mx + self.score(x, x_modal)

            """sigma fusion"""
            # if i % 2 == 1: # 6sigma
            # if i % 3 == 2: # 4sigma
            # if i % 4 == 3: # 3sigma
            # if i % 6 == 5: # 2sigma
            if i % 12 == 11:  # 1sigma
                # sigma fusion
                x, z = self.token2wh(x, xw, xh, zw, zh, B)  # x -> (B, 16,16, 768) z - > (B, 8,8, 768)
                x_modal, z_modal = self.token2wh(x_modal, xw, xh, zw, zh, B)
                mx, mz = self.token2wh(mx, xw, xh, zw, zh, B)

                x, x_modal = self.cross_mamba[i // 12](x, x_modal)
                x_fuse = self.channel_attn_mamba[i // 12](x, x_modal)
                mx = mx + x_fuse

                z, z_modal = self.cross_mamba[i // 12](z, z_modal)
                z_fuse = self.channel_attn_mamba[i // 12](z, z_modal)
                mz = mz + z_fuse

                x = self.wh2token(x, z, xw, xh, zw, zh, B)
                x_modal = self.wh2token(x_modal, z_modal, xw, xh, zw, zh, B)
                mx = self.wh2token(mx, mz, xw, xh, zw, zh, B)

                # 加个norm防止过拟合
                x = self.sigma_norm(x)
                x_modal = self.sigma_norm(x_modal)
                mx = self.sigma_norm(mx)

        x = x + mx + x_modal
        # x = x  # id01
        # x = mx # id02
        # x = x_modal  # id03
        # x = x + mx # id04
        # x = mx + x_modal # id05
        # x = x + x_modal # id06
        x = self.norm(x)
        aux_dict = {"attn": None}
        return x, aux_dict

    def forward(self, z, x, ce_template_mask=None, ce_keep_rate=None,
                tnc_keep_rate=None,
                return_last_attn=False):

        x, aux_dict = self.forward_features(z, x)

        return x, aux_dict

    def token2wh(self, token, xw, xh, zw, zh, B):
        return token[:, zw * zh:, :].reshape(B, xw, xh, -1), token[:, :zw * zh, :].reshape(B, zw, zh, -1)

    def wh2token(self, x, z, xw, xh, zw, zh, B):
        x = x.reshape(B, xw * xh, -1)
        z = z.reshape(B, zw * zh, -1)
        return combine_tokens(z, x, mode=self.cat_mode)


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerP(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print("missing_keys")
            print(missing_keys)
            print("unexpected_keys")
            print(unexpected_keys)
            print('Load pretrained model from: ' + pretrained)

    return model


def vit_base_patch16_224(pretrained=False, **kwargs):
    """
    ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
