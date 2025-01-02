import torch
import torch.nn as nn
from einops import rearrange
from .st_blocks import STBlock
from .dino_encoder import FrozenDinov2ImageEmbedder

from typing import Tuple
class IGOR_IDM(nn.Module):
    def __init__(
        self,
        d_t: int,
        encoder_depth: int,
        encoder_embed_dim: int,
        encoder_n_heads: int,
        pretrained_dino_size: str,
        mlp_ratio: float = 4.0,
        use_flash_attn: bool = True,
    ) -> None:
        super().__init__()

        self.d_t = d_t 
        self.mlp_ratio = mlp_ratio
        self.encoder_depth = encoder_depth
        self.encoder_embed_dim = encoder_embed_dim
        self.encoder_n_heads = encoder_n_heads

        self.use_flash_attn = use_flash_attn
        
        self.pretrained_dino_size = pretrained_dino_size
        self.dino_encoder = FrozenDinov2ImageEmbedder(pretrained=True,
                                                      freeze=True,
                                                      size=self.pretrained_dino_size)
        self.dino_encoder.eval()

        self.patch2embed = None
        self.num_patches = 256
        if self.pretrained_dino_size == 'base':
            dino_channels = 768
        else:
            raise NotImplementedError        
        self.proj_in = nn.Linear(dino_channels, self.encoder_embed_dim, bias=True) if dino_channels != self.encoder_embed_dim else nn.Identity()

        self.encoder_pe = nn.Parameter(
            torch.zeros(1, self.num_patches, self.encoder_embed_dim),
            requires_grad=False,
        )
        self.encoder_pe_t = nn.Parameter(
            torch.zeros(1, self.d_t, self.encoder_embed_dim),
            requires_grad=False,
        )

        self.encoder_blocks = nn.ModuleList(
            [
                STBlock(
                    self.encoder_embed_dim,
                    self.encoder_n_heads,
                    d_s=self.num_patches,
                    d_t=d_t,
                    mlp_ratio=self.mlp_ratio,
                    drop_path=0.0,
                    enable_flashattn=use_flash_attn,
                    enable_layernorm_kernel=False,
                    st_use_qk_norm=True,
                )
                for _ in range(self.encoder_depth)
            ]
        )

    def freeze_all(self):
        """Freeze both DINOv2 and ST-Transformer"""
        print("NOTICE >>>>>>>>>>>>>>>>  freeze_all")
        self.eval()
        for param in self.parameters():
            param.requires_grad = False
            
    def unfreeze_st_transformer(self):
        """Unfreeze everything except DINOv2"""
        print("NOTICE >>>>>>>>>>>>>>>>  unfreeze_st_transformer")
        # 1. DINOv2 freeze 유지
        self.dino_encoder.eval()
        for param in self.dino_encoder.parameters():
            param.requires_grad = False
        
        # 2. 나머지 컴포넌트들 명시적으로 unfreeze
        if isinstance(self.proj_in, nn.Linear):
            self.proj_in.train()
            for param in self.proj_in.parameters():
                param.requires_grad = True
                
        if self.encoder_pe.requires_grad:
            self.encoder_pe.requires_grad = True
        if self.encoder_pe_t.requires_grad:
            self.encoder_pe_t.requires_grad = True
            
        for block in self.encoder_blocks:
            block.train()
            for param in block.parameters():
                param.requires_grad = True


    def forward_encoder(
        self,
        imgs: torch.Tensor,
        pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        imgs = self.proj_in(imgs)
        print(f"\nBefore ST-transformer - shape: {imgs.shape}")
        print(f"                        range: [{imgs.min():.4f}, {imgs.max():.4f}]")
        print(f"                        mean/std: {imgs.mean():.4f}/{imgs.std():.4f}")


        patches = rearrange(imgs, "bsz ctx seq dim -> (bsz ctx) seq dim")
        patches_pe = patches + self.encoder_pe
        ctx_patches = rearrange(patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=self.d_t)

        pad_seq_len = tuple(torch.sum(pad_mask, dim=1).cpu().numpy()) 
        imgs_list = [ctx_patches[i][-pad_seq_len[i]:] for i in range(len(pad_seq_len)) if pad_seq_len[i] > 1]
        pad_len = [tensor.shape[0] for tensor in imgs_list]
        hidden_states = torch.cat(imgs_list, dim=0)
        
        # 각 block을 순차적으로 처리하면서 gradient flow 유지
        for block_id, block in enumerate(self.encoder_blocks):
            if block_id == 0:
                hidden_states = block(hidden_states, pad_len, tpe=self.encoder_pe_t)
            else:
                hidden_states = block(hidden_states, pad_len, tpe=None)

        split_x = list(torch.split(hidden_states, split_size_or_sections=pad_len, dim=0))

        return split_x


    def normalize_images(self, img, img_norm_type="imagenet"):
        if img_norm_type == "default":
            # put pixels in [-1, 1]
            return img.to(torch.float32) / 127.5 - 1.0
        elif img_norm_type == "imagenet":
            # put pixels in [0,1]
            img = img.to(torch.float32) / 255
            assert img.shape[-1] % 3 == 0, "images should have rgb channels!"

            # define pixel-wise mean/std stats calculated from ImageNet
            mean = torch.tensor([0.485, 0.456, 0.406]).reshape((1, 1, 1, 3)).to(img.device)
            std = torch.tensor([0.229, 0.224, 0.225]).reshape((1, 1, 1, 3)).to(img.device)

            # tile mean and std (to account for stacked early_fusion images)
            num_tile = (1, 1, 1, int(img.shape[-1] / 3))
            mean_tile = torch.tile(mean, num_tile)
            std_tile = torch.tile(std, num_tile)

            # tile the mean/std, normalize image, and return
            return (img - mean_tile) / std_tile
        raise ValueError()


    def forward(
        self, imgs: torch.Tensor, pad_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # imgs = self.normalize_images(imgs)
        imgs = rearrange(imgs, "B T H W C -> B T C H W")

        batch_size = imgs.shape[0]
        imgs = rearrange(imgs, "B T C H W -> (B T) C H W")

        print("\n=== IGOR_IDM Feature Flow ===")
        with torch.no_grad():
            images_patches, _ = self.dino_encoder(imgs)
        print(f"DINOv2 output - shape: {images_patches.shape}")
        print(f"                 range: [{images_patches.min():.4f}, {images_patches.max():.4f}]")
        print(f"                 mean/std: {images_patches.mean():.4f}/{images_patches.std():.4f}")


        images_patches = rearrange(images_patches, "(b t) l c -> b t l c", b=batch_size)  

        images_patches = self.forward_encoder(images_patches, pad_mask)
        images_patches = torch.cat([images_patches_i for images_patches_i in images_patches], dim=0)

        return images_patches
    


# class IGOR_IDM(nn.Module):
#     def __init__(
#         self,
#         d_t: int,
#         encoder_depth: int,
#         encoder_embed_dim: int,
#         encoder_n_heads: int,
#         pretrained_dino_size: str,
#         mlp_ratio: float = 4.0,
#         use_flash_attn: bool = True,
#     ) -> None:
#         super().__init__()

#         self.d_t = d_t 
#         self.mlp_ratio = mlp_ratio
#         self.encoder_depth = encoder_depth
#         self.encoder_embed_dim = encoder_embed_dim
#         self.encoder_n_heads = encoder_n_heads
#         self.use_flash_attn = use_flash_attn
        
#         # DINOv2 Encoder (always frozen)
#         self.dino_encoder = FrozenDinov2ImageEmbedder(
#             pretrained=True,
#             freeze=True,
#             size=pretrained_dino_size
#         )
#         # self.dino_encoder.eval()

#         # Project DINOv2 features if needed
#         self.patch2embed = None
#         self.num_patches = 256
#         if pretrained_dino_size == 'base':
#             dino_channels = 768
#         else:
#             raise NotImplementedError        
#         self.proj_in = nn.Linear(dino_channels, self.encoder_embed_dim, bias=True) if dino_channels != self.encoder_embed_dim else nn.Identity()

#         # Positional embeddings
#         self.encoder_pe = nn.Parameter(
#             torch.zeros(1, self.num_patches, self.encoder_embed_dim),
#             requires_grad=False,
#         )
#         self.encoder_pe_t = nn.Parameter(
#             torch.zeros(1, self.d_t, self.encoder_embed_dim),
#             requires_grad=False,
#         )

#         # ST-Transformer blocks
#         self.encoder_blocks = nn.ModuleList(
#             [
#                 STBlock(
#                     self.encoder_embed_dim,
#                     self.encoder_n_heads,
#                     d_s=self.num_patches,
#                     d_t=d_t,
#                     mlp_ratio=self.mlp_ratio,
#                     drop_path=0.0,
#                     enable_flashattn=use_flash_attn,
#                     enable_layernorm_kernel=False,
#                     st_use_qk_norm=True,
#                 )
#                 for _ in range(self.encoder_depth)
#             ]
#         )

#     def freeze_all(self):
#         """Freeze both DINOv2 and ST-Transformer"""
#         self.eval()
#         for param in self.parameters():
#             param.requires_grad = False
            
#     def unfreeze_st_transformer(self):
#         """Unfreeze only ST-Transformer blocks"""
#         self.train()
#         for block in self.encoder_blocks:
#             for param in block.parameters():
#                 param.requires_grad = True

#     def forward(self, imgs: torch.Tensor, pad_mask: torch.Tensor):
#         # imgs: (B, T, H, W, C)
#         imgs = rearrange(imgs, "B T H W C -> (B T) C H W")

#         # DINOv2 feature extraction (always frozen)
#         with torch.no_grad():
#             images_patches, _ = self.dino_encoder(imgs) # [(B*T), L, D_dino]

#         # Reshape and project features
#         batch_size = imgs.shape[0] // self.d_t
#         patches = rearrange(images_patches, "(b t) l c -> b t l c", b=batch_size) # [B, T, L, D_dino]
#         patches = self.proj_in(patches) # [B, T, L, D]

#         # Add position encoding
#         patches = rearrange(patches, "bsz ctx seq dim -> (bsz ctx) seq dim") # [(B*T), L, D]
#         patches_pe = patches + self.encoder_pe # Add spatial PE, encoder_pe: [1, L, D]
#         ctx_patches = rearrange(patches_pe, "(bsz ctx) seq embed -> bsz ctx seq embed", ctx=self.d_t) # [B, T, L, D]

#         # ST-Transformer preprocessing: temporal masking - mask out frames to be removed temporally
#         pad_seq_len = tuple(torch.sum(pad_mask, dim=1).cpu().numpy())
#         imgs_list = [ctx_patches[i][-pad_seq_len[i]:] for i in range(len(pad_seq_len)) if pad_seq_len[i] > 1]
        # pad_len = [tensor.shape[0] for tensor in imgs_list]
        # hidden_states = torch.cat(imgs_list, dim=0) # [(B*T), L, D]

        # # Forward through ST blocks
        # for block_id, block in enumerate(self.encoder_blocks):
        #     if block_id == 0:
        #         hidden_states = block(hidden_states, pad_len, tpe=self.encoder_pe_t) # temporal position encoding
        #     else:
        #         hidden_states = block(hidden_states, pad_len, tpe=None)

        # # Handle variable-length sequences: split by pad_len and recombine to maintain sequence structure
        # split_x = list(torch.split(hidden_states, split_size_or_sections=pad_len, dim=0))
        # final_img_patches = torch.cat([split_x_i for split_x_i in split_x], dim=0)

        # return final_img_patches
