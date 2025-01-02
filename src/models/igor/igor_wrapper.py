import torch
import torch.nn as nn
from .igor_idm import IGOR_IDM

"""
여기서 주목할 점들:

JEPA 인터페이스 요구사항 (embed_dim, num_heads) => 어차피 IGOR은 변경이 어려우니 JEPA를 IGOR에 맞춰야함. 근데 다행이 그냥 맞음.
clips 형식에 맞춘 입력 처리 => 여기에서 window를 나눠서 처리한다는 건가? => 이건 JEPA에서 함께 돌려보면서 확인 가능할듯.
DINOv2는 항상 frozen 상태 유지 
Fine-tuning을 위한 선택적 freezing 지원

이 구현이 적절해 보이나요? 추가로 체크해야 할 부분이 있을까요?
"""

class IGORWrapper(nn.Module):
    """JEPA interface wrapper for IGOR model.
    
    This wrapper handles:
    1. Compatibility with JEPA's video classification interface
    2. Processing 16-frame inputs through 8-frame IGOR windows
    """
    def __init__(
        self,
        d_t=8,
        encoder_depth=10,
        encoder_embed_dim=768,
        encoder_n_heads=12,
        pretrained_dino_size='base',
        use_flash_attn=True,
        attend_across_segments=True, # JEPA dataloader True. for IGOR
    ):
        super().__init__()
        
        # JEPA interface requirements
        self.embed_dim = encoder_embed_dim
        self.num_heads = encoder_n_heads
        self.attend_across_segments = attend_across_segments
        
        # Initialize IGOR model
        self.igor = IGOR_IDM(
            d_t=d_t,
            encoder_depth=encoder_depth,
            encoder_embed_dim=encoder_embed_dim,
            encoder_n_heads=encoder_n_heads,
            pretrained_dino_size=pretrained_dino_size,
            use_flash_attn=use_flash_attn
        )

    def process_frames(self, x):
        """Process 16-frame input using 8-frame windows.
        
        Args:
            x: [B, T=16, H, W, C] tensor
            
        Returns:
            Tensor of shape [B, T*N, D] 
        """
        # Split into two 8-frame windows
        first_half = x[:, :8]   # [B, 8, H, W, C]
        second_half = x[:, 8:]  # [B, 8, H, W, C]
        
        # Create padding masks (assuming all frames are valid)
        B = x.shape[0]
        pad_mask = torch.ones((B, 8), dtype=torch.bool, device=x.device)

        print("\n=== Process Frames Stats ===")
        print("Split shapes:", first_half.shape, second_half.shape)
        print("First half:")
        print(f"  Range: [{first_half.min():.4f}, {first_half.max():.4f}]")
        print(f"  Mean: {first_half.mean():.4f}")
        print(f"  Std: {first_half.std():.4f}")
        
        # Process each window
        print("\nProcessing through IGOR...")
        feat1 = self.igor(first_half, pad_mask) # [B*8, 256, 768]
        print("First half IGOR output:")
        print(f"  Shape: {feat1.shape}")
        print(f"  Range: [{feat1.min():.4f}, {feat1.max():.4f}]")
        print(f"  Mean: {feat1.mean():.4f}")
        print(f"  Std: {feat1.std():.4f}")
        feat2 = self.igor(second_half, pad_mask) # [B*8, 256, 768]
        print("\nSecond half IGOR output:")
        print(f"  Shape: {feat2.shape}")
        print(f"  Range: [{feat2.min():.4f}, {feat2.max():.4f}]")
        print(f"  Mean: {feat2.mean():.4f}")
        print(f"  Std: {feat2.std():.4f}")
        
        # Restore temporal structure and combine
        feat1 = feat1.reshape(B, 8, 256, 768)
        feat2 = feat2.reshape(B, 8, 256, 768)
        features = torch.cat([feat1, feat2], dim=1) # [B, 16, 256, 768]

        print("\nFinal combined features:")
        print(f"  Shape: {features.shape}")
        print(f"  Range: [{features.min():.4f}, {features.max():.4f}]")
        print(f"  Mean: {features.mean():.4f}")
        print(f"  Std: {features.std():.4f}")
        print("===========================\n")
        
        # Flatten temporal and spatial dimensions
        features = features.reshape(B, -1, 768)  # [B, 16*256, 768] ~ [16, 4096, 768]
        return features

    def forward(self, clips, clip_indices=None):
        """
        Args:
            clips: List[List[Tensor]], outer list for temporal segments, inner list for spatial views
                  - [num_segments][num_views][B, C, T, H, W]
            clip_indices: Optional clip indexing information
            
        Returns:
            List[Tensor]: List of features for each spatial view
                         Each tensor has shape [B*num_segments, 256, 768]
        """
        # import pdb;pdb.set_trace()
        num_segments = len(clips)      # temporal segments 수
        num_views = len(clips[0])      # spatial views 수
        batch_size = clips[0][0].shape[0]

        # Input data check
        print("=== Input Data Format and Values Check ===")
        print(f"Number of segments: {num_segments}")
        print(f"Number of views: {num_views}")
        print(f"Batch size: {batch_size}")
        print(f"Single clip shape: {clips[0][0].shape}")
        print("First clip statistics:")
        input_data = clips[0][0]  # [B, C, T, H, W]
        print(f"  Range: [{input_data.min():.4f}, {input_data.max():.4f}]")
        print(f"  Mean: {input_data.mean():.4f}")
        print(f"  Std: {input_data.std():.4f}")
        print("Channel-wise statistics:")
        for c in range(input_data.shape[1]):
            print(f"  Channel {c}:")
            print(f"    Mean: {input_data[:,c].mean():.4f}")
            print(f"    Std: {input_data[:,c].std():.4f}")
        print("=======================================\n")

        all_outputs = []
        # Process each spatial view independently
        for view_idx in range(num_views):
            # Get all segments for current view
            view_clips = [clips[i][view_idx] for i in range(num_segments)]  # List[Tensor] each [B, C, T, H, W] ~ [2][8, 3, 16, 224, 224]
            
            # Stack all segments along batch dimension
            x = torch.cat(view_clips, dim=0)  # [B*num_segments, C, T, H, W] ~ [16, 3, 16, 224, 224]
            
            # Reshape to IGOR format
            x = x.permute(0, 2, 3, 4, 1)  # [B*num_segments, T, H, W, C] ~ [16, 16, 224, 224, 3]

            # Pre-IGOR check
            print(f"\nView {view_idx} - Pre IGOR:")
            print(f"Shape after cat & permute: {x.shape}")
            print(f"Range: [{x.min():.4f}, {x.max():.4f}]")
            print(f"Mean: {x.mean():.4f}")
            print(f"Std: {x.std():.4f}")

            # Process through IGOR (16 frames -> 2 x 8 frames)
            features = self.process_frames(x)  # [B*num_segments, T*N, D] ~ [16, 16*256, 768]

            if self.attend_across_segments:
                # Reshape to separate batch and segments
                features = features.reshape(batch_size, num_segments, -1, self.embed_dim)
                # Concatenate segments
                features = features.reshape(batch_size, -1, self.embed_dim) # [B, num_segments*T*N, D] ~ [8, 8192, 768]
            else:
                # First reshape to separate segments properly
                features = features.reshape(batch_size, num_segments, -1, self.embed_dim)
                # Convert to list of segment features
                features = [features[:, i, :, :] for i in range(num_segments)] # [num_segments][B, T*N, D] ~ [2][8, 4096, 768]
            
            all_outputs.append(features) # [spatial][]
            
        return all_outputs
    
    def train(self, mode=True):
        """Override to ensure DINOv2 stays in eval mode"""
        super().train(mode)
        if mode:
            # DINOv2는 항상 frozen
            self.igor.dino_encoder.eval()
        return self

    def freeze_all(self):
        """Freeze both DINOv2 and ST-Transformer"""
        self.igor.freeze_all()
            
    def unfreeze_st_transformer(self):
        """Unfreeze only ST-Transformer blocks for fine-tuning"""
        self.igor.unfreeze_st_transformer()