
import torch
from igor_idm import IGOR_IDM


print ("came to main.py")

# checkpoint_PATH = "/mnt/jaden/igor_encoder/checkpoint/igor_video_encoder.pth"
checkpoint_PATH = "/storage/igor_encoder/checkpoint/igor_video_encoder.pth"


model = IGOR_IDM(d_t=8,
                 encoder_depth=10,
                 encoder_embed_dim=768,
                 encoder_n_heads=12,
                 pretrained_dino_size='base',)

model.load_state_dict(torch.load(checkpoint_PATH))
model = model.to('cuda')

# psudo input:
model_T = 8  # should be fixed
batch_size = 12
pixel_h, pixel_w = 224, 224

# Create float tensor and normalize to [0, 1] range
image = torch.randint(0, 256, (batch_size, model_T, pixel_h, pixel_w, 3), dtype=torch.float32) / 255.0
pad_mask = torch.ones((batch_size, model_T), dtype=torch.bool)  # for padding information


with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    with torch.no_grad():
        output = model(image.to('cuda'),    
                    pad_mask.to('cuda'))  # [ B * T, 256, 768 ]

print (" - done - done- -done")
from IPython import embed; embed(); exit(1)