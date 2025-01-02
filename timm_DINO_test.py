from urllib.request import urlopen
from PIL import Image
import timm

img = Image.open(urlopen(
    'https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png'
))

model = timm.create_model(
    'vit_base_patch14_reg4_dinov2.lvd142m',
    pretrained=True,
    num_classes=0,  # remove classifier nn.Linear
)
model = model.eval()


# get model specific transforms (normalization, resize)
data_config = timm.data.resolve_model_data_config(model) # data_config => {'input_size': (3, 518, 518), 'interpolation': 'bicubic', 'mean': (0.485, 0.456, 0.406), 'std': (0.229, 0.224, 0.225), 'crop_pct': 1.0, 'crop_mode': 'center'}
transforms = timm.data.create_transform(**data_config, is_training=False)

output = model(transforms(img).unsqueeze(0))  # output is (batch_size, num_features) shaped tensor

# or equivalently (without needing to set num_classes=0)

output = model.forward_features(transforms(img).unsqueeze(0))
# output is unpooled, a (1, 1374, 768) shaped tensor

output = model.forward_head(output, pre_logits=True)
# output is a (1, num_features) shaped tensor



# (Pdb) transforms
# Compose(
#     Resize(size=518, interpolation=bicubic, max_size=None, antialias=True)
#     CenterCrop(size=(518, 518))
#     MaybeToTensor()
#     Normalize(mean=tensor([0.4850, 0.4560, 0.4060]), std=tensor([0.2290, 0.2240, 0.2250]))
# )

# openVLA의 경우에는 raw embedding을 사용.
# https://github.com/openvla/openvla/blob/main/prismatic/models/backbones/vision/dinosiglip_vit.py

# DINO 공식의 경우에는 norm을 output.
# https://github.com/facebookresearch/dinov2/blob/main/dinov2/models/vision_transformer.py

# (Pdb) transforms(img).shape
# torch.Size([3, 518, 518])
# (Pdb) transforms(img).mean()
# tensor(0.9638)
# (Pdb) transforms(img).std()
# tensor(1.0603)
# (Pdb) transforms(img).min()
# tensor(-2.1179)
# (Pdb) transforms(img).max()
# tensor(2.6400)

# (Pdb) output_2f = model.forward_features(transforms(img).unsqueeze(0))
# (Pdb) output_2.shape
# torch.Size([1, 1374, 768])
# (Pdb) output_2.mean()
# tensor(0.0085, grad_fn=<MeanBackward0>)
# (Pdb) output_2.std()
# tensor(0.8842, grad_fn=<StdBackward0>)
# (Pdb) output_2.min()
# tensor(-8.5748, grad_fn=<MinBackward1>)
# (Pdb) output_2.max()
# tensor(20.4968, grad_fn=<MaxBackward1>)

# (Pdb) output_2f = model.forward_head(output_2, pre_logits=True)
# (Pdb) output_2f.shape
# torch.Size([1, 768])
# (Pdb) output_2f.mean()
# tensor(0.0136, grad_fn=<MeanBackward0>)
# (Pdb) output_2f.std()
# tensor(0.8978, grad_fn=<StdBackward0>)


# (Pdb) output_1 = model(transforms(img).unsqueeze(0))
# (Pdb) output_1.shape
# torch.Size([1, 768])
# (Pdb) output_1.mean()
# tensor(0.0136, grad_fn=<MeanBackward0>)
# (Pdb) output_1.std()
# tensor(0.8978, grad_fn=<StdBackward0>)
# (Pdb) output_1.min()
# tensor(-2.6580, grad_fn=<MinBackward1>)
# (Pdb) output_1.max()
# tensor(2.5131, grad_fn=<MaxBackward1>)



