import torch
import torch.nn as nn

from upernet import UperNet_swin, UperNet_swinv2

model = UperNet_swinv2(img_size=512, num_classes=1)
checkpoint = torch.load("E:\\popar_swinv2_allxrays_512.pth", map_location='cpu')
checkpoint = {k.replace("module.", ""): v for k, v in checkpoint.items()}
checkpoint = {k.replace("backbone.", ""): v for k, v in checkpoint.items()}
checkpoint = {k.replace("swin_model.", ""): v for k, v in checkpoint.items()}

for k in ['head.weight', 'head.bias', 'head_dist.weight', 'head_dist.bias']:
    if k in checkpoint:
        print(f"Removing key {k} from pretrained checkpoint")
        del checkpoint[k]
result = model.backbone.load_state_dict(checkpoint, strict=False)
print(result)


input = torch.rand((2,3,512,512))
out = model(input)
print(out.shape)
