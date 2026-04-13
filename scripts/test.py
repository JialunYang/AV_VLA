import torch
checkpoint = torch.load("training/scene_branch/scene_branch.pth")
print(checkpoint["category_mapping"])