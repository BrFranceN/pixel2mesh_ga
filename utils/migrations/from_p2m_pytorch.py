import re

import torch


checkpoint = torch.load("checkpoints/debug/20190705192654/000001_000001.pt")
pretrained = torch.load("checkpoints/pretrained/network_4.pth")

weights = checkpoint["model"]

for k in weights.keys():
    match = k
    match = re.sub("gcns\.(\d)", "GCN_\\1", match)
    match = re.sub("conv(\d)\.weight", "conv\\1.weight_2", match)
    match = re.sub("conv(\d)\.loop_weight", "conv\\1.weight_1", match)
    match = re.sub("gconv\.weight", "GConv.weight_2", match)
    match = re.sub("gconv\.loop_weight", "GConv.weight_1", match)
    match = re.sub("gconv\.", "GConv.", match)
    if match not in pretrained:
        print(k, match)
    else:
        weights[k] = pretrained[match]
torch.save(checkpoint, "checkpoints/debug/migration/network_4.pt")


# missing keys _IncompatibleKeys(
# missing_keys=['gcns.0.blocks.0.conv1.adj_mat',
#  'gcns.0.blocks.0.conv2.adj_mat',
#  'gcns.0.blocks.1.conv1.adj_mat', 
# 'gcns.0.blocks.1.conv2.adj_mat', 
# 'gcns.0.blocks.2.conv1.adj_mat', 
# 'gcns.0.blocks.2.conv2.adj_mat', 
# 'gcns.0.blocks.3.conv1.adj_mat',
#  'gcns.0.blocks.3.conv2.adj_mat',
#  'gcns.0.blocks.4.conv1.adj_mat',
#  'gcns.0.blocks.4.conv2.adj_mat', 
# 'gcns.0.blocks.5.conv1.adj_mat',
#  'gcns.0.blocks.5.conv2.adj_mat', 
# 'gcns.0.conv1.adj_mat', 
# 'gcns.0.conv2.adj_mat', 
# 'gcns.1.blocks.0.conv1.adj_mat', 
# 'gcns.1.blocks.0.conv2.adj_mat', 
# 'gcns.1.blocks.1.conv1.adj_mat', 
# 'gcns.1.blocks.1.conv2.adj_mat', 
# 'gcns.1.blocks.2.conv1.adj_mat', 
# 'gcns.1.blocks.2.conv2.adj_mat',
#  'gcns.1.blocks.3.conv1.adj_mat', 
# 'gcns.1.blocks.3.conv2.adj_mat', 
# 'gcns.1.blocks.4.conv1.adj_mat', 
# 'gcns.1.blocks.4.conv2.adj_mat', 
# 'gcns.1.blocks.5.conv1.adj_mat', 
# 'gcns.1.blocks.5.conv2.adj_mat', 
# 'gcns.1.conv1.adj_mat', 
# 'gcns.1.conv2.adj_mat', 
# 'gcns.2.blocks.0.conv1.adj_mat', 
# 'gcns.2.blocks.0.conv2.adj_mat',
#  'gcns.2.blocks.1.conv1.adj_mat', 
# 'gcns.2.blocks.1.conv2.adj_mat', 
# 'gcns.2.blocks.2.conv1.adj_mat',
#  'gcns.2.blocks.2.conv2.adj_mat', 
# 'gcns.2.blocks.3.conv1.adj_mat', 
# 'gcns.2.blocks.3.conv2.adj_mat', 
# 'gcns.2.blocks.4.conv1.adj_mat', 
# 'gcns.2.blocks.4.conv2.adj_mat', 
# 'gcns.2.blocks.5.conv1.adj_mat', 
# 'gcns.2.blocks.5.conv2.adj_mat',
#  'gcns.2.conv1.adj_mat', 
# 'gcns.2.conv2.adj_mat', 
# 'gconv.adj_mat'], 
# unexpected_keys=[])
