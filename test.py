import torch
import torch.nn as nn
from model import Linear_QNet


mymodel=Linear_QNet(11,256,3)
mymodel.load_state_dict(torch.load(r"model/model24.pth"))
mymodel.eval()
