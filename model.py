import torch
from torchvision import models
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
from torchvision import models
import math 

class network(nn.Module):

	def __init__(self):
		super(network,self).__init__()
		self.conv1 = nn.Conv2d(1,64,kernel_size=(46,8),stride=(1,1),bias=True,padding=(0,0))
		self.pool = nn.MaxPool2d(kernel_size=(1,3),stride=(1,3))
		self.conv2 = nn.Conv2d(64,64,kernel_size=(18,4),stride=(1,1),bias=True,padding=(0,0))
		self.lin1 = nn.Linear(64,32)
		self.rel = nn.ReLU()
		self.dnn = nn.Sequential(nn.Linear(10656, 10656), nn.ReLU(inplace=True), nn.Linear(10656, 5328), nn.ReLU(inplace=True), 
			nn.Linear(5328, 2664), nn.ReLU(inplace=True), nn.Linear(2664, 1))

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				m.weight.data.normal_(0, math.sqrt(2. / n))

	def forward(self,x):
		import pdb
		pdb.set_trace()
		x = self.pool(self.conv1(x))
		x = self.conv2(x)
		x = x.permute(0,2,3,1)
		x = self.rel(self.lin1(x))
		x = x.view(x.size(0),-1)
		x = self.dnn(x)
		return x

