import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, sampler
from PIL import Image
import random
from scipy import signal
import wave 
import struct 
import random


class wave_spec(Dataset):

	def __init__(self,root_dir,trans = None):
		self.root_dir = root_dir
		self.trans = trans
		classes = [fn for fn in os.listdir(self.root_dir) if os.path.isdir(self.root_dir + '/' + fn) ]
		class_index = {}
		j = 0
		for i in classes:
			class_index[i] = j
			j+=1
		self.sounds = []
		for i in classes:
			self.sounds = self.sounds + [(root_dir + '/' + i + '/' + fn,class_index[i]) for fn in os.listdir(root_dir + '/' + i + '/') if fn.endswith('.wav') ]

	def __getitem__(self,idx):

		wave_file = wave.open(self.sounds[idx][0],'rb')
		fs = wave_file.getframerate()
		N = wave_file.getnframes()
		stft_img = self.stft_img(wave_file,fs,N)
		img = Image.fromarray(stft_img.astype('uint8'),'L')
		
		if self.trans:
			img = self.trans(img)

		label = self.sounds[idx][1]
		sample = {'image': img , 'label': label, 'path': self.sounds[idx][0]}

		return sample

	def __len__(self): 
		return len(self.sounds)

	def stft_img(self, wave_file,fs,N):
		wf = wave_file.readframes(N)
		wf = np.array(struct.unpack('h'*N, wf))
		_, _, Sxx = signal.spectrogram(wf, fs,return_onesided=True)
		return Sxx
