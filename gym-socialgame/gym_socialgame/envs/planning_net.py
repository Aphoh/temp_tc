import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pytorch_lightning as pl

class Net(torch.nn.Module):
	def __init__(self, n_feature, n_hidden, n_output, n_layers):
		super(Net, self).__init__()
		self.first_layer = nn.Linear(n_feature, n_hidden)
		self.hiddens = nn.ModuleList([nn.Linear(n_hidden, n_hidden) for _ in range(n_layers)])
		self.batchnorms = nn.ModuleList([nn.BatchNorm1d(n_hidden) for _ in range(n_layers)])
		#self.hidden = torch.nn.Linear(n_feature, n_hidden)
		#self.hidden2 = torch.nn.Linear(n_hidden, n_hidden)
		#self.hidden3 = torch.nn.Linear(n_hidden, n_hidden)  
		self.predict_layer = torch.nn.Linear(n_hidden, n_output)  

	def forward(self, x):
		x = F.relu(self.first_layer(x))
		for hidden, batchnorm in zip(self.hiddens, self.batchnorms):
			x = batchnorm(x)
			x = F.relu(hidden(x)) + x
		x = self.predict_layer(x)
		#x = F.relu(self.hidden(x))      
		#x = F.relu(self.hidden2(x))
		#x = F.relu(self.hidden3(x))
		#x = self.predict_layer(x)        
		x = x**2 # ensures costs will all be positive     
		return x, None

class Ensemble(nn.Module):
	def __init__(self, n_feature, n_hidden, n_output, n_layers, n_networks, device="cpu") -> None:
		super(Ensemble, self).__init__()
		self.n_output = n_output
		self.n_networks = n_networks
		self.device=device
		self.networks = nn.ModuleList([Net(n_feature, n_hidden, n_output, n_layers) for _ in range(n_networks)])
	
	def forward(self, x):
		outputs=  []
		for i, network in enumerate(self.networks):
			output, _ = network(x)
			outputs.append(output)
		outputs = torch.stack(outputs)
		return outputs.mean(dim=0), outputs.std(dim=0) / (self.n_networks ** 0.5), outputs

class EnsembleModule(pl.LightningModule):
	def __init__(self, n_feature, n_hidden, n_output, n_layers, n_networks, lr) -> None:
		super(EnsembleModule, self).__init__()
		self.n_output = n_output
		self.n_networks = n_networks
		self.lr = lr
		self.loss = nn.MSELoss()
		self.abs_loss = nn.L1Loss()
		self.model = Ensemble(n_feature, n_hidden, n_output, n_layers, n_networks, self.device)
		

	def shared_step(self, batch, batch_idx):
		x, y = batch
		y_hat, y_std = self.model(x)
		loss = self.loss(y_hat, y)
		abs_loss = self.abs_loss(y_hat, y)
		return loss, abs_loss, y_std.mean()

	def training_step(self, batch, batch_idx):
		loss, abs_loss, y_std = self.shared_step(batch, batch_idx)
		# logs metrics for each training_step,
		# and the average across the epoch, to the progress bar and logger
		self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log("train_std", y_std.mean(), on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log("train_abs_loss", abs_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		return loss

	def validation_step(self, batch, batch_idx):
		loss, abs_loss, y_std = self.shared_step(batch, batch_idx)
		# logs metrics for each training_step,
		# and the average across the epoch, to the progress bar and logger
		self.log("val_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log("val_std", y_std, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		self.log("val_abs_loss", abs_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
		
		return loss

	def configure_optimizers(self):
		return optim.Adam(self.parameters(), lr=self.lr)