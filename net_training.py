# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pandas as pd
import os

import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint


# %%

class Person():
	""" Person (parent?) class -- will define how the person takes in a points signal and puts out an energy signal 
	baseline_energy = a list or dataframe of values. This is data from SinBerBEST 
	points_multiplier = an int which describes how sensitive each person is to points 

	"""

	def __init__(self, baseline_energy_df, points_multiplier = 1):
		self.baseline_energy_df = baseline_energy_df
		self.baseline_energy = np.array(self.baseline_energy_df["net_energy_use"])
		self.points_multiplier = points_multiplier
		
		baseline_min = self.baseline_energy.min()
		baseline_max = self.baseline_energy.max()
		baseline_range = baseline_max - baseline_min
		
		self.min_demand = np.maximum(0, baseline_min + baseline_range * .05)
		self.max_demand = np.maximum(0, baseline_min + baseline_range * .95)


	def energy_output_simple_linear(self, points):
		"""Determines the energy output of the person, based on the formula:
		
		y[n] = -sum_{rolling window of 5} points + baseline_energy + noise

		inputs: points - list or dataframe of points values. Assumes that the 
		list will be in the same time increment that energy_output will be. 

		For now, that's in 1 hour increments

		"""
		points_df = pd.DataFrame(points)
		
		points_effect = (
			points_df
				.rolling(
						window = 5,
						min_periods = 1)
				.mean()
			)

		time = points_effect.shape[0]
		energy_output= []

		for t in range(time):
			temp_energy = self.baseline_energy[t] - points_effect.iloc[t]*self.points_multiplier + 				np.random.normal(1)
			energy_output.append(temp_energy)
			
		return pd.DataFrame(energy_output)

	def pure_linear_signal(self, points, baseline_day=0):
		"""
		A linear person. The more points you give them, the less energy they will use
		(within some bounds) for each hour. No rolling effects or anything. The simplest
		signal. 
		"""

		# hack here to always grab the first day from the baseline_energy
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]

		points_effect = np.array(points * self.points_multiplier)
		output = output - points_effect

		# impose bounds/constraints
		output = np.maximum(output, self.min_demand)
		output = np.minimum(output, self.max_demand)
		return output



	def get_min_demand(self):
		return self.min_demand
		# return np.quantile(self.baseline_energy, .05)

	def get_max_demand(self):
		return self.max_demand
		# return np.quantile(self.baseline_energy, .95)

class FixedDemandPerson(Person):

	def __init__(self, baseline_energy_df, points_multiplier = 1):
		super().__init__(baseline_energy_df, points_multiplier)


	def demand_from_points(self, points, baseline_day=0):
		# hack here to always grab the first day from the baseline_energy
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]
		total_demand = np.sum(output)


		points_effect = np.array(points * self.points_multiplier)
		output = output - points_effect

		# scale to keep total_demand (almost) constant
		# almost bc imposing bounds afterwards
		output = output * (total_demand/np.sum(output))

		# impose bounds/constraints
		output = np.maximum(output, self.min_demand)
		output = np.minimum(output, self.max_demand)

		return output

	def adverserial_linear(self, points, baseline_day=0):
		# hack here to always grab the first day from the baseline_energy
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]
		total_demand = np.sum(output)


		points_effect = np.array(points * self.points_multiplier)
		output = output + points_effect

		# scale to keep total_demand (almost) constant
		# almost bc imposing bounds afterwards
		output = output * (total_demand/np.sum(output))

		# impose bounds/constraints
		output = np.maximum(output, self.min_demand)
		output = np.minimum(output, self.max_demand)

		return output


# %%
class CurtailAndShiftPerson(Person):
	def __init__(self, baseline_energy_df, points_multiplier = 1, shiftable_load_frac = .7, 
			curtailable_load_frac = .4, shiftByHours = 3, maxCurtailHours=5, response = None, **kwargs):
		super().__init__(baseline_energy_df, points_multiplier)
		self.shiftableLoadFraction = shiftable_load_frac
		self.shiftByHours = shiftByHours
		self.curtailableLoadFraction = curtailable_load_frac
		self.maxCurtailHours = maxCurtailHours #Person willing to curtail for no more than these hours

	def shiftedLoad(self, points, baseline_day=0, day_of_week=None):
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]
		points = np.array(points) * self.points_multiplier
		shiftableLoad = self.shiftableLoadFraction*output
		shiftByHours = self.shiftByHours
		
		# 10 hour day. Rearrange the sum of shiftableLoad into these hours by treating points as the 'price' at that hour
		# Load can be shifted by a max of shiftByHours (default = 3 hours)
		# For each hour, calculate the optimal hour to shift load to within +- 3 hours
		shiftedLoad = np.zeros(10)
		for hour in range(10):
			candidatePrices = points[max(hour-shiftByHours,0): min(hour+shiftByHours,9)+1]
			shiftToHour = max(hour-shiftByHours,0) + np.argmin(candidatePrices)
			shiftedLoad[shiftToHour] += shiftableLoad[hour]		
		return shiftedLoad

	def curtailedLoad(self, points, baseline_day=0, day_of_week=None):
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]
		points = np.array(points) * self.points_multiplier
		curtailableLoad = self.curtailableLoadFraction*output
		maxPriceHours = np.argsort(points)[0:self.maxCurtailHours]
		for hour in maxPriceHours:
			curtailableLoad[hour] = 0
		return curtailableLoad

	def get_response(self, points, day_of_week=None):
		baseline_day = 0
		output = np.array(self.baseline_energy)[baseline_day*24:baseline_day*24+10]
		energy_resp = output*(1 - self.curtailableLoadFraction - self.shiftableLoadFraction) + self.curtailedLoad(points) + self.shiftedLoad(points)
		
			
		self.min_demand = np.maximum(0, min(energy_resp))
		self.max_demand = np.maximum(0, max(energy_resp))

		return energy_resp




# %%
a = {"net_energy_use":[15.09,  35.6, 123.5,  148.7,  158.49, 149.13, 159.32, 157.62, 158.8,  156.49]}


# %%
bob = CurtailAndShiftPerson(baseline_energy_df=a)


# %%
def get_datasets(train_size, val_size):
	output_data = []
	# square_waves = np.array([[0,0,0,0,0,0,0,0,1,0],
	#                         [0,0,0,0,0,0,0,0,1,0],
	#                         [1,0,0,0,0,0,0,0,0,0],
	#                         [1,0,0,0,0,0,0,0,0,0],
	#                         [0,1,0,0,0,0,0,0,0,0],
	#                         [0,1,0,0,0,0,0,0,0,0],
	#                         [0,0,1,0,0,0,0,0,0,1],
	#                         [0,0,1,0,0,0,0,0,0,1],
	#                         [0,0,0,1,0,0,0,0,0,0],
	#                         [0,0,0,1,0,0,0,0,0,0]])
	#square_waves = np.random.uniform(size=[train_size, 10])
	square_waves = np.random.lognormal(0, 1, size=[train_size, 10])
	#square_waves /= np.sum(square_waves, axis=-1, keepdims=True)
	#validation_waves = np.random.uniform(size=[val_size, 10])
	#validation_waves = validation_waves  / np.sum(validation_waves, axis=-1, keepdims=True)
	validation_waves = np.random.lognormal(0, 1, size=[val_size, 10])
	validation_data = []

	for day in range(square_waves.shape[0]):
		output_data.append(bob.get_response(square_waves[day]))
	for day in range(validation_waves.shape[0]):
		validation_data.append(bob.get_response(validation_waves[day]))
		
	output_data=np.array(output_data)
	validation_data = np.array(validation_data)
	return square_waves, validation_waves, output_data, validation_data


# %%


# %%
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
		return outputs.mean(dim=0), outputs.std(dim=0) / (self.n_networks ** 0.5)

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

# %%
class BaseData(Dataset):
	def __init__(self, num_samples) -> None:
		super().__init__()
		self.num_samples = num_samples
		self.waves, _, self.output, _ = get_datasets(num_samples, 0)

	def __getitem__(self, idx):
		return torch.tensor(self.waves[idx], dtype=torch.float32), torch.tensor(self.output[idx], dtype=torch.float32)    

	def __len__(self):
		return len(self.waves)

class LitData(pl.LightningDataModule):
	def __init__(self, num_train, num_val, batch_size) -> None:
		super().__init__()
		self.num_train = num_train
		self.num_val = num_val
		self.batch_size = batch_size
	
	def setup(self, stage = None):
		self.train_dataset = BaseData(self.num_train)
		self.val_dataset = BaseData(self.num_val)

	def train_dataloader(self):
		return DataLoader(self.train_dataset, self.batch_size, num_workers = 4)

	def val_dataloader(self):
		return DataLoader(self.val_dataset, self.batch_size, num_workers=4)

# %%
#net = Net(n_feature=10, n_hidden=32, n_output=10)     # define the network


# %%
# from sklearn.linear_model import ElasticNet
# l1_ratios = [0, 0.1, 0.5, 0.7, 0.95, 0.99, 1]
# alphas = [10**i for i in range(-4, 2, 1)]
# min_loss = None
# best_l1_ratio=None
# best_alpha=None
# for l1_ratio in l1_ratios:
#     for alpha in alphas:
#         elastic_net = ElasticNet(alpha=alpha, l1_ratio=l1_ratio)
#         elastic_net.fit(square_waves.reshape(-1, 1), output_data_normalized.reshape(-1, 1))
#         val_out = elastic_net.predict(validation_waves.reshape(-1, 1))
#         val_loss = loss_func(torch.tensor(val_data_normalized.reshape(-1, 1)), torch.tensor(val_out.reshape(-1, 1)))
#         if min_loss is None or val_loss < min_loss:
#             min_loss = val_loss
#             best_l1_ratio = l1_ratio
#             best_alpha = alpha
# print("Min loss: {}, best l1 ratio: {}, best alpha: {}".format(min_loss, best_l1_ratio, best_alpha))


# %%
from copy import deepcopy
import wandb

n_networks = [24]
n_train_datas = [500, 1000, 2000, 5000, 7500, 10000, 15000, 20000, 30000, 50000, 75000, 100000]
#lrs = [10**i for i in range(-5, 0)]
lrs = [0.001]
#n_hiddens = [4, 8, 16, 32, 64]
#n_hiddens = [64]
#weight_decays = [10**i for i in range(-4, 0)]
weight_decays = [0]
best_final_val_loss = 10000000
best_abs_loss = None
best_final_model = None
best_params = None # 24 0.1 64 10**-4
best_final_std = None
best_stds = []
best_val_losses = []
best_val_abs_losses = []
ckpt_folder = "./planning_ckpts_diverse2/"
for n_network in n_networks:
	for weight_decay in weight_decays:
		for n_train_data in n_train_datas:

			n_hidden=64
			#square_waves, validation_waves, output_data_normalized, val_data_normalized = get_datasets(n_train_data, 256)

			n_layer=5 
			net = EnsembleModule(n_feature=10, n_hidden=n_hidden, n_output=10, n_layers=n_layer, n_networks=n_network, lr=3e-4)
			data = LitData(n_train_data, 512, batch_size=2048)
			save_folder = os.path.join(ckpt_folder, str(n_train_data))
			logger = pl.loggers.WandbLogger(project='energy-demand-planning-model', entity='social-game-rl', reinit=True, tags=["big_network", "even_more_patience2"])
			early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=0.00, patience=50, divergence_threshold=100000, verbose=True, mode="min")
			checkpoint_callback = ModelCheckpoint(
				monitor="val_loss",
				dirpath=save_folder,
				mode="min",
			)
			logger.log_hyperparams({"n_train_data": n_train_data})
			trainer = pl.Trainer(gpus=1, auto_lr_find=True, logger=logger, callbacks=[checkpoint_callback, early_stop_callback], max_epochs=10000, min_epochs=50)
			logger.experiment.define_metric("train_loss", summary="min")
			logger.experiment.define_metric("val_loss", summary="min")
			logger.experiment.define_metric("train_abs_loss", summary="min")
			logger.experiment.define_metric("val_abs_loss", summary="min")
			trainer.tune(net, datamodule=data)
			trainer.fit(net, datamodule=data)
			logger.experiment.finish()
				




# %%
