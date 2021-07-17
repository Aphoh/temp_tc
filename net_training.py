# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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
output_data = []

square_waves = np.array([[0,0,0,0,0,0,0,0,1,0],
                        [0,0,0,0,0,0,0,0,1,0],
                        [1,0,0,0,0,0,0,0,0,0],
                        [1,0,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0,0],
                        [0,1,0,0,0,0,0,0,0,0],
                        [0,0,1,0,0,0,0,0,0,1],
                        [0,0,1,0,0,0,0,0,0,1],
                        [0,0,0,1,0,0,0,0,0,0],
                        [0,0,0,1,0,0,0,0,0,0]])


for day in range(square_waves.shape[1]):
    output_data.append(bob.get_response(square_waves[:, day]))
    
output_data=np.array(output_data)


# %%
square_waves[:, 1]


# %%
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(-1, 1))
train_data_normalized = output_data.reshape(1, -1)#scaler.fit_transform(output_data.reshape(-1, 1))


# %%
output_data_normalized = train_data_normalized.reshape(10,10)
print(output_data_normalized)

# %%
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  
        self.predict = torch.nn.Linear(n_hidden, n_output)  

    def forward(self, x):
        x = F.relu(self.hidden(x))      
        x = self.predict(x)             
        return x


# %%
net = Net(n_feature=1, n_hidden=10, n_output=1)     # define the network
print(net)  # net architecture
optimizer = torch.optim.Adam(net.parameters())
loss_func = torch.nn.MSELoss() 


# %%

# train the network
for t in range(10000):
    loss = 0
    for i in range(square_waves.shape[1]):
        prediction = net(torch.tensor(square_waves[:,i].reshape(-1,1)).type(torch.FloatTensor))   
        loss +=  loss_func(prediction, torch.tensor(output_data_normalized[:,i].reshape(-1,1)).type(torch.FloatTensor))     # must be (1. nn output, 2. target)
    optimizer.zero_grad()   # clear gradients for next train
    loss.backward()         # backpropagation, compute gradients
    print(loss)
    
    optimizer.step()     
    


# %%
torch.save(net.state_dict(), 'model_weights.pth')


# %%



