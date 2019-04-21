import tensorflow as tf
import sonnet as snt
import numpy as np
from bnn.BNN_MLP import BNN_MLP
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
from abc import abstractmethod
from bnn.utils import create_mixture

class Model(BNN_MLP):
	"""docstring for Model"""
	def __init__(self,**kwargs):
		name = kwargs.pop('name','model')
		which_dataset = kwargs.pop('which_dataset','mnist')
		if which_dataset == 'mnist':
			n_inputs = kwargs.pop('n_inputs',784)
		elif which_dataset == 'cifar10':
			n_inputs = kwargs.pop('n_inputs',2048)
		n_output = kwargs.pop('n_output',10)
		hidden_units = kwargs.pop('hidden_units',[50,50])
		num_task = kwargs.pop('num_task',2)
		self.num_task = num_task
		super(Model, self).__init__(n_inputs,n_output,hidden_units,num_task)
		
	def _st_smooth(self,var_idx,x_v,y_v=None,n_component=1,thresh_hold=0.3,dp=False):
		mixture_dist = []
		for task_idx in range(self.num_task):
		    if y_v is not None:
		        mean = self.params_mean[task_idx][var_idx][x_v][y_v]
		        var = self.transform_var(self.params_var[task_idx][var_idx][x_v][y_v])
		    else:
		        mean = self.params_mean[task_idx][var_idx][x_v]
		        var = self.transform_var(self.params_var[task_idx][var_idx][x_v])
		    mixture_dist.append({'kwargs':{'loc':mean,'scale':var}})

		sample = create_mixture(mixture_dist,self.alpha_list)
		if dp:
		    gmm = DPGMM( max_iter=1000,  n_components=n_component, covariance_type='spherical')
		else:
		    gmm = GMM( max_iter=500,  n_components=n_component, covariance_type='spherical')
		gmm.fit(sample)
		new_idx_list = []
		for task_idx in range(self.num_task):
		    if y_v is not None:
		        predict_probability = gmm.predict_proba(np.array(self.params_mean[task_idx][var_idx][x_v][y_v]).reshape(-1,1))
		    else:
		        predict_probability = gmm.predict_proba(np.array(self.params_mean[task_idx][var_idx][x_v]).reshape(-1,1))
		    f_ = True
		    while f_:
		        if gmm.weights_[np.argmax(predict_probability)] > thresh_hold:
		            new_idx = np.argmax(predict_probability)
		            f_ = False
		        else:
		            predict_probability[0][np.argmax(predict_probability)] = 0.0
		            self.num_merged_params += 1
		    
		    new_idx_list.append(new_idx)
		    if y_v is not None:
		        self.params_mean[task_idx][var_idx][x_v][y_v] = gmm.means_[new_idx]
		        self.params_var[task_idx][var_idx][x_v][y_v] = self.retransform_var(gmm.covariances_[new_idx])
		    else:
		        self.params_mean[task_idx][var_idx][x_v] = gmm.means_[new_idx]
		        self.params_var[task_idx][var_idx][x_v] = self.retransform_var(gmm.covariances_[new_idx])

	def st_smooth(self,n_component=1,dp=True,thresh_hold=0.3):
		self.num_merged_params = 0
		#The range of len(params)
		_step = 0
		for var_idx in tqdm(range(len(self.params_mean[0]))):
		    for x_v in range(len(self.params_mean[0][var_idx])):
		        print('Step %d'%_step,end='\r')
		        _step += 1
		        try:
		            for y_v in range(len(self.merge_var[0][var_idx][x_v])):
		                self._st_smooth(var_idx,x_v,y_v=y_v,n_component=n_component,thresh_hold=threshold,dp=dp)

		        except TypeError:
		            self._st_smooth(var_idx,x_v,n_component=n_component,thresh_hold=threshold,dp=dp)
		            

	def imm_mean(self,sess,alpha=0.5):
		sum_idx = max(self.params_mean,key=int)

		sum_mean = self.params_mean[sum_idx]
		sum_var = self.params_var[sum_idx]
		for i in range(len(sum_mean)):
			sum_mean[i] *= alpha
			sum_var[i] = (self.transform_var(sum_var[i]))**2 * alpha**2

		for idx in range(self.num_task-1):
			for i in range(len(sum_mean)):
				sum_mean[i] += self.params_mean[idx][i] * (1-alpha) / (self.num_task - 1)
				sum_var[i] += ((self.transform_var(self.params_var[idx][i]))**2) * (((1-alpha) / (self.num_task - 1))**2)

		for i in range(len(sum_var)):
			sum_var[i] = self.retransform_var(sum_var[i])
			sess.run(self.tensor_mean[v].assign(sum_mean[v]))
			sess.run(self.tensor_var[v].assign(sum_var[v]))

	def imm_mode(self,sess,alpha=0.5):
		sum_idx = max(self.params_mean,key=int)
		sum_mean = self.params_mean[sum_idx] 
		sum_var = self.params_var[sum_idx]
		fisher_sum = self.FISHER[sum_idx] 
		for i in range(len(sum_mean)):
			fisher_sum[i] *= alpha

		for n in range(self.num_task-1):
			for idx in range(len(fisher_sum)):
				fisher_sum[idx] += self.FISHER[n][idx] * (1-alpha) / (self.num_task - 1)

		for i in range(len(sum_mean)):
			sum_mean[i] *= alpha * self.FISHER[sum_idx][i] / fisher_sum[i]
			sum_var[i] = (self.transform_var(sum_var[i]))**2 * alpha**2
				
		for idx in range(self.num_task-1):
			for i in range(len(sum_mean)):
				sum_mean[i] += self.params_mean[idx][i] * (1-alpha) / (self.num_task - 1) * self.FISHER[idx][i] / fisher_sum[i]
				sum_var[i] += ((self.transform_var(self.params_var[idx][i]))**2) * (((1-alpha) / (self.num_task - 1))**2)

		for i in range(len(sum_var)):
			sum_var[i] = self.retransform_var(sum_var[i])
			sess.run(self.tensor_mean[v].assign(sum_mean[v]))
			sess.run(self.tensor_var[v].assign(sum_var[v]))

	def dist_merge(self,sess, alpha=0.5):
		sum_idx = max(self.params_mean,key=int)
		sum_mean = self.params_mean[sum_idx] 
		sum_var = self.params_var[sum_idx]
		uncertainty = self.params_var[sum_idx]

		for i in range(len(sum_mean)):
			uncertainty[i] = self.transform_var(sum_var[i]) * (self.num_task - 1)

		for n in range(self.num_task-1):
			for idx in range(len(uncertainty)):
				uncertainty[idx] += self.params_var[n][idx] * (self.num_task - 1)

		for i in range(len(sum_mean)):
			sum_mean[i] *= alpha * slef.params_var[sum_idx][i] / uncertainty[i]
			sum_var[i] = (self.transform_var(sum_var[i]))**2 * alpha**2


				
		for idx in range(self.num_task-1):
			for i in range(len(sum_mean)):
				sum_mean[i] += self.params_mean[idx][i] * (1-alpha) / (self.num_task - 1) * (1 - 
					self.transform_var(self.params_var[idx][i]) / uncertainty[i])

				sum_var[i] += ((self.transform_var(self.params_var[idx][i]))**2) * (((1-alpha) / (self.num_task - 1))**2)

		for i in range(len(sum_var)):
			sum_var[i] = self.retransform_var(sum_var[i])
			sess.run(self.tensor_mean[v].assign(sum_mean[v]))
			sess.run(self.tensor_var[v].assign(sum_var[v]))



	













		

		