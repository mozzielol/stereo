import tensorflow as tf
import sonnet as snt
import numpy as np
from bnn.BNNLayer import *
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display

class Model(object):
	"""docstring for Model"""
	def __init__(self, X_holder,y_holder,hidden_units=[]):
		super(Model, self).__init__()
		self.X_holder = X_holder
		self.y_holder = y_holder
		
		