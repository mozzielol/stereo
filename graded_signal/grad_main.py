from Grad.models import Model
from Grad.solver import Solver
from keras.datasets import mnist,fashion_mnist
from keras.utils import to_categorical
import numpy as np
from Grad.data_utils import get_CIFAR10_data
from sklearn.mixture import GaussianMixture as GMM

'''
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')


mean_image = np.mean(x_train, axis=0)
x_train -= mean_image
x_test -= mean_image
'''


def load_data(name):
  if name =='mnist':
    data = mnist
  elif name == 'fashion_mnist':
    data = fashion_mnist

  (x_train, y_train), (x_test, y_test) = data.load_data()
  x_train = x_train.astype('float32')
  x_test = x_test.astype('float32')

  mean_image = np.mean(x_train, axis=0)
  x_train -= mean_image
  x_test -= mean_image

  x_train /= 255
  x_test /= 255
  num_train = x_train.shape[0]
  num_val = x_test.shape[0]
  #num_train = 200
  #num_val = 200

  small_data = {
    'X_train': x_train[:num_train],
    'y_train': y_train[:num_train],
    'X_val': x_test[:num_val],
    'y_val': y_test[:num_val],
  }

  data_part1 = {    
    'X_train': x_train[:num_train][y_train[:num_train]<5],
    'y_train': y_train[:num_train][y_train[:num_train]<5],
    'X_val': x_test[:num_val][y_test[:num_val]<5],
    'y_val': y_test[:num_val][y_test[:num_val]<5]
  }

  #(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

  data_part2 = {    
    'X_train': x_train[:num_train][y_train[:num_train]>=5],
    'y_train': y_train[:num_train][y_train[:num_train]>=5],
    'X_val': x_test[:num_val][y_test[:num_val]>=5],
    'y_val': y_test[:num_val][y_test[:num_val]>=5]
  }

  return small_data,data_part1,data_part2


#1. clustering
#2. 

num_split = 2
num_cluster = 2

  

small_data,data_part1,data_part2 = load_data('fashion_mnist')

clusters = []
for i in range(num_split):
  clusters.append(GMM(n_components = num_cluster))

X = small_data['X_train']
X = X.reshape(X.shape[0],-1)

step = X.shape[1] // num_split
start = 0
end = 0

for i in range(num_split):
    start = end
    end = (i+1) * step if i != num_split-1 else X.shape[1]
    clusters[i].fit(X[:,start:end])
    
    
    


#small_data = get_CIFAR10_data()
model = Model(hidden_dims=[256,512,256],dropout=0.5,num_split=num_split,input_dim=784,num_cluster=num_cluster,clusters=clusters)


solver = Solver(model, data_part1,
                print_every=200, num_epochs=10, batch_size=128,
                update_rule='adam',checkpoint_name ='./check_point/first',pre_X=data_part1['X_val'],pre_y=data_part1['y_val'],
                optim_config={
                  'learning_rate': 5e-4,
                }
         )
solver.train()


print('--'*20)
#small_data = load_data('mnist')
#model.define_parameters(which_network=[0]*num_networks,trainable_mask=[1]*num_networks)
solver = Solver(model, data_part2,
                print_every=200, num_epochs=10, batch_size=128,
                update_rule='adam',checkpoint_name ='./check_point/first',
                optim_config={
                  'learning_rate': 5e-4,
                },pre_X=data_part1['X_val'],pre_y=data_part1['y_val']
         )
solver.train()

print('---')
print(model.predict(data_part1['X_val'], data_part1['y_val']))
print(model.predict(data_part2['X_val'], data_part2['y_val']))

'''
(x_train, y_train), (x_test, y_test) = mnist.load_data()
model.predict(x_test, y_test,which_network=[0,0,0])
model.predict(x_test, y_test,which_network=[1,1,1])
model.predict(x_test, y_test,which_network=[2,2,2])
(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
model.predict(x_test, y_test,which_network=[0,0,0])
model.predict(x_test, y_test,which_network=[1,1,1])
model.predict(x_test, y_test,which_network=[2,2,2])

'''




