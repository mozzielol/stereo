from builtins import range
from builtins import object
import numpy as np

from Grad.layer.layers import *
from Grad.layer.layer_utils import *
from copy import deepcopy



class Model(object): 
    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None, num_split=1, num_cluster=1, clusters=None):

        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}
        self.network_params = {}
        self.grad_params = {}
        self.trained_networks = []
        self.training_mask = {}
        self._mask = []
        self.clusters = clusters


        net_dims = [input_dim]+hidden_dims+[num_classes]

        #1. Init the params

        #How many sub params
        self.num_split = num_split

        for i in range(self.num_layers):
            step = net_dims[i+1] // num_split
            for j in range(self.num_split):
                _size = step if j!= self.num_split -1 else net_dims[i+1] - step * j
                self.network_params[('W%d' %(i+1),j)] = np.random.normal(loc=0.0,
                                                         scale=weight_scale,
                                                         size=(net_dims[i],
                                                         _size))
                self.network_params[('b%d' %(i+1),j)] = np.zeros(_size)

            self.training_mask['W%d' %(i+1)] = np.ones((net_dims[i],net_dims[i+1]))
            self.training_mask['b%d' %(i+1)] = np.ones(net_dims[i+1])
            '''    
            self.params['W%d' %(i+1)] = np.random.normal(loc=0.0,
                                                         scale=weight_scale,
                                                         size=(net_dims[i],
                                                         net_dims[i+1]))
            self.params['b%d' %(i+1)] = np.zeros(net_dims[i+1])
            '''
            if (self.use_batchnorm) & (i!=self.num_layers-1):
                self.params['gamma%d' %(i+1)] = np.ones(net_dims[i+1])
                self.params['beta%d' %(i+1)] = np.zeros(net_dims[i+1])

        for k, v in self.network_params.items():
            self.network_params[k] = v.astype(dtype)

        for i in range(num_cluster):
            self.grad_params[i] = deepcopy(self.network_params)

        self._mask = deepcopy(self.grad_params)

        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed


        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


        self._default_ref = []
        for i in range(num_split):
            self._default_ref.append(0)

        #self.update_training_params(self._default_ref)




    def update_training_params(self,ref):
        
        self.training_ref = ref
        trained = np.array(self.trained_networks)
        temp_mask = {}
        for idx_network,idx_clust in enumerate(ref):

            for i in range(self.num_layers):
                if idx_network == 0:
                    self.params['W%d'%(i+1)] = self.grad_params[idx_clust][('W%d'%(i+1),idx_network)]
                    self.params['b%d'%(i+1)] = self.grad_params[idx_clust][('b%d'%(i+1),idx_network)]

                    self.training_mask['W%d'%(i+1)] = self._mask[idx_clust][('W%d'%(i+1),idx_network)].copy()
                    self.training_mask['b%d'%(i+1)] = self._mask[idx_clust][('b%d'%(i+1),idx_network)].copy()

                    if len(trained) == 0 or idx_clust not in trained[:,idx_network]:
                        self.training_mask['W%d'%(i+1)].fill(1)
                        self.training_mask['b%d'%(i+1)].fill(1)

                    else:
                        self.training_mask['W%d'%(i+1)].fill(0)
                        self.training_mask['b%d'%(i+1)].fill(0)


                else:

                    self.params['W%d'%(i+1)] = np.concatenate([self.params['W%d'%(i+1)] ,self.grad_params[idx_clust][('W%d'%(i+1),idx_network)]],axis=1)
                    self.params['b%d'%(i+1)] = np.concatenate([self.params['b%d'%(i+1)] ,self.grad_params[idx_clust][('b%d'%(i+1),idx_network)]])


                    temp_mask_w = self._mask[idx_clust][('W%d'%(i+1),idx_network)]
                    temp_mask_b = self._mask[idx_clust][('b%d'%(i+1),idx_network)]
                    if len(trained) == 0 or idx_clust not in trained[:,idx_network]:
                        temp_mask_w.fill(1)
                        temp_mask_b.fill(1)
                    else:
                        temp_mask_w.fill(0)
                        temp_mask_b.fill(0)

                    self.training_mask['W%d'%(i+1)] = np.concatenate([self.training_mask['W%d'%(i+1)] ,temp_mask_w],axis=1)
                    self.training_mask['b%d'%(i+1)] = np.concatenate([self.training_mask['b%d'%(i+1)] ,temp_mask_b])

        for k,v in self.training_mask.items():
            self.training_mask[k] = v.astype(bool)

        
        

    def update_trained_networks(self):
        if self.training_ref in self.trained_networks:
            pass
        else:
            self.trained_networks.append(self.training_ref)

        self._update_params()


    def set_training_mask(self):
        self.training_mask = {}
        trained = np.array(self.trained_networks)
        for idx_network,idx_clust in enumerate(self.training_ref):
            if idx_clust in trained[:,idx_network] :
                for i in range(self.num_layers):
                    step = net_dims[i+1] // num_split

                    #self.training_mask['W%'%(i+1)][]

    def _update_params(self):
        
        trained = np.array(self.trained_networks)
        temp_mask = {}
        
        
        for i in range(self.num_layers):
            start = 0
            end = 0
            step = self.params['W%d'%(i+1)].shape[1] // self.num_split
            for idx_network,idx_clust in enumerate(self.training_ref):
                start = end
                end = step * (idx_network + 1) if idx_network != len(self.training_ref)-1 else self.params['W%d'%(i+1)].shape[1]
                self.grad_params[idx_clust][('W%d'%(i+1),idx_network)] = self.params['W%d'%(i+1)][:,start:end]
                self.grad_params[idx_clust][('b%d'%(i+1),idx_network)] = self.params['b%d'%(i+1)][start:end]


                
                

                
                


                


    def loss(self, X, y=None):

        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'


        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None

        scores = {}
        cache = {}
        dropout_cache = {}

        
        scores[0] = X

        
        if self.use_batchnorm:
            for i in range(1, self.num_layers+1):
                if i!=self.num_layers:
                    scores[i], cache[i] = batch_relu_forward(
                                          scores[i-1],
                                          self.params['W%d' %i],
                                          self.params['b%d' %i],
                                          self.params['gamma%d' %i],
                                          self.params['beta%d' %i],
                                          bn_param=self.bn_params[i-1]
                                          )
                    if self.use_dropout:
                        scores[i], dropout_cache[i] = dropout_forward(
                                                      scores[i],
                                                      self.dropout_param
                                                      )

                else:
                    scores[i], cache[i] = affine_forward(
                                          scores[i-1],
                                          self.params['W%d' %i],
                                          self.params['b%d' %i]
                                          )
        
        else:
            for i in range(1, self.num_layers+1):
                if i!=self.num_layers:
                    scores[i], cache[i] = affine_relu_forward(
                                          scores[i-1],
                                          self.params['W%d' %i],
                                          self.params['b%d' %i]
                                          )
                    if self.use_dropout:
                        scores[i], dropout_cache[i] = dropout_forward(
                                                      scores[i],
                                                      self.dropout_param
                                                      )
                else:
                    scores[i], cache[i] = affine_forward(
                                          scores[i-1],
                                          self.params['W%d' %i],
                                          self.params['b%d' %i]
                                          )


        if mode == 'test':
            return scores[self.num_layers]

        loss, grads = 0.0, {}

        loss, dscores = softmax_loss(scores[self.num_layers], y)

        
        for i in range(1, self.num_layers+1):
            loss += 0.5*self.reg*np.sum(self.params['W%d' %i]*self.params['W%d' %i])

        
        if self.use_batchnorm:
            for i in range(self.num_layers, 0, -1):
                if i!=self.num_layers:
                    if self.use_dropout:
                        dscores = dropout_backward(dscores, dropout_cache[i])
                    (dscores, grads['W%d' %i], grads['b%d' %i],
                     grads['gamma%d' %i], grads['beta%d' %i]) = batch_relu_backward(
                                                                dscores,
                                                                cache[i]
                                                                )
                    
                    grads['W%d' %i] += self.reg*self.params['W%d' %i]
                else:
                    dscores, grads['W%d' %i], grads['b%d' %i] = affine_backward(
                                                                dscores,
                                                                cache[i]
                                                                )

                    grads['W%d' %i] += self.reg*self.params['W%d' %i]


        else:
            for i in range(self.num_layers, 0, -1):
                if i!=self.num_layers:
                    if self.use_dropout:
                        dscores = dropout_backward(dscores, dropout_cache[i])
                    dscores, grads['W%d' %i], grads['b%d' %i] = affine_relu_backward(
                                                                dscores, cache[i])

                    grads['W%d' %i] += self.reg*self.params['W%d' %i]
                else:
                    dscores, grads['W%d' %i], grads['b%d' %i] = affine_backward(
                                                                dscores,
                                                                cache[i]
                                                                )

                    grads['W%d' %i] += self.reg*self.params['W%d' %i]
                    




        return loss, grads


    def _cluster(self,X):
        X = X.reshape(X.shape[0],-1)
        step = X.shape[1] // self.num_split
        start = 0
        end = 0
        ref = []
        for i in range(self.num_split):
            start = end
            end = (i+1) * step if i != self.num_split-1 else X.shape[1]
            idx = self.clusters[i].predict(X[:,start:end])
            u,c = np.unique(idx,return_counts=True)
            c = np.argsort(c)
            ref.append(c[-1])


        self.update_training_params(ref)

    def predict(self, X, y, num_samples=None, batch_size=100):
        
        ref_memory = self.training_ref
        self._update_params()
        self._cluster(X)

        # Maybe subsample the data
        N = X.shape[0]
        if num_samples is not None and N > num_samples:
            mask = np.random.choice(N, num_samples)
            N = num_samples
            X = X[mask]
            y = y[mask]

        # Compute predictions in batches
        num_batches = N // batch_size
        if N % batch_size != 0:
            num_batches += 1
        y_pred = []
        for i in range(num_batches):
            start = i * batch_size
            end = (i + 1) * batch_size
            scores = self.loss(X[start:end])
            y_pred.append(np.argmax(scores, axis=1))
        y_pred = np.hstack(y_pred)
        acc = np.mean(y_pred == y)

        self.update_training_params(ref_memory)
        return acc



