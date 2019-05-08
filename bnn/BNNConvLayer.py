import tensorflow as tf
import sonnet as snt

import numpy as np


class BNNConvLayer(snt.AbstractModule):
    """
    Implementation of a linear Bayesian layer with n_inputs and n_outputs, and a diagonal gaussian
    posterior. The weights are initialized to mean init_mu mean and standard deviation log(1+exp(init_rho)).
    """

    def __init__(self,inputs,n_filters, filter_shape,stride, num_task=2, name="BNNConvLayer"):
        super(BNNConvLayer, self).__init__(name=name)
        self.num_task = num_task
        init_mu=0.0, 
        init_rho=0.0
        filter_shape = tf.TensorShape(filter_shape + [inputs.shape[-1], n_filters])
        self.stride = stride
        self.vert_stride = 1 if 1 in filter_shape else self.stride


        self.w_mean = tf.Variable(init_mu*tf.ones(filter_shape))#tf.random_normal([self.n_inputs, self.n_outputs], 0.0, 0.01))
        self.w_rho = tf.Variable(init_rho*tf.ones(filter_shape))#tf.Variable(tf.random_normal([self.n_inputs, self.n_outputs], -4, 0.1)) ## -4 or even -3 are good initializations
        self.w_sigma = tf.log(1.0 + tf.exp(self.w_rho))
        self.w_distr = tf.distributions.Normal(loc=self.w_mean, scale=self.w_sigma)

        self.b_mean = tf.Variable(init_mu*tf.ones([n_filters]))#tf.Variable(tf.random_normal([self.n_outputs], 0.0, 0.01))
        self.b_rho = tf.Variable(init_rho*tf.ones([n_filters]))#tf.Variable(tf.random_normal([self.n_outputs], -4, 0.1))
        self.b_sigma = tf.log(1.0 + tf.exp(self.b_rho))
        self.b_distr = tf.distributions.Normal(loc=self.b_mean, scale=self.b_sigma)

    

        self.w_prior_mean = tf.Variable(tf.zeros_like(self.w_mean,dtype=tf.float32),trainable=False)
        self.w_prior_sigma = tf.Variable(tf.ones_like(self.w_sigma,dtype=tf.float32),trainable=False)
        self.b_prior_mean = tf.Variable(tf.zeros_like(self.b_mean,dtype=tf.float32),trainable=False)
        self.b_prior_sigma = tf.Variable(tf.ones_like(self.b_sigma,dtype=tf.float32),trainable=False)
        self.w_prior_distr = tf.distributions.Normal(loc=self.w_prior_mean, scale=self.w_prior_sigma)
        self.b_prior_distr = tf.distributions.Normal(loc=self.b_prior_mean, scale=self.b_prior_sigma)

        self.init_learned_dist()

        #self.w_prior_distr = tf.distributions.Normal(loc=0.0, scale=1.0)
        #self.b_prior_distr = tf.distributions.Normal(loc=0.0, scale=1.0)


        ## No need to limit ourselves to diagonal gaussian priors!

        # Mixture of gaussians
        #tfd = tf.distributions
        #pi = 0.5
        #self.w_prior_distr = tf.contrib.distributions.Mixture(cat=tfd.Categorical(probs=[pi, 1.-pi]), components=[tfd.Normal(loc=0.0, scale=1.0), tfd.Normal(loc=0.0, scale=float(np.exp(-6.0)))])
        #self.b_prior_distr = tf.contrib.distributions.Mixture(cat=tfd.Categorical(probs=[pi, 1.-pi]), components=[tfd.Normal(loc=0.0, scale=1.0), tfd.Normal(loc=0.0, scale=float(np.exp(-6.0)))])

    def get_mean_list(self):
        return [self.w_mean,self.b_mean]

    def get_var_list(self):
        return [self.w_rho,self.b_rho]

    def get_prior_mean_list(self):
        return [self.w_prior_mean,self.b_prior_mean]

    def get_prior_var_list(self):
        return [self.w_prior_sigma,self.b_prior_sigma]


    def init_learned_dist(self,alpha=0.5):
        self.w_learned_dist = []
        self.b_learned_dist = []
        self.learned_mean = {}
        self.learned_var = {}
        alpha_list = []
        self.alpha = tf.Variable(alpha,trainable=False)
        with tf.variable_scope('prior',reuse=tf.AUTO_REUSE) as scope:
            for n in range(self.num_task):
                w_mean = tf.Variable(tf.zeros_like(self.w_mean,dtype=tf.float32),trainable=False)
                w_sigma = tf.Variable(tf.ones_like(self.w_sigma,dtype=tf.float32),trainable=False)
                b_mean = tf.Variable(tf.zeros_like(self.b_mean,dtype=tf.float32),trainable=False)
                b_sigma = tf.Variable(tf.ones_like(self.b_sigma,dtype=tf.float32),trainable=False)
                w_dist = tf.distributions.Normal(loc=w_mean, scale=w_sigma)
                b_dist = tf.distributions.Normal(loc=b_mean, scale=b_sigma)
                self.w_learned_dist.append(w_dist)
                self.b_learned_dist.append(b_dist)
                self.learned_mean[n] = [w_mean,b_mean]
                self.learned_var[n] = [w_sigma,b_sigma]
                if n < self.num_task - 1:
                    alpha_list.append((1.0 - self.alpha) / (self.num_task - 1.0))
                else:
                    alpha_list.append(self.alpha)
            self.alpha_list = tf.Variable(alpha_list,trainable=False)


    def get_learned_dist(self):
        return self.learned_mean, self.learned_var

    def get_sampled_weights(self):
        #w = self.w_mean + (self.w_sigma * tf.random_normal([self.n_inputs, self.n_outputs], 0.0, 1.0, tf.float32))
        #b = self.b_mean + (self.b_sigma * tf.random_normal([self.n_outputs], 0.0, 1.0, tf.float32))
        w = self.w_distr.sample()
        b = self.b_distr.sample()
        return w,b


    def _build(self, inputs, activation,sample=False, drop_out=False,pre_w=None,pre_b=None):
        """
        Constructs the graph for the layer.

        Args:
          inputs: `tf.Tensor` input to be used by the MLP
          sample: boolean; whether to compute the output of the MLP by sampling its weights from their posterior or by returning a MAP estimate using their mean value

        Returns:
          output: `tf.Tensor` output averaged across n_samples, if sample=True, else MAP output
          log_probs: `tf.Tensor` KL loss of the network
        """

        if sample:
            w = self.w_distr.sample()
            b = self.b_distr.sample()
            #w = self.w_mean + (self.w_sigma * tf.random_normal([self.n_inputs, self.n_outputs], 0.0, 1.0, tf.float32))
            #b = self.b_mean + (self.b_sigma * tf.random_normal([self.n_outputs], 0.0, 1.0, tf.float32))
        else:
            w = self.w_mean
            b = self.b_mean

        z = tf.nn.conv2d(inputs, filter=w, strides=[1, self.stride, self.vert_stride, 1],
                     padding='SAME', data_format='NHWC') + b
        if drop_out and pre_w is not None:
            z = tf.nn.dropout(z,keep_prob=0.5) + tf.nn.conv2d(inputs, filter=pre_w, strides=[1, self.stride, self.vert_stride, 1],
                     padding='SAME', data_format='NHWC') + pre_b
            
        z = activation(z)
        #log_probs_w = self.w_distr.log_prob(w) - self.w_prior_distr.log_prob(w)
        #log_probs_b = self.b_distr.log_prob(b) - self.b_prior_distr.log_prob(b)
        log_probs_w = tf.distributions.kl_divergence(self.w_distr,self.w_prior_distr)
        log_probs_b = tf.distributions.kl_divergence(self.b_distr,self.b_prior_distr)

        return z, log_probs_w,log_probs_b


