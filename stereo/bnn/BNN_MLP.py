import tensorflow as tf
import sonnet as snt

import numpy as np

from bnn.BNNLayer import *
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display

class BNN_MLP(snt.AbstractModule):
    """
    Implementation of an Bayesian MLP with structure [n_inputs] -> hidden_units -> [n_outputs], with a diagonal gaussian
    posterior. The weights are initialized to mean init_mu mean and standard deviation log(1+exp(init_rho)).
    """

    def __init__(self, n_inputs, n_outputs, hidden_units=[], init_mu=0.0, init_rho=0.0, activation=tf.nn.relu,
                         last_activation=tf.nn.softmax, prior_dist=None,name="BNN_MLP"):
        super(BNN_MLP, self).__init__(name=name)


        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.last_activation = last_activation
        self.hidden_units = [n_inputs] + hidden_units + [n_outputs]
        
        

    
        self.gstep = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        self.layers = []
        self.var_list = []
        self.prior_mean_list = []
        self.prior_var_list = []
        self.uncertainty_list = []
        self.full_var_list = []
        self.mask_list = []

        for i in range(1, len(self.hidden_units)):
            self.layers.append( BNNLayer(self.hidden_units[i-1], self.hidden_units[i], init_mu=init_mu, init_rho=init_rho) )
            self.var_list += self.layers[i-1].get_var_list()
            self.prior_mean_list += self.layers[i-1].get_prior_mean_list()
            self.prior_var_list += self.layers[i-1].get_prior_var_list()
            self.uncertainty_list += self.layers[i-1].get_uncertainty_list()
            self.full_var_list += self.layers[i-1].get_full_var_list()
            self.mask_list += self.layers[i-1].get_mask_list()


        self.F_accum = []
        self.star_var = []
        self.star_mask_list = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))
            self.star_var.append(np.zeros(self.var_list[v].get_shape().as_list()))

        self.start_uncertain_list = []
        for v in range(len(self.uncertainty_list)):
            self.start_uncertain_list.append(np.zeros(self.uncertainty_list[v].get_shape().as_list()))

        for v in range(len(self.mask_list)):
            self.star_mask_list.append(np.ones(self.mask_list[v].get_shape().as_list()))

        
        self.lams = tf.get_variable('lams',initializer=tf.constant(0.1),trainable=False)



    def store(self):
        self.star_full_var = []
        self.start_uncertain_list = []
        self.star_var = []
        for v in self.full_var_list:
            self.star_full_var.append(v.eval())
        for v in self.uncertainty_list:
            self.start_uncertain_list.append(v.eval())
        for v in self.var_list:
            self.star_var.append(v.eval())


    def store_fisher(self):
        self.Pre_F_accum = deepcopy(self.F_accum)




    def dist_mean_merge(self,sess,alpha=0.5):

        for v in range(len(self.kalman_mean)):
            new_var = []
            new_uncertainty = []
            new_var = self.star_var[v] * alpha + self.kalman_mean[v] * (1-alpha)
            new_uncertainty = (self.start_uncertain_list[v]**2) * (alpha**2) + ((1 - alpha)**2) * (self.kalman_uncertain[v]**2)
            new_uncertainty = tf.log(tf.exp(new_uncertainty) - 1.0)
            sess.run(self.var_list[v].assign(new_var))
            sess.run(self.uncertainty_list[v].assign(new_uncertainty))

    def dist_fisher_merge(self,sess,alpha=0.5):


        for v in range(len(self.F_accum)):
            new_var = []
            new_uncertainty = []
            w1 = np.exp(self.Pre_F_accum[v],dtype=np.float32)
            w2 = np.exp(self.F_accum[v],dtype=np.float32)

            total_weights = w1 * alpha + w2 * (1 - alpha)

            new_var = self.star_var[v] * alpha * w1 / total_weights + self.kalman_mean[v] * (1-alpha) * w2 / total_weights
            new_uncertainty = (self.start_uncertain_list[v]**2) * (alpha**2) * (w1 / total_weights) + ((1 - alpha)**2) * (self.kalman_uncertain[v]**2)* (w2 / total_weights)
            new_uncertainty = tf.log(tf.exp(new_uncertainty) - 1.0)
            sess.run(self.var_list[v].assign(new_var))
            sess.run(self.uncertainty_list[v].assign(new_uncertainty))


    def set_mask(self,sess,task_num,sigma=1.0):
        for v in range(len(self.star_mask_list)):
            num = len(self.star_mask_list[v])
            if task_num == 1:
                self.star_mask_list[v][num//2:] = sigma
            else:
                self.star_mask_list[v][num//2:] = 1.0
            sess.run(self.mask_list[v].assign(self.star_mask_list[v]))



    def restore(self,sess):
        if hasattr(self,'star_full_var'):
            for v in range(len(self.full_var_list)):
                sess.run(self.full_var_list[v].assign(self.star_full_var[v]))


    def set_prior(self,sess):
        if hasattr(self,'star_var'):
            for v in range(len(self.prior_mean_list)):
                sess.run(self.prior_mean_list[v].assign(self.star_var[v]))
                sess.run(self.prior_var_list[v].assign(tf.log(1.0 + tf.exp(self.start_uncertain_list[v]))))


    def set_fisher_graph(self,x,y_):
        self.x = x
        inputs = x
        for i in range(len(self.layers)):
            inputs, _ ,_= self.layers[i](inputs)
            if i == len(self.layers)-1:
                inputs = self.last_activation(inputs)
            else:
                inputs = self.activation(inputs)
        self.y = inputs

        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def _build(self, inputs, sample=False, n_samples=5, targets=None, 
        loss_function=lambda y, y_target: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=y) ):
        """
        Constructs the MLP graph.

        Args:
          inputs: `tf.Tensor` input to be used by the MLP
          sample: boolean; whether to compute the output of the MLP by sampling its weights from their posterior or by returning a MAP estimate using their mean value
          n_samples: number of sampled networks to average output of the MLP over
          targets: target outputs of the MLP, used to compute the loss function on each sampled network
          loss_function: lambda function to compute the loss of the network given its output and targets.

        Returns:
          output: `tf.Tensor` output averaged across n_samples, if sample=True, else MAP output
          log_probs: `tf.Tensor` KL loss of the network
          avg_loss: `tf.Tensor` average loss across n_samples, computed using `loss_function'
        """

        log_probs = 0.0
        avg_loss = 0.0
        kl_diver = []

        if not sample:
            n_samples = 1

        output = 0.0 ## avg. output logits
        for ns in range(n_samples):
            x = inputs
            for i in range(len(self.layers)):
                x, log_probs_w,log_probs_b = self.layers[i](x, sample)
                if i == len(self.layers)-1:
                    x = self.last_activation(x)
                else:
                    x = self.activation(x)
                kl_diver.append(log_probs_w)
                kl_diver.append(log_probs_b)
                log_probs += tf.reduce_sum(log_probs_w + log_probs_b)

            output += x

            if targets is not None:
                if loss_function is not None:
                    loss = tf.reduce_mean(loss_function(x, targets), 0)

                    #loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=targets, logits=x))
                    #loss = 0.5*tf.reduce_mean(tf.reduce_sum( tf.square(targets-x), 1), 0)
                    avg_loss += loss


        log_probs /= n_samples
        avg_loss /= n_samples
        output /= n_samples


        return output, log_probs, avg_loss,kl_diver

    def set_vanilla_loss(self,log_probs,nll,num_batches):
        #uncertain_loss = 0.0
        #for v in self.uncertainty_list:
        #    uncertain_loss += tf.reduce_sum(v**2)
        self.loss = (self.lams/2)*log_probs/num_batches + nll #+ uncertain_loss
        optim = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optim.minimize( self.loss ,global_step=self.gstep)


    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('accuracy',self.accuracy)
            tf.summary.histogram('histogram',self.loss)
            self.summary_op = tf.summary.merge_all()




    def set_ewc_loss(self, kl_diver,nll,num_batches):
        # elastic weight consolidation
        # lam is weighting for previous task(s) constraints

        if not hasattr(self, "ewc_loss"):
            self.ewc_loss = nll

        for v in range(len(self.F_accum)):
            #self.ewc_loss += (self.lams/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),kl_diver[v]))/num_batches
            self.ewc_loss += (self.lams/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.var_list[v] - self.star_var[v])))
            self.ewc_loss += (self.lams/2) * tf.reduce_sum(tf.multiply(self.F_accum[v].astype(np.float32),tf.square(self.uncertainty_list[v] - self.start_uncertain_list[v])))

        self.ewc_train_op = tf.train.AdamOptimizer(0.001).minimize(self.ewc_loss)



    def compute_fisher(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter

        # initialize Fisher information for most recent task
        self.F_accum = []
        for v in range(len(self.var_list)):
            self.F_accum.append(np.zeros(self.var_list[v].get_shape().as_list()))

        # sampling a random class from softmax
        probs = tf.nn.softmax(self.y)
        class_ind = tf.to_int32(tf.multinomial(tf.log(probs), 1)[0][0])

        if(plot_diffs):
            # track differences in mean Fisher info
            F_prev = deepcopy(self.F_accum)
            mean_diffs = np.zeros(0)

        for i in range(num_samples):
            # select random input image
            im_ind = np.random.randint(imgset.shape[0])
            # compute first-order derivatives
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), self.var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
            # square the derivatives and add to total
            for v in range(len(self.F_accum)):
                self.F_accum[v] += np.square(ders[v])
            if(plot_diffs):
                if i % disp_freq == 0 and i > 0:
                    # recording mean diffs of F
                    F_diff = 0
                    for v in range(len(self.F_accum)):
                        F_diff += np.sum(np.absolute(self.F_accum[v]/(i+1) - F_prev[v]))
                    mean_diff = np.mean(F_diff)
                    mean_diffs = np.append(mean_diffs, mean_diff)
                    for v in range(len(self.F_accum)):
                        F_prev[v] = self.F_accum[v]/(i+1)
                    plt.plot(range(disp_freq+1, i+2, disp_freq), mean_diffs)
                    plt.xlabel("Number of samples")
                    plt.ylabel("Mean absolute Fisher difference")
                    display.display(plt.gcf())
                    display.clear_output(wait=True)

        # divide totals by number of samples
        for v in range(len(self.F_accum)):
            self.F_accum[v] /= num_samples




