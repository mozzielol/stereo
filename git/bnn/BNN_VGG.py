import tensorflow as tf
import sonnet as snt
from tqdm import tqdm
import numpy as np
from sklearn.mixture import GaussianMixture as GMM
from sklearn.mixture import BayesianGaussianMixture as DPGMM
from bnn.BNNLayer import *
from copy import deepcopy
import matplotlib.pyplot as plt
from IPython import display
from bnn.load_vgg import VGG

class BNN_VGG(snt.AbstractModule):
    """
    Implementation of an Bayesian MLP with structure [n_inputs] -> hidden_units -> [n_outputs], with a diagonal gaussian
    posterior. The weights are initialized to mean init_mu mean and standard deviation log(1+exp(init_rho)).
    """

    def __init__(self,n_inputs, n_outputs, hidden_units=[], init_mu=0.0, init_rho=0.0, activation=tf.nn.relu,
                         last_activation=tf.nn.softmax, prior_dist=None, num_task=5 ,name="BNN_VGG"):
        super(BNN_VGG, self).__init__(name=name)
        self.vgg = VGG()
        self.num_task = num_task
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.activation = activation
        self.last_activation = last_activation
        self.hidden_units = [n_inputs] + hidden_units + [n_outputs]
        
        
        self.kl = False
    
        self.gstep = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        self.layers = []
        self.var_list = []
        self.prior_mean_list = []
        self.prior_var_list = []
        self.uncertainty_list = []
        self.full_var_list = []
        self.mask_list = []
        self.merge_params = {}
        self.merge_var = {}
        self.merge_uncertainty = {}
        self.learned_mean = {}
        self.learned_var = {}
        for n in range(num_task):
            self.learned_mean[n] = []
            self.learned_var[n] = []


        for i in range(1, len(self.hidden_units)):
            self.layers.append( BNNLayer(self.hidden_units[i-1], self.hidden_units[i], init_mu=init_mu, init_rho=init_rho,num_task=num_task) )
            self.var_list += self.layers[i-1].get_var_list()
            self.prior_mean_list += self.layers[i-1].get_prior_mean_list()
            self.prior_var_list += self.layers[i-1].get_prior_var_list()
            self.uncertainty_list += self.layers[i-1].get_uncertainty_list()
            self.full_var_list += self.layers[i-1].get_full_var_list()
            self.mask_list += self.layers[i-1].get_mask_list()
            mean,var = self.layers[i-1].get_learned_dist()
            for n in range(num_task):
                self.learned_mean[n] += mean[n]
                self.learned_var[n] += var[n]



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
        self.store_fisher()


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

    def set_learned_dist(self,sess):
        for n in range(self.num_task):
            for idx in range(len(self.learned_mean[n])):
                sess.run(self.learned_mean[n][idx].assign(self.merge_var[n][idx]))
                sess.run(self.learned_var[n][idx].assign(tf.log(1.0 + tf.exp(self.merge_uncertainty[n][idx]))))

    def store_fisher(self):
        self.Pre_F_accum = deepcopy(self.F_accum)


    def store_merge_params(self,key):
        self.merge_params[key] = []
        for v in self.full_var_list:
            self.merge_params[key].append(v.eval())

    def stroe_gauss(self,key):
        print('Stroing Task %d...'%key)
        self.merge_var[key] = []
        self.merge_uncertainty[key] = []
        for v in self.var_list:
            self.merge_var[key].append(v.eval())
        for v in self.uncertainty_list:
            self.merge_uncertainty[key].append(v.eval())

    def store_gauss(self):
        self.back_merge_var = deepcopy(self.merge_var)
        self.back_merge_uncertainty = deepcopy(self.merge_uncertainty)

    def restore_gauss(self):
        self.merge_var = deepcopy(self.back_merge_var)
        self.merge_uncertainty = deepcopy(self.back_merge_uncertainty)

    def em_stereo(self,n_component=1,dp=True,thresh_hold=0.4):
        self.num_params = 0
        #The range of len(params)
        _step = 0
        for var_idx in tqdm(range(len(self.merge_var[0]))):

            for x_v in range(len(self.merge_var[0][var_idx])):
                print('Step %d'%_step,end='\r')
                _step += 1
                try:
                    
                    for y_v in range(len(self.merge_var[0][var_idx][x_v])):
                        #print('cluster weights ....%d'%var_idx)
                        dist = []
                        for task_idx in range(len(self.merge_var)):
                            nor = np.random.normal(self.merge_var[task_idx][var_idx][x_v][y_v],np.log(1.0+np.exp(self.merge_uncertainty[task_idx][var_idx][x_v][y_v])),200)
                            dist.append(nor)
                        
                        dist = np.array(np.asmatrix(np.concatenate(dist)).T)
                        if dp:
                            print('Initializing DPGMM%d ... '%_step,end='\r')
                            gmm = DPGMM( max_iter=1000,  n_components=n_component, covariance_type='spherical')
                        else:
                            gmm = GMM( max_iter=200,  n_components=n_component, covariance_type='spherical')
                        gmm.fit(dist)
                        new_idx_list = []
                        for task_idx in range(len(self.merge_var)):
                            #if dp:
                            #Strategy 1. Set threshold
                            predict_probability = gmm.predict_proba(np.array(self.merge_var[task_idx][var_idx][x_v][y_v]).reshape(-1,1))
                            f_ = True
                            
                            while f_:
                                #if gmm.weights_[np.argmax(predict_probability)] > ( 1 / len(self.merge_var)):
                                if gmm.weights_[np.argmax(predict_probability)] > thresh_hold:
                                    new_idx = np.argmax(predict_probability)
                                    f_ = False
                                else:
                                    predict_probability[0][np.argmax(predict_probability)] = 0.0
                                    self.num_params += 1
                            

                        #else:
                        #    new_idx = gmm.predict(np.array(self.merge_var[task_idx][var_idx][x_v][y_v]).reshape(-1,1))
                        #    if new_idx in new_idx_list:
                                self.num_params += 1
                            new_idx_list.append(new_idx)
                            self.merge_var[task_idx][var_idx][x_v][y_v] = gmm.means_[new_idx]
                            self.merge_uncertainty[task_idx][var_idx][x_v][y_v] = np.log(np.exp(gmm.covariances_[new_idx]) - 1.0)


                except TypeError:
                    dist = []
                    
                    
                    for task_idx in range(len(self.merge_var)):
                        nor = np.random.normal(self.merge_var[task_idx][var_idx][x_v],np.log(1.0+np.exp(self.merge_uncertainty[task_idx][var_idx][x_v])),200)
                        dist.append(nor)
                    dist = np.array(np.asmatrix(np.concatenate(dist)).T)
                    if dp:
                        print('Initializing DPGMM%d ... '%_step,end='\r')
                        gmm = DPGMM( max_iter=200,  n_components=n_component, covariance_type='spherical')
                    else:
                        gmm = GMM( max_iter=200,  n_components=n_component, covariance_type='spherical')
                    gmm.fit(dist)
                    new_idx_list = []
                    for task_idx in range(len(self.merge_var)):
                        #if dp:
                        #Strategy 1. Set threshold
                        predict_probability = gmm.predict_proba(np.array(self.merge_var[task_idx][var_idx][x_v]).reshape(-1,1))
                        f_ = True
                        while f_:
                            #if gmm.weights_[np.argmax(predict_probability)] > ( 1 / len(self.merge_var)):
                            if gmm.weights_[np.argmax(predict_probability)] > thresh_hold:
                                new_idx = np.argmax(predict_probability)
                                f_ = False
                            else:
                                predict_probability[0][np.argmax(predict_probability)] = 0.0
                                self.num_params += 1

                    #else:
                    #    new_idx = gmm.predict(np.array(self.merge_var[task_idx][var_idx][x_v]).reshape(-1,1))
                    #    if new_idx in new_idx_list:
                    #        self.num_params += 1
                        new_idx_list.append(new_idx)
                        self.merge_var[task_idx][var_idx][x_v] = gmm.means_[new_idx]
                        self.merge_uncertainty[task_idx][var_idx][x_v] = np.log(np.exp(gmm.covariances_[new_idx]) - 1.0)
                    
                    
                

    def set_merged_params(self,sess,num):
        for idx in range(len(self.merge_var[num])):
            sess.run(self.var_list[idx].assign(self.merge_var[num][idx]))
            sess.run(self.uncertainty_list[idx].assign(self.merge_uncertainty[num][idx]))



    def imm_mean(self,sess,alpha=0.5):
        for v in range(len(self.star_var)):
            current_mean = self.merge_var[1][v]
            current_uncertainty = np.log(1.0 + np.exp(self.merge_uncertainty[1][v]))
            pre_mean = self.merge_var[0][v]
            pre_uncertainty = np.log(1.0 + np.exp(self.merge_uncertainty[0][v]))
            new_mean = (current_mean * (1-alpha) + pre_mean * alpha) 
            new_var = (pre_uncertainty**2 * alpha**2 + current_uncertainty**2 * (1 - alpha)**2)
            new_var = np.log(np.exp(new_var - 1.0))
            sess.run(self.var_list[v].assign(new_mean))
            sess.run(self.uncertainty_list[v].assign(new_var))

    def dist_merge(self,sess, alpha=0.5):
        for v in range(len(self.star_var)):
            current_mean = self.merge_var[1][v]
            current_uncertainty = np.log(1.0 + np.exp(self.merge_uncertainty[1][v]))
            pre_mean = self.merge_var[0][v]
            pre_uncertainty = np.log(1.0 + np.exp(self.merge_uncertainty[0][v]))
            new_mean = (current_mean * pre_uncertainty * (1-alpha) + pre_mean * current_uncertainty * alpha) / (pre_uncertainty * alpha + current_uncertainty * (1-alpha))
            new_var = (pre_uncertainty**2 * alpha**2 + current_uncertainty**2 * (1 - alpha)**2)
            new_var = np.log(np.exp(new_var - 1.0))
            sess.run(self.var_list[v].assign(new_mean))
            sess.run(self.uncertainty_list[v].assign(new_var))





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

    def set_alpha(self,sess,alpha):
        for l in self.layers:
            sess.run(l.alpha.assign(alpha))


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
        inputs = self.vgg.inference(x)
        for i in range(len(self.layers)):
            inputs, _ ,_= self.layers[i](inputs)
            if i == len(self.layers)-1:
                inputs = self.last_activation(inputs)
            else:
                inputs = self.activation(inputs)
        self.y = inputs

        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def _build(self, inputs, vgg=None,sample=False, n_samples=1, targets=None, 
        loss_function=lambda y, y_target: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=y),kl=False ):
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
            #x = inputs
            x = self.vgg.inference(inputs)
            if vgg is not None:
                x = vgg.predict(x)
            for i in range(len(self.layers)):
                if kl:
                    f = i*2
                    x, log_probs_w,log_probs_b = self.layers[i](x, sample,kl,(self.Pre_F_accum[f],self.F_accum[f],self.Pre_F_accum[f+1],self.F_accum[f+1]))
                else:
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

        if kl:
            avg_loss = 0.0

        return output, log_probs, avg_loss,kl_diver

    def set_vanilla_loss(self,log_probs,nll,num_batches):
        #uncertain_loss = 0.0
        #for v in self.uncertainty_list:
        #    uncertain_loss += tf.reduce_sum(v**2)
        self.loss = (self.lams/2)*log_probs/num_batches + nll #+ uncertain_loss
        optim = tf.train.AdamOptimizer(learning_rate=0.001)
        self.train_op = optim.minimize( self.loss ,global_step=self.gstep)

    def set_kl_loss(self,log_probs,nll,num_batches):
        self.kl_loss = log_probs +  nll * 10**-20
        optim = tf.train.AdamOptimizer(learning_rate=0.001)
        self.kl_train_op = optim.minimize( self.kl_loss ,global_step=self.gstep)

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




