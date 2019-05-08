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
from bnn.utils import create_mixture
from scipy.stats import norm 


class BNN_MLP(snt.AbstractModule):
    """
    Implementation of an Bayesian MLP with structure [n_inputs] -> hidden_units -> [n_outputs], with a diagonal gaussian
    posterior. The weights are initialized to mean init_mu mean and standard deviation log(1+exp(init_rho)).
    """

    def __init__(self, n_inputs, n_outputs, hidden_units=[], num_task=2, init_mu=0.0, init_rho=0.0, activation=tf.nn.relu,
                         last_activation=tf.nn.softmax,name="BNN_MLP"):
        super(BNN_MLP, self).__init__(name=name)

        self.num_task = num_task
        self.activation = activation
        self.last_activation = last_activation
        self.hidden_units = [n_inputs] + hidden_units + [n_outputs]
        self.gstep = tf.Variable(0,dtype=tf.int32,trainable=False,name='global_step')
        self.lams = tf.get_variable('lams',initializer=tf.constant(0.1),trainable=False)
        #Initialize the storage
        self.tensor_mean = []
        self.tensor_var = []
        self.tensor_prior_mean = []
        self.tensor_prior_var = []
        self.tensor_learned_mean = {}
        self.tensor_learned_var = {}

        self.params_mean = {}
        self.params_var = {}
        self.FISHER = {}

        self.layers = []
        self.x_placeholder = tf.placeholder(tf.float32, shape=[None, 784])
        self.y_placeholder = tf.placeholder(tf.float32, shape=[None, 10])

        for i in range(num_task):
            self.tensor_learned_mean[i] = []
            self.tensor_learned_var[i] = []

        for i in range(1, len(self.hidden_units)):
            self.layers.append( BNNLayer(self.hidden_units[i-1], self.hidden_units[i], init_mu=init_mu, init_rho=init_rho,num_task=num_task))

            self.tensor_mean += self.layers[i-1].get_mean_list()
            self.tensor_var += self.layers[i-1].get_var_list()
            self.tensor_prior_mean += self.layers[i-1].get_prior_mean_list()
            self.tensor_prior_var += self.layers[i-1].get_prior_var_list()
            
            mean,var = self.layers[i-1].get_learned_dist()
            for n in range(num_task):
                self.tensor_learned_mean[n] += mean[n]
                self.tensor_learned_var[n] += var[n]
        
        F_accum = []
        for v in range(len(self.tensor_mean)):
            F_accum.append(np.zeros(self.tensor_mean[v].get_shape().as_list()))

        for i in range(num_task):
            self.FISHER[i] = deepcopy(F_accum)

        
        self.num_batches = 100.0

    def initialize_default_params(self,sess):
        sess.run( tf.global_variables_initializer() )
        self.params_mean = {}
        self.params_var = {}
        
        
        '''
        op = []
        for i in range(len(self.layers)):
            op.append(self.tensor_mean[i].initializer)
            op.append(self.tensor_var[i].initializer)
        sess.run(op)
        '''

    def print_sampled(self):
        print(self.layers[1].get_sampled_weights()[0].eval())


    def transform_var(self,var):
        return np.log(1.0 + np.exp(var))

    def retransform_var(self,var):
        return np.log(np.exp(var) - 1.0)

    def store_fisher(self,num):
        self.FISHER[num] = deepcopy(self.F_accum)

    def store_params(self,num):
        #if num == 1 : pass
        mean_list = []
        var_list = []
        for v in self.tensor_mean:
            mean_list.append(v.eval())
        for v in self.tensor_var:
            var_list.append(v.eval())
        self.params_mean[num] = mean_list
        self.params_var[num] = var_list

    def back_up_params(self):
        return
        self.back_up_mean = deepcopy(self.params_mean)
        self.back_up_var = deepcopy(self.params_var)

    def restore_last_params(self,sess):
        idx = max(self.params_mean,key=int)
        for v in range(len(self.params_mean[idx])):
            sess.run(self.tensor_mean[v].assign(self.params_mean[idx][v]))
            sess.run(self.tensor_var[v].assign(self.params_var[idx][v]))

    def restore_params_from_backup(self):
        return
        try:
            self.params_mean = deepcopy(self.back_up_params)
            self.params_var = deepcopy(self.back_up_var)
        except AttributeError:
            print('Have not backup params yet')


    def restore_first_params(self,sess,clean=True):
        
        for v in range(len(self.params_mean[0])):
            sess.run(self.tensor_mean[v].assign(self.params_mean[0][v]))
            sess.run(self.tensor_var[v].assign(self.params_var[0][v]))
        if clean:
            pop_list = []
            for key in self.params_mean.keys():
                if key == 0:
                    pass
                else:
                    pop_list.append(key)
            for key in pop_list:
                    self.params_mean.pop(key)
                    self.params_var.pop(key)



    def set_learned_dist(self,sess):
        for n in range(self.num_task):
            for idx in range(len(self.tensor_learned_mean[n])):
                sess.run(self.tensor_learned_mean[n][idx].assign(self.params_mean[n][idx]))
                sess.run(self.tensor_learned_var[n][idx].assign(tf.log(1.0 + tf.exp(self.params_var[n][idx]))))

    def set_task_params(self,sess,num):
        for idx in range(len(self.params_mean[num])):
            sess.run(self.tensor_mean[idx].assign(self.params_mean[num][idx]))
            sess.run(self.tensor_var[idx].assign(self.params_var[num][idx]))

    def set_alpha(self,sess,alpha):
        if alpha > 1:
            raise ValueError('alpha cannot larger than 1')
        else:
            for l in self.layers:
                sess.run(l.alpha.assign(alpha))

    def set_prior(self,sess,idx):
        #idx = max(self.params_mean,key=int)
        for v in range(len(self.tensor_prior_mean)):
            sess.run(self.tensor_prior_mean[v].assign(self.params_mean[idx][v]))
            sess.run(self.tensor_prior_var[v].assign(self.transform_var(self.params_var[idx][v])))


    def set_fisher_graph(self,x,y_):
        self.x = x
        self.y_ = y_
        inputs = x
        for i in range(len(self.layers)):
            if i == len(self.layers)-1:
                inputs, _ ,_= self.layers[i](inputs,self.last_activation,sample=False)
            else:
                inputs, _ ,_= self.layers[i](inputs,self.activation,sample=False)
        self.y = inputs
        correct_prediction = tf.equal(tf.argmax(self.y,1), tf.argmax(y_,1))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))




    def set_uncertain_prediction(self):
        marks = 0.0
        x = self.x_placeholder
        y_ = self.y_placeholder
        inputs = x
        for i in range(len(self.layers)):
            if i == len(self.layers)-1:
                inputs, _ ,_= self.layers[i](inputs,self.last_activation,sample=True)
            else:
                inputs, _ ,_= self.layers[i](inputs,self.activation,sample=True)
        self.predictions = inputs
        #self.predictions = tf.stack(make_prediction(x),axis=0)
        #avg_prd = tf.reduce_mean(predictions,axis=0)
        #risk = self.calc_risk(predictions)[0]
        
        correct_prediction = tf.equal(tf.argmax(self.predictions,1), tf.argmax(y_,1))
        self.em_accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        #self.uncertainty = risk

    def reduce_entropy(self,X, axis=-1):
        """
        calculate the entropy over axis and reduce that axis
        :param X:
        :param axis:
        :return:
        """
        return -1 * tf.reduce_sum(X * tf.log(X+1E-12), axis=axis)

    def calc_risk(self,preds, labels=None):
        """
        Calculates the parameters we can possibly use to examine risk of a neural net
        :param preds: preds in shape [num_runs, num_batch, num_classes]
        :param labels:
        :return:
        """
        if isinstance(preds, list):
            preds = tf.stack(preds)
        # preds in shape [num_runs, num_batch, num_classes]
        num_runs, num_batch = preds.shape[:2]

        ave_preds = tf.mean(preds, axis=0)
        pred_class = tf.argmax(ave_preds, axis=1)

        # entropy of the posterior predictive
        entropy = self.reduce_entropy(ave_preds, axis=1)

        # Expected entropy of the predictive under the parameter posterior
        entropy_exp = tf.mean(self.reduce_entropy(preds, axis=2), axis=0)
        mutual_info = entropy - entropy_exp  # Equation 2 of https://arxiv.org/pdf/1711.08244.pdf

        # Average and variance of softmax for the predicted class
        variance = tf.math.reduce_std(preds[:, range(num_batch), pred_class], 0)
        ave_softmax = tf.reduce_mean(preds[:, range(num_batch), pred_class], 0)

        # And calculate accuracy if we know the labels
        if labels is not None:
            correct = tf.equal(pred_class, labels)
        else:
            correct = None
        return entropy, mutual_info, variance, ave_softmax, correct


    def uncertain_predict(self,sess,x,y_):
        marks = []
        uncertainty = []
        for key in self.params_mean.keys():
            self.set_task_params(sess,key)
            ma, un = self.set_uncertain_prediction(x,y_)
            marks.append(ma)
            uncertainty.append(un)
        marks = np.array(marks)
        uncertainty = np.array(uncertainty)
        idx = np.argmin(uncertainty)
        marks = marks[idx]
        correct_prediction = tf.equal(tf.argmax(marks,1), tf.argmax(y_,1))
        return tf.reduce_mean(tf.cast(correct_prediction, tf.float32))



    def _build(self, inputs,sample=False, n_samples=1, targets=None, drop_out=False,
        loss_function=lambda y, y_target: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=y)):
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
        pre_w = None
        pre_b = None
        if not sample:
            n_samples = 1

        output = 0.0 ## avg. output logits
        for ns in range(n_samples):
            x = inputs
            for i in range(len(self.layers)):
                if drop_out :
                    try:
                        idx = max(self.params_mean,key=int)
                        pre_w = self.params_mean[idx][i*2]
                        pre_b = self.params_mean[idx][i*2+1]
                    except ValueError:
                        pre_w = None
                        pre_b = None
                    
                if i == len(self.layers)-1:
                    x, log_probs_w,log_probs_b = self.layers[i](x,self.last_activation,sample,drop_out,pre_w,pre_b)
                else:
                    x, log_probs_w,log_probs_b = self.layers[i](x,self.activation,sample,drop_out,pre_w,pre_b)
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


    def set_kl_loss(self,log_probs,nll,num_batches):
        self.kl_loss = log_probs + nll * 10**-20
        optim = tf.train.AdamOptimizer(learning_rate=0.001)
        self.kl_train_op = optim.minimize( self.kl_loss ,global_step=self.gstep)

    def set_drop_loss(self,log_probs,nll,num_batches):
        self.drop_loss = (self.lams/2)*log_probs/num_batches + nll #+ uncertain_loss
        optim = tf.train.AdamOptimizer(learning_rate=0.001)
        self.drop_train_op = optim.minimize( self.loss ,global_step=self.gstep)

    def apply_dropout(self,drop,idx):
        if drop and idx>0:
            return self.drop_train_op
        else:
            return self.train_op 

    def summary(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss',self.loss)
            tf.summary.scalar('accuracy',self.accuracy)
            tf.summary.histogram('histogram',self.loss)
            self.summary_op = tf.summary.merge_all()


    def compute_fisher(self,  x, sess,mb=1000):
        import math
        print('Computing Fisher ...')
        """new version of Calculating Fisher Matrix    

        Returns:
            FM: consist of [FisherMatrix(layer)] including W and b.
                and Fisher Matrix is dictionary of numpy array
                i.e. Fs[idxOfLayer]['W' or 'b'] -> numpy
        """
        FM = []
        data_size = x.shape[0]
        total_step = int(math.ceil(float(data_size)/mb))

        for step in range(total_step):
            ist = (step * mb) % data_size
            ied = min(ist + mb, data_size)
            y_sample = tf.reshape(tf.one_hot(tf.multinomial(self.y, 1), 10), [-1, 10])
            cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_sample, logits=self.y))

            for l in range(len(self.layers)):
                if len(FM) < 2*(l + 1):
                    FM.append(np.zeros(self.layers[l].w_mean.get_shape().as_list()))
                    FM.append(np.zeros(self.layers[l].b_mean.get_shape().as_list()))
                W_grad = tf.reduce_sum(tf.square(tf.gradients(cross_entropy,[self.layers[l].w_mean])), 0)
                b_grad = tf.reduce_sum(tf.square(tf.gradients(cross_entropy,[self.layers[l].b_mean])), 0)
                W_grad_val, b_grad_val = sess.run([W_grad, b_grad],
                                    feed_dict={ self.x:x[ist:ied]})
                FM[l*2] += W_grad_val
                FM[l*2+1] += b_grad_val
        FM = np.array(FM)
        FM += 1e-8
        
        self.F_accum = FM

    def CalculateFisherMatrix(self, imgset, sess, num_samples=200, plot_diffs=False, disp_freq=10):
        # computer Fisher information for each parameter
        print('Computing Fisher ...')
        # initialize Fisher information for most recent task
        self.F_accum = []
        var_list = []
        for v in range(len(self.tensor_mean)):
            var_list.append(self.tensor_mean[v])
        
        for v in range(len(var_list)):
            self.F_accum.append(np.zeros(var_list[v].get_shape().as_list()))

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
            ders = sess.run(tf.gradients(tf.log(probs[0,class_ind]), var_list), feed_dict={self.x: imgset[im_ind:im_ind+1]})
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
        #for v in range(len(self.F_accum)):
            #self.F_accum[v] /= num_samples

    def _st_smooth(self,var_idx,x_v,y_v=None,n_component=1,thresh_hold=0.3,dp=False,alpha=0.5):
        mixture_dist = []

        dist = [] # second
        for task_idx in range(self.num_task):
            if y_v is not None:
                mean = self.params_mean[task_idx][var_idx][x_v][y_v]
                var = self.transform_var(self.params_var[task_idx][var_idx][x_v][y_v])
            else:
                mean = self.params_mean[task_idx][var_idx][x_v]
                var = self.transform_var(self.params_var[task_idx][var_idx][x_v])
            mixture_dist.append({'kwargs':{'loc':mean,'scale':var}})
        
            nor = np.random.normal(mean,var,200) #second
            dist.append(nor)    #second
        dist = np.array(np.asmatrix(np.concatenate(dist)).T)    #second

        alpha_list = [(1-alpha)/(self.num_task-1)] * (self.num_task-1)
        alpha_list.append(alpha)
        sample = create_mixture(mixture_dist,alpha_list=alpha_list)
        if dp:
            gmm = DPGMM( max_iter=1000,  n_components=n_component, covariance_type='spherical')
        else:
            gmm = GMM( max_iter=500,  n_components=n_component, covariance_type='spherical')
        #gmm.fit(sample)
        gmm.fit(dist)
        
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
                    #self.num_merged_params += 1
            if new_idx in new_idx_list:
                self.num_merged_params += 1
            new_idx_list.append(new_idx)
            if y_v is not None:
                self.params_mean[task_idx][var_idx][x_v][y_v] = gmm.means_[new_idx]
                self.params_var[task_idx][var_idx][x_v][y_v] = self.retransform_var(gmm.covariances_[new_idx])
            else:
                self.params_mean[task_idx][var_idx][x_v] = gmm.means_[new_idx]
                self.params_var[task_idx][var_idx][x_v] = self.retransform_var(gmm.covariances_[new_idx])
        """
        if gmm.weights_ > 0.6:
            mean = 0.0
            var = 0.0
            for task_idx in range(self.num_task):
                if y_v is not None:
                    mean += self.params_mean[task_idx][var_idx][x_v][y_v] * 0.5
                    var += self.transform_var(self.params_var[task_idx][var_idx][x_v][y_v])**2 * 0.5**2
                else:
                    mean += self.params_mean[task_idx][var_idx][x_v] * 0.5 
                    var += self.transform_var(self.params_var[task_idx][var_idx][x_v])**2 * 0.5**2
                var = self.retransform_var(var)
            for task_idx in range(self.num_task):
                if y is not None:
                    self.params_mean[task_idx][var_idx][x_v][y_v] = mean
                    self.params_var[task_idx][var_idx][x_v][y_v] = var
                else:
                    self.params_mean[task_idx][var_idx][x_v] = mean
                    self.params_var[task_idx][var_idx][x_v] = var
            self.num_merged_params += 1
        """

    def st_smooth(self,n_component=1,dp=True,thresh_hold=0.3,alpha=0.5):
        self.num_merged_params = 0
        #The range of len(params)
        _step = 0
        for var_idx in tqdm(range(len(self.params_mean[0]))):
            for x_v in range(len(self.params_mean[0][var_idx])):
                print('Step %d'%_step,end='\r')
                _step += 1
                try:
                    for y_v in range(len(self.params_mean[0][var_idx][x_v])):
                        self._st_smooth(var_idx,x_v,y_v=y_v,n_component=n_component,thresh_hold=thresh_hold,dp=dp,alpha=alpha)

                except TypeError:
                    self._st_smooth(var_idx,x_v,n_component=n_component,thresh_hold=thresh_hold,dp=dp,alpha=alpha)
                    

    def imm_mean(self,sess,alpha=0.5):
        #sum_idx = max(self.params_mean,key=int)
        sum_idx = self.num_task - 1
        
        alpha_list = [(1-alpha)/(self.num_task-1)] * (self.num_task-1)
        alpha_list.append(alpha)
        ops = []

        for i in range(len(self.layers)):
            sum_mean = 0.0
            sum_var = 0.0
            for j in range(self.num_task):
                sum_mean += self.params_mean[j][i] * alpha_list[j]
                sum_var += (self.transform_var(self.params_var[j][i]))**2 * alpha_list[j]**2
            sum_var = self.retransform_var(sum_var)
            ops.append(self.tensor_mean[i].assign(sum_mean))
            ops.append(self.tensor_var[i].assign(sum_var))

        sess.run(ops)
        

    def imm_mode(self,sess,alpha=0.5):
        #sum_idx = max(self.params_mean,key=int)
        sum_idx = self.num_task - 1
        sum_mean = self.params_mean[sum_idx] 
        sum_var = self.params_var[sum_idx]
        alpha_list = [(1-alpha)/(self.num_task-1)] * (self.num_task-1)
        alpha_list.append(alpha)
        ops = []

        fisher_sum = []
        for i in range(len(self.layers)):
            fisher_sum.append(alpha_list[self.num_task-1] * self.FISHER[self.num_task-1][i])
            for j in range(self.num_task-1):
                fisher_sum[i] += alpha_list[j] * self.FISHER[j][i]

        for i in range(len(self.layers)):
            sum_mean = 0.0
            sum_var = 0.0
            for j in range(self.num_task):
                sum_mean += self.params_mean[j][i] * (alpha_list[j] * self.FISHER[j][i] + 10**-8/self.num_task) / (fisher_sum[i] + 10**-8)
                sum_var += (self.transform_var(self.params_var[j][i]))**2 * alpha_list[j]**2
            sum_var = self.retransform_var(sum_var)
            ops.append(self.tensor_mean[i].assign(sum_mean))
            ops.append(self.tensor_var[i].assign(sum_var))

        sess.run(ops)

    def bayes_imm_mean(self,sess,alpha=0.5,num_samples=200):
        #sum_idx = max(self.params_mean,key=int)
        sum_idx = self.num_task - 1
        
        alpha_list = [(1-alpha)/(self.num_task-1)] * (self.num_task-1)
        alpha_list.append(alpha)
        ops = []

        for i in range(len(self.layers)):
            sum_mean = 0.0
            sum_var = 0.0
            for sample in range(num_samples):
                mean_samples = []
                for task_idx in range(self.num_task):
                    mean_samples.append(np.random.normal(self.params_mean[task_idx][i],self.transform_var(self.params_var[task_idx][i])))
                for j in range(self.num_task):
                    sum_mean += mean_samples[j] * alpha_list[j]
            sum_mean /= num_samples


            for var_idx in range(self.num_task):
                sum_var += (self.transform_var(self.params_var[j][i]))**2 * alpha_list[j]**2

            sum_var = self.retransform_var(sum_var)
            ops.append(self.tensor_mean[i].assign(sum_mean))
            ops.append(self.tensor_var[i].assign(sum_var))

        sess.run(ops)

    def bayes_imm_mean_kl(self,sess,alpha=0.5,num_samples=200):

        #sum_idx = max(self.params_mean,key=int)
        sum_idx = self.num_task - 1
        
        alpha_list = [(1-alpha)/(self.num_task-1)] * (self.num_task-1)
        alpha_list.append(alpha)
        ops = []

        for i in range(len(self.layers)):
            sum_mean = 0.0
            sum_var = 0.0
            for sample in range(num_samples):
                mean_samples = []
                dist = []
                sum_pdf = 0.0
                for task_idx in range(self.num_task):
                    dist.append(norm(self.params_mean[task_idx][i],self.transform_var(self.params_var[task_idx][i])))
                    sample_data = dist[task_idx].rvs()
                    mean_samples.append(sample_data)
                    sum_pdf += dist[task_idx].pdf(sample_data) * alpha_list[task_idx]
                for j in range(self.num_task):
                    sum_mean += mean_samples[j] * alpha_list[j] * dist[j].pdf(mean_samples[j]) / sum_pdf
            sum_mean = sum_mean / num_samples

            for var_idx in range(self.num_task):
                sum_var += (self.transform_var(self.params_var[j][i]))**2 * alpha_list[j]**2

            sum_var = self.retransform_var(sum_var)
            ops.append(self.tensor_mean[i].assign(sum_mean))
            ops.append(self.tensor_var[i].assign(sum_var))

        sess.run(ops)
        

    def bayes_imm_mode(self,sess,alpha=0.5):
        #sum_idx = max(self.params_mean,key=int)
        sum_idx = self.num_task - 1
        sum_mean = self.params_mean[sum_idx] 
        sum_var = self.params_var[sum_idx]
        alpha_list = [(1-alpha)/(self.num_task-1)] * (self.num_task-1)
        alpha_list.append(alpha)
        ops = []

        fisher_sum = []
        for i in range(len(self.layers)):
            fisher_sum.append(alpha_list[self.num_task-1] * self.FISHER[self.num_task-1][i])
            for j in range(self.num_task-1):
                fisher_sum[i] += alpha_list[j] * self.FISHER[j][i]

        for i in range(len(self.layers)):
            sum_mean = 0.0
            sum_var = 0.0

            for sample in range(num_samples):
                mean_samples = []
                for task_idx in range(self.num_task):
                    mean_samples.append(np.random.normal(self.params_mean[task_idx][i],self.transform_var(self.params_var[task_idx][i])))
                for j in range(self.num_task):
                    sum_mean += mean_samples[j] * (alpha_list[j] * self.FISHER[j][i] + 10**-8/self.num_task) / (fisher_sum[i] + 10**-8)
            sum_mean = sum_mean / num_samples

            for j in range(self.num_task):
                sum_var += (self.transform_var(self.params_var[j][i]))**2 * alpha_list[j]**2
            sum_var = self.retransform_var(sum_var)
            ops.append(self.tensor_mean[i].assign(sum_mean))
            ops.append(self.tensor_var[i].assign(sum_var))

        sess.run(ops)

    def bayes_imm_mode_kl(self,sess,alpha=0.5,num_samples=200):
        #sum_idx = max(self.params_mean,key=int)
        sum_idx = self.num_task - 1
        sum_mean = self.params_mean[sum_idx] 
        sum_var = self.params_var[sum_idx]
        alpha_list = [(1-alpha)/(self.num_task-1)] * (self.num_task-1)
        alpha_list.append(alpha)
        ops = []

        fisher_sum = []
        for i in range(len(self.layers)):
            fisher_sum.append(alpha_list[self.num_task-1] * self.FISHER[self.num_task-1][i])
            for j in range(self.num_task-1):
                fisher_sum[i] += alpha_list[j] * self.FISHER[j][i]

        for i in range(len(self.layers)):
            sum_mean = 0.0
            sum_var = 0.0
            for sample in range(num_samples):
                mean_samples = []
                dist = []
                sum_pdf = 0.0
                for task_idx in range(self.num_task):
                    dist.append(norm(self.params_mean[task_idx][i],self.transform_var(self.params_var[task_idx][i])))
                    sample_data = dist[task_idx].rvs()
                    mean_samples.append(sample_data)
                    sum_pdf += dist[task_idx].pdf(sample_data) * alpha_list[task_idx] * self.FISHER[task_idx][i]
                for j in range(self.num_task):
                    sum_mean += mean_samples[j] * (alpha_list[j] * self.FISHER[j][i] *  dist[j].pdf(mean_samples[j]) + 10**-8/self.num_task) / (sum_pdf + 10**-8)
            sum_mean = sum_mean / num_samples

            for var_idx in range(self.num_task):
                sum_var += (self.transform_var(self.params_var[j][i]))**2 * alpha_list[j]**2

            sum_var = self.retransform_var(sum_var)
            ops.append(self.tensor_mean[i].assign(sum_mean))
            ops.append(self.tensor_var[i].assign(sum_var))

        sess.run(ops)

    def dist_merge(self,sess, alpha=0.5):
        #sum_idx = max(self.params_mean,key=int)
        sum_idx = self.num_task - 1
        sum_mean = self.params_mean[sum_idx] 
        sum_var = self.params_var[sum_idx]
        alpha_list = [(1-alpha)/(self.num_task-1)] * (self.num_task-1)
        alpha_list.append(alpha)
        ops = []

        fisher_sum = []
        for i in range(len(self.layers)):
            var = self.params_var[self.num_task-1][i]
            fisher_sum.append(alpha_list[self.num_task-1] * self.transform_var(var))
            for j in range(self.num_task-1):
                var = self.transform_var(self.params_var[j][i])
                fisher_sum[i] += alpha_list[j] * var

        for i in range(len(self.layers)):
            sum_mean = 0.0
            sum_var = 0.0
            for j in range(self.num_task):
                var = self.transform_var(self.params_var[j][i])
                sum_mean += self.params_mean[j][i] * (alpha_list[j] * var + 10**-8/self.num_task) / (fisher_sum[i] + 10**-8)
                sum_var += (self.transform_var(self.params_var[j][i]))**2 * alpha_list[j]**2
            sum_var = self.retransform_var(sum_var)
            ops.append(self.tensor_mean[i].assign(sum_mean))
            ops.append(self.tensor_var[i].assign(sum_var))

        sess.run(ops)



