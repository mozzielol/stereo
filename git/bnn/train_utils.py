import tensorflow as tf
import sonnet as snt
import numpy as np
from bnn.utils import *
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
import time
from bnn.BNN_MLP import BNN_MLP as Model
from bnn.model_utils import *

class Train_split(object):
    """docstring for Train"""
    def __init__(self, sess,num_task,task_labels=[[0,1,2,3,4], [5,6,7,8,9]],num_epoch=5):
        super(Train_split, self).__init__()
        self.num_epoch = num_epoch
        self.BATCH_SIZE = 128
        self.sess = sess
        self.num_task = num_task
        self.init_split_mnist(task_labels)
        graph_path = './graph/split/train/'
        self.init_writer_and_acc(graph_path)



    def init_mask(self):
        self.output_dim = 10
        self.output_mask = tf.Variable(tf.zeros(10), name="mask", trainable=False)


    def set_active_outputs(self,labels):
        new_mask = np.zeros(self.output_dim)
        for l in labels:
            new_mask[l] = 1.0
        self.sess.run(self.output_mask.assign(new_mask))


    def masked_softmax(self,logits):
        # logits are [batch_size, output_dim]
        x = tf.where(tf.tile(tf.equal(self.output_mask[None, :], 1.0), [tf.shape(logits)[0], 1]), logits, -1e32 * tf.ones_like(logits))
        return tf.nn.softmax(x)


    def init_model(self,clean=False):

        if not hasattr(self,'model') or clean:
            self.init_mask()
            self.model = Model(n_inputs=784,n_outputs=10,hidden_units=[100,100],num_task = self.num_task,last_activation=self.masked_softmax)
            self.model.set_fisher_graph(self.img,self.label)
            out, log_probs, nll, kl_diver= self.model(self.img, targets=self.label, sample=True, n_samples=1, 
                                      loss_function=lambda y, y_target: 
                                           tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))
            self.model.set_vanilla_loss(log_probs,nll,self.num_batches)
            '''
            _, kl_log_probs, kl_nll, _= self.model(self.img, targets=self.label, sample=True, n_samples=1, 
                                      loss_function=lambda y, y_target: 
                                           tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_target, logits=y),kl=True)
            self.model.set_kl_loss(kl_log_probs,kl_nll,self.num_batches)
            _, mode_kl_log_probs, mode_kl_nll, _= self.model(self.img, targets=self.label, sample=True, n_samples=1, 
                                      loss_function=lambda y, y_target: 
                                           tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_target, logits=y),kl=True,mode=True)
            self.model.set_mode_kl_loss(mode_kl_log_probs,mode_kl_nll,self.num_batches)
            '''
            self.model.summary()

            self.sess.run(tf.global_variables_initializer())

    def make_iterator(self):
        def make_data_initializer(data,iterator):
            with tf.name_scope('data'):
                data = tf.data.Dataset.from_tensor_slices(data)
                #data = data.shuffle(10000)
                data = data.batch(self.BATCH_SIZE)
                init = iterator.make_initializer(data)
                return init
        with tf.name_scope('data'):
            data = tf.data.Dataset.from_tensor_slices(self.trainset[0])
            data = data.batch(self.BATCH_SIZE)
            iterator = tf.data.Iterator.from_structure(data.output_types,data.output_shapes)
            self.img,self.label = iterator.get_next()
            self.train_init = []
            self.test_init = []
            for t in range(len(self.trainset)):
                self.train_init.append(make_data_initializer(self.trainset[t],iterator))
            for t in range(len(self.testsets)):
                self.test_init.append(make_data_initializer(self.testsets[t],iterator))

    def init_split_mnist(self,task_labels):
        
        self.task_labels = task_labels
        self.trainset = construct_split_mnist(self.task_labels)
        self.testsets = construct_split_mnist(self.task_labels,split='test')
        self.num_batches = len(self.trainset[0][0]) / self.BATCH_SIZE
        self.make_iterator()
        self.init_model()


    def init_writer_and_acc(self,graph_path=None):
        if graph_path is not None:
            mkdir(graph_path)
            self.writer = tf.summary.FileWriter(graph_path,tf.get_default_graph())
        else:
            self.writer = None
        self.test_accs = {}
        self.test_accs['avg'] = []
        for t in range(self.num_task):
            self.test_accs[t] = []

    def eval(self,writer=None,disp=True,record=True,st=False):
        avg_acc_all = 0.0
        for test_idx in range(self.num_task):
            self.set_active_outputs(self.task_labels[test_idx])
            #test_init = make_data_initializer(testsets[test_idx],iterator)
            self.sess.run(self.test_init[test_idx])
            if st:
                self.model.set_task_params(self.sess,test_idx)

            avg_acc = 0.0
            for _ in range(10):
                try:
                    if writer is not None:
                        acc,summaries,step = self.sess.run([self.model.accuracy,self.model.summary_op,self.model.gstep])#,feed_dict={x:batch[0],y_:batch[1]})
                    else:
                        acc,step = sess.run([self.model.accuracy,self.model.gstep])
                except tf.errors.OutOfRangeError:
                    pass
                    #sess.run(test_init)
                    #acc,summaries,step = sess.run([self.model.accuracy,self.model.summary_op,self.model.gstep])#,feed_dict={x:batch[0],y_:batch[1]})
                avg_acc += acc
                if writer is not None:
                    writer.add_summary(summaries,global_step = step)
            if record:
                self.test_accs[test_idx].append(avg_acc / 10)
            avg_acc_all += avg_acc / 10
            if disp:
                plot_accs((self.num_task+1) // 3 + 1,3,test_idx+1,self.test_accs[test_idx],step)
        if record:
            self.test_accs['avg'].append(avg_acc_all / self.num_task)
        if disp:
            plot_accs((self.num_task+1) // 3 + 1,3,self.num_task+1,self.test_accs['avg'],step)
        return avg_acc_all / self.num_task

    def train(self,name,disp=True,**method):
        print('Training start ...')
        sequential = method.pop('sequential',False)
        dist_merge = method.pop('dist_merge',False)
        imm_mean = method.pop('imm_mean',False)
        imm_mode = method.pop('imm_mode',False)
        st = method.pop('st',False)
        kl = method.pop('kl',False)
        dp = method.pop('dp',False)
        merge_method = method.pop('merge_method',None)
        mkdir('./images/')
        self.image_path = './images/'+name
        
        for idx in range(self.num_task):
            if idx == 0:
                try:
                    self.model.restore_first_params(self.sess,clean=True)
                except KeyError:
                    print('First Training Start ...')
            #Sequential Bayesian Inference
            if idx > 0 and sequential:
                self.model.set_prior(self.sess)

            #  Training Start
            for e in range(self.num_epoch):
                self.set_active_outputs(self.task_labels[idx])
                self.sess.run(self.train_init[idx])
                try:
                    while True:
                        _, summaries,step = self.sess.run([self.model.train_op,self.model.summary_op,self.model.gstep])
                        self.writer.add_summary(summaries,global_step = step)
                except tf.errors.OutOfRangeError:
                    avg_acc = self.eval(self.writer,disp=disp)
                    
                print('Training {}th task, Epoch: {}, Accuracy: {:.4f}'.format(idx,e,avg_acc),end='\r')
            self.model.store_params(idx)
            if kl or imm_mode:
                self.model.compute_fisher(trainset[idx][0],self.sess)
                self.model.store_fisher(idx)
            #  End of Train

        if merge_method is not None:
            for name in merge_method:
                self.search_merge_best(name,disp)
        if kl:
            self.search_kl_best(disp)
        if st:
            self.search_st_best(disp,dp)

    def search_st_best(self,disp,dp):
        best_component = 0
        best_thresh_hold = 0.0
        best_acc = 0.0
        self.model.back_up_params()
        for n_component in range(1,self.num_task):
            for thresh_hold in range(0.0,0.5,10):
                self.model.restore_params_from_backup()
                self.model.st_smooth(n_component,dp,thresh_hold)
                avg_acc = self.eval(disp=False,record=False)
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_component = n_component
                    best_thresh_hold = thresh_hold
        self.model.st_smooth(best_component,dp,best_thresh_hold)
        self.eval(self.writer,disp=disp)


    def search_kl_best(self,disp):

        def kl_search(alpha):
            self.sess.run(self.train_init[0])
            self.model.restore_last_params(self.sess)
            self.model.set_alpha(self.sess,alpha)
            try:
                for _ in tqdm(range(20000),ascii=True,desc='KL Smooth Process'):
                    self.sess.run(train_op)
            except tf.errors.OutOfRangeError:
                pass
            avg_acc = self.eval(disp=False,record=False)
            return avg_acc


        self.model.set_learned_dist(self.sess)
        
        for train_op in [self.model.kl_train_op,self.model.mode_kl_train_op]:
            best_acc = 0.0
            best_alpha = 0.0
            for alpha in tqdm(np.linspace(0,1,20)):
                avg_acc = kl_search(alpha)
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_alpha = alpha
            kl_search(best_alpha)
            self.eval(self.writer,disp=disp)
            if not disp:
                print('{} best alpha is:{}, best accuracy is {}'.format(train_op.name,best_alpha,best_acc))





    def search_merge_best(self,method_name,disp):
        best_acc = 0.0
        best_alpha = 0.0
        func = getattr(self.model,method_name)
        for alpha in tqdm(np.linspace(0,1,20),ascii=True,desc='{} Smooth Process'.format(method_name)):
            func(self.sess,alpha)
            avg_acc = self.eval(disp=False,record=False)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_alpha = alpha
        fuc(self.sess,best_alpha)
        self.eval(self.writer,disp=disp)
        if not disp:
            print('{} best alpha is:{}, best accuracy is {}'.format(method_name,best_alpha,best_acc))

            


