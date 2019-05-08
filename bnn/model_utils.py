import tensorflow as tf
import sonnet as snt
import numpy as np
from bnn.utils import mkdir
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
import time

def plot_test_acc(plot_handles):
    plt.legend(handles=plot_handles, loc="center right")
    plt.xlabel("Iterations")
    plt.ylabel("Test Accuracy")
    plt.ylim(0,1)
    display.display(plt.gcf())
    display.clear_output(wait=True)

def make_iterator(dataset,BATCH_SIZE):
    with tf.name_scope('data'):
        data = tf.data.Dataset.from_tensor_slices(dataset)
        data = data.batch(BATCH_SIZE)
        iterator = tf.data.Iterator.from_structure(data.output_types,data.output_shapes)
        img,label = iterator.get_next()
        return img,label,iterator

def make_data_initializer(data,iterator,BATCH_SIZE=128):
    with tf.name_scope('data'):
        data = tf.data.Dataset.from_tensor_slices(data)
        #data = data.shuffle(10000)
        data = data.batch(BATCH_SIZE)
        init = iterator.make_initializer(data)
        return init


def plot_accs(x_axis,y_axis,idx,accs,step,last=False):
    plt.subplot(x_axis,y_axis,idx)
    plt.plot(accs)
    plt.ylim(0,1)
    if not last:
        plt.title('Task {}, acc:{:4f}'.format(idx,accs[-1]))
    else:
        plt.title('Average Accuracy:{:4f}'.format(accs[-1]))
    display.display(plt.gcf())
    display.clear_output(wait=True)
    plt.gcf().set_size_inches(y_axis*5, 3.5)
    plt.tight_layout()



def initialize_model(net,trainset,BATCH_SIZE=128):
    print('Initialization ... ')
    num_batches = len(trainset[0][0]) / BATCH_SIZE
    net.num_batches = num_batches
    X_holder,y_holder,iterator = make_iterator(trainset[0],BATCH_SIZE)
    
    net.set_fisher_graph(X_holder,y_holder)
    net.set_uncertain_prediction()
    out, log_probs, nll, kl_diver= net(X_holder, targets=y_holder, sample=True, n_samples=1, 
                              loss_function=lambda y, y_target: 
                                   tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))
    net.set_vanilla_loss(log_probs,nll,num_batches)
    '''
    _, kl_log_probs, kl_nll, _= net(X_holder, targets=y_holder, sample=True, n_samples=1, 
                              loss_function=lambda y, y_target: 
                                   tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))
    net.set_kl_loss(kl_log_probs,kl_nll,num_batches)
    '''
    _, mode_kl_log_probs, mode_kl_nll, _= net(X_holder, targets=y_holder, sample=True, n_samples=1, drop_out=True,
                              loss_function=lambda y, y_target: 
                                   tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))
    net.set_drop_loss(mode_kl_log_probs,mode_kl_nll,num_batches)
    
    net.summary()
    return iterator



def get_data_init(trainset,testsets,iterator):
    train_init = []
    test_init = []
    for t in range(len(trainset)):
        train_init.append(make_data_initializer(trainset[t],iterator))
    for t in range(len(testsets)):
        test_init.append(make_data_initializer(testsets[t],iterator))
    return train_init,test_init


def load_iterator(net,trainset,testsets):
    iterator = initialize_model(net,trainset)
    train_init,test_init = get_data_init(trainset,testsets,iterator)
    return train_init,test_init



#train_split
'''
params:
    |- sequential: Sequential Bayesian Updating
    |- dist_merge: merge two distribution
    |- imm: mean_imm
    |- imm: mode_imm
    |- st: em_smooth
    |- dp: dirichlet smooth
    |- kl: kl_smooth
    |- _fisher_flag: True - do not compute fisher
    |- terminal_out: print output in terminal
    |

'''
"""

def eval(net,sess,num_task,writer,test_init,test_accs,task_labels,st=False,disp=True,record=True):
    avg_acc_all = 0.0
    for test_idx in range(num_task):
        #test_init = make_data_initializer(testsets[test_idx],iterator)
        sess.run(test_init[test_idx])
        if st:
            net.set_task_params(sess,test_idx)
        avg_acc = 0.0
        for _ in range(10):
            try:
                if writer is not None:
                    acc,summaries,step = sess.run([net.accuracy,net.summary_op,net.gstep])#,feed_dict={x:batch[0],y_:batch[1]})
                else:
                    acc,step = sess.run([net.accuracy,net.gstep])
            except tf.errors.OutOfRangeError:
                pass
                #sess.run(test_init)
                #acc,summaries,step = sess.run([net.accuracy,net.summary_op,net.gstep])#,feed_dict={x:batch[0],y_:batch[1]})
            avg_acc += acc
            if writer is not None:
                writer.add_summary(summaries,global_step = step)
        if record:
            test_accs[test_idx].append(avg_acc / 10)
        avg_acc_all += avg_acc / 10
        if disp:
            plot_accs((num_task+1) // 3 + 1,3,test_idx+1,test_accs[test_idx],step)
    if record:
        test_accs['avg'].append(avg_acc_all / num_task)
    if disp:
        plot_accs((num_task+1) // 3 + 1,3,num_task+1,test_accs['avg'],step,last=True)
    return avg_acc_all / num_task

def em_train(model,sess,num_epoch,disp_freq,trainset,testsets,train_init,test_init,lams=[0],BATCH_SIZE=128,sequential=False,terminal_out=False):
    test_accs = {}
    test_accs['avg'] = []
    for t in range(len(testsets)):
        test_accs[t] = []
    print('Training start ...')
    graph_path = './graph/split/em_smooth'
    mkdir(graph_path)
    writer = tf.summary.FileWriter(graph_path,tf.get_default_graph())
    num_task = len(trainset)
    sess.run(model.lams.assign(lams[0]))
    for idx in range(num_task):
        if idx == 0:
            try:
                model.restore_first_params(sess,clean=True)
                print(' ****** Restoring ... ****** ')
                continue
            except KeyError:
                print('First Training Start ...')
        #Sequential Bayesian Inference
        if idx > 0 and sequential:
            model.set_prior(sess,idx-1)

        #  Training Start
        for e in range(num_epoch):
            sess.run(train_init[idx])
            try:
                while True:
                    _, summaries,step = sess.run([model.train_op,model.summary_op,model.gstep])
                    writer.add_summary(summaries,global_step = step)
            except tf.errors.OutOfRangeError:
                avg_acc = eval(model,sess,num_task,writer,test_init,test_accs,task_labels)

            if terminal_out: 
                print('Training {}th task, Epoch: {}, Accuracy: {:.4f}'.format(idx,e,avg_acc),end='\r')
        model.store_params(idx)
        #  End of Train

    disp = False
    best_component = 0
    best_thresh_hold = 0.0
    best_acc = 0.0
    model.back_up_params()
    #for n_component in range(1,num_task+1):
    for n_component in [2]:
        #for thresh_hold in np.linspace(0.1,0.45,3):
        for thresh_hold in [0.4]:
            #model.restore_params_from_backup()
            model.st_smooth(n_component,dp=False,thresh_hold=thresh_hold)
            avg_acc = eval(model,sess,num_task,None,test_init,test_accs,task_labels,st=True,record=False,disp=False)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_component = n_component
                best_thresh_hold = thresh_hold
    model.st_smooth(best_component,dp,best_thresh_hold)
    eval(model,sess,num_task,None,test_init,test_accs,task_labels,st=True,record=True,disp=True)

    filename = 'em'
    np.savetxt('results/{}_lam={}.csv'.format(filename,lams[0]),[p for p in zip([best_thresh_hold],[best_component],[best_acc])],delimiter=', ', fmt='%.4f')

    plt.savefig('./images/{}_l={}.png'.format(method_name,lams[0]))


def train(model,sess,num_epoch,disp_freq,trainset,testsets,train_init,test_init,lams=[0],BATCH_SIZE=128,sequential=False,dist_merge=False,
                                    imm=False,st=False,kl=False,imm_mode=False,
                                    task_labels=None,dp=False,terminal_out=False,_fisher_flag=True,**method):

    def search_merge_best(method_name):
        best_acc = 0.0
        best_alpha = 0.0
        disp = False
        func = getattr(model,method_name)
        for alpha in tqdm(np.linspace(0,1,20),ascii=True,desc='{} Smooth Process'.format(method_name)):
        #for alpha in [1.0]:
            func(sess,alpha)
            avg_acc = eval(model,sess,num_task,None,test_init,test_accs,task_labels,record=False,disp=False)
            print('alpha :{} Accuracy:{}'.format(alpha,avg_acc))
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_alpha = alpha
        func(sess,best_alpha)
        eval(model,sess,num_task,None,test_init,test_accs,task_labels,record=True,disp=True)
        if not disp:
            print('{} best alpha is:{}, best accuracy is {}'.format(method_name,best_alpha,best_acc))
        np.savetxt('results/'+method_name+'_lam={}.csv'.format(lams[0]),[p for p in zip([best_alpha],[best_acc])],delimiter=', ', fmt='%.4f')


    def search_st_best():
        disp = False
        best_component = 0
        best_thresh_hold = 0.0
        best_acc = 0.0
        model.back_up_params()
        #for n_component in range(1,num_task):
        for n_component in [2]:
            #for thresh_hold in np.linspace(0.1,0.45,3):
            for thresh_hold in [0.4]:
                #model.restore_params_from_backup()
                model.st_smooth(n_component,dp,thresh_hold)
                avg_acc = eval(model,sess,num_task,None,test_init,test_accs,task_labels,st=True,record=False,disp=False)
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_component = n_component
                    best_thresh_hold = thresh_hold
        model.st_smooth(best_component,dp,best_thresh_hold)
        eval(model,sess,num_task,None,test_init,test_accs,task_labels,st=True,record=True,disp=True)
        if dp:
            filename = 'st'
        else:
            filename = 'em'
        np.savetxt('results/{}_lam={}.csv'.format(filename,lams[0]),[p for p in zip([best_thresh_hold],[best_component],[best_acc])],delimiter=', ', fmt='%.4f')


    def search_kl_best():
        disp = False
        def kl_search(alpha):
            sess.run(train_init[0])
            model.restore_last_params(sess)
            model.set_alpha(sess,alpha)
            try:
                for _ in tqdm(range(5000),ascii=True,desc='KL Smooth Process'):
                    sess.run(train_op)
            except tf.errors.OutOfRangeError:
                pass
            avg_acc = eval(model,sess,num_task,None,test_init,test_accs,task_labels,record=False,disp=False)
            return avg_acc


        model.set_learned_dist(sess)
        #op_name = ['kl','mode_kl']
        op_name = ['kl']
        for idx,train_op in enumerate([model.kl_train_op,model.mode_kl_train_op]):

            best_acc = 0.0
            best_alpha = 0.0
            for alpha in tqdm(np.linspace(0,1,20)):
            #for alpha in [0.5]:
                avg_acc = kl_search(alpha)
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_alpha = alpha
            kl_search(best_alpha)
            eval(model,sess,num_task,None,test_init,test_accs,task_labels,record=True,disp=True)
            if not disp:
                print('{} best alpha is:{}, best accuracy is {}'.format(train_op.name,best_alpha,best_acc))
            np.savetxt('results/{}_lam={}.csv'.format(op_name[idx],lams[0]),[p for p in zip([best_alpha],[best_acc])],delimiter=', ', fmt='%.4f')

    _fisher_flag = _fisher_flag
    test_accs = {}
    test_accs['avg'] = []
    for t in range(len(testsets)):
        test_accs[t] = []
    print('Training start ...')
    merge_method = method.pop('merge_method',None)
    method_name = method.pop('method_name','Default')
    graph_path = './graph/split/'
    mkdir(graph_path)
    writer = tf.summary.FileWriter(graph_path,tf.get_default_graph())
    num_task = len(trainset)
    sess.run(model.lams.assign(lams[0]))
    for idx in range(num_task):
        if idx == 0:
            try:
                model.restore_first_params(sess,clean=True)
                print(' ****** Restoring ... ****** ')
                if not _fisher_flag:
                    model.compute_fisher(trainset[idx][0],sess)
                    model.store_fisher(idx)
                continue
            except KeyError:
                print('First Training Start ...')
        #Sequential Bayesian Inference
        if idx > 0 and sequential:
            model.set_prior(sess,idx-1)

        #  Training Start
        for e in range(num_epoch):
            sess.run(train_init[idx])
            try:
                while True:
                    _, summaries,step = sess.run([model.train_op,model.summary_op,model.gstep])
                    writer.add_summary(summaries,global_step = step)
            except tf.errors.OutOfRangeError:
                avg_acc = eval(model,sess,num_task,writer,test_init,test_accs,task_labels)
            
            if terminal_out: 
                print('Training {}th task, Epoch: {}, Accuracy: {:.4f}'.format(idx,e,avg_acc),end='\r')
        model.store_params(idx)
        if (kl or imm_mode) and not _fisher_flag:
            model.compute_fisher(trainset[idx][0],sess)
            model.store_fisher(idx)
            _fisher_flag = True
        #  End of Train

    if merge_method is not None:
        for name in merge_method:
            search_merge_best(name)
    if kl:
        search_kl_best()
    if st:
        search_st_best()
    plt.savefig('./images/{}_l={}.png'.format(method_name,lams[0]))
"""


