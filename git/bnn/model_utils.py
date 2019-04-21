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
    X_holder,y_holder,iterator = make_iterator(trainset[0],BATCH_SIZE)
    net.set_fisher_graph(X_holder,y_holder)
    out, log_probs, nll, kl_diver= net(X_holder, targets=y_holder, sample=True, n_samples=1, 
                              loss_function=lambda y, y_target: 
                                   tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y))
    net.set_vanilla_loss(log_probs,nll,num_batches)
    _, kl_log_probs, kl_nll, _= net(X_holder, targets=y_holder, sample=True, n_samples=1, 
                              loss_function=lambda y, y_target: 
                                   tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y),kl=True)
    net.set_kl_loss(kl_log_probs,kl_nll,num_batches)
    _, mode_kl_log_probs, mode_kl_nll, _= net(X_holder, targets=y_holder, sample=True, n_samples=1, 
                              loss_function=lambda y, y_target: 
                                   tf.nn.softmax_cross_entropy_with_logits(labels=y_target, logits=y),kl=True,mode=True)
    net.set_mode_kl_loss(mode_kl_log_probs,mode_kl_nll,num_batches)
    net.summary()
    return iterator


def eval(net,sess,num_task,writer,test_init,test_accs,set_active_output,task_labels,st=False,disp=True,record=True):
    avg_acc_all = 0.0
    for test_idx in range(num_task):
        if set_active_output is not None:
            set_active_output(task_labels[test_idx])
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



def save_result(name,data):
    header = '{}, accuracy, alpha / '.format(name)


#train_split
def train_permute(model,sess,num_epoch,disp_freq,trainset,testsets,train_init,test_init,lams=[0],BATCH_SIZE=128,sequential=False,dist_merge=False,
                                    imm=False,st=False,set_active_output=None,kl=False,imm_mode=False,
                                    task_labels=None,dp=False,terminal_out=False,_fisher_flag=True,**method):

    def search_merge_best(method_name):
        best_acc = 0.0
        best_alpha = 0.0
        disp = False
        func = getattr(model,method_name)
        for alpha in tqdm(np.linspace(0,1,20),ascii=True,desc='{} Smooth Process'.format(method_name)):
        #for alpha in [1.0]:
            func(sess,alpha)
            avg_acc = eval(model,sess,num_task,None,test_init,test_accs,set_active_output,task_labels,record=False,disp=False)
            print('alpha :{} Accuracy:{}'.format(alpha,avg_acc))
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_alpha = alpha
        func(sess,best_alpha)
        eval(model,sess,num_task,None,test_init,test_accs,set_active_output,task_labels,record=True,disp=True)
        if not disp:
            print('{} best alpha is:{}, best accuracy is {}'.format(method_name,best_alpha,best_acc))
        np.savetxt('results/'+method_name+'_lam={}.csv'.format(lams[0]),[p for p in zip([best_alpha],[best_acc])],delimiter=', ', fmt='%.4f')


    def search_st_best():
        disp = False
        best_component = 0
        best_thresh_hold = 0.0
        best_acc = 0.0
        model.back_up_params()
        for n_component in range(1,num_task):
            #for thresh_hold in np.linspace(0.1,0.45,3):
            for thresh_hold in [0.4]:
                #model.restore_params_from_backup()
                model.st_smooth(n_component,dp,thresh_hold)
                avg_acc = eval(model,sess,num_task,None,test_init,test_accs,set_active_output,task_labels,st=True,record=False,disp=False)
                if avg_acc > best_acc:
                    best_acc = avg_acc
                    best_component = n_component
                    best_thresh_hold = thresh_hold
        model.st_smooth(best_component,dp,best_thresh_hold)
        eval(model,sess,num_task,None,test_init,test_accs,set_active_output,task_labels,st=True,record=True,disp=True)
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
            avg_acc = eval(model,sess,num_task,None,test_init,test_accs,set_active_output,task_labels,record=False,disp=False)
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
            eval(model,sess,num_task,None,test_init,test_accs,set_active_output,task_labels,record=True,disp=True)
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
            if set_active_output is not None:
                set_active_output(task_labels[idx])
            sess.run(train_init[idx])
            try:
                while True:
                    _, summaries,step = sess.run([model.train_op,model.summary_op,model.gstep])
                    writer.add_summary(summaries,global_step = step)
            except tf.errors.OutOfRangeError:
                avg_acc = eval(model,sess,num_task,writer,test_init,test_accs,set_active_output,task_labels)
            
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

def train_permute1(net,sess,num_epoch,disp_freq,trainset,testsets,train_init,test_init,lams=[0],BATCH_SIZE=128,sequential=False,dist_merge=False,
                                    imm=False,stereo=False,set_active_output=None,kl=False,task_labels=None,dp=False):
    #Initialize lam for every test
    #iterator = initialize_model(net,trainset)
    #train_init,test_init = get_data_init(trainset,testsets,iterator)
    for l in range(len(lams)):
        sess.run(tf.global_variables_initializer())
        sess.run(net.lams.assign(lams[l]))
        train_path = './graph/permute/lam={}/train/'.format(l)
        test_path = './graph/permute/lam={}/test/'.format(l)
        mkdir(train_path)
        mkdir(test_path)
        writer = tf.summary.FileWriter(train_path,tf.get_default_graph())
        
        
        test_accs = {}
        test_accs['avg'] = []
        for t in range(len(testsets)):
            test_accs[t] = []
        print('Training Start ... ')

        for idx in range(len(trainset)):
            #train_init = make_data_initializer(trainset[idx],iterator)

            net.store()
            if idx > 0 and sequential:
                print('Setting Prior Knowledge for the next Training ...')
                net.set_prior(sess)

            for e in range(num_epoch):
                if task_labels is not None:
                    set_active_output(task_labels[idx])
                sess.run(train_init[idx])
                try:
                    while True:
                        _, summaries,step = sess.run([net.train_op,net.summary_op,net.gstep])#,feed_dict={x:batch[0],y_:batch[1]})
                        writer.add_summary(summaries,global_step = step)
                except tf.errors.OutOfRangeError:
                    eval(net,sess,testsets,writer,test_init,test_accs,set_active_output,task_labels)
            if stereo or kl or dist_merge:
                net.stroe_gauss(idx)
            if idx>0 and dist_merge:
                best_dist_acc = 0.0
                best_dist_alpha = 0.0
                for alpha in tqdm(np.linspace(0,1,20)):
                    net.dist_merge(sess,alpha)
                    _avg_acc = eval(net,sess,testsets,writer,test_init,test_accs,set_active_output,task_labels)
                    if _avg_acc > best_dist_acc:
                        best_dist_acc = _avg_acc
                        best_dist_alpha = alpha
                net.dist_merge(sess,best_dist_alpha)
                print(best_dist_alpha)
                time.sleep(10)

            if imm or stereo:
                net.store_merge_params(idx)

            if kl:
                print('Computing Fisher ...')
                net.compute_fisher(trainset[idx][0],sess)
                if idx == len(trainset) - 1:
                    net.set_learned_dist(sess)
                    net.set_prior(sess)
                    
                    for e in range(1):
                        print('KL Merging Epoch %d ...'%e)
                        sess.run(train_init[idx])
                        S = 0
                        try:
                            while S<5000:
                                print('Step %d ...'%S,end='\r')
                                _ = sess.run(net.kl_train_op)
                                S += 1
                        except tf.errors.OutOfRangeError:
                            eval(net,sess,testsets,None,test_init,test_accs,set_active_output,task_labels)
                else:
                    net.store_fisher()
        if imm:
            best_imm_acc = 0.0
            best_imm_alpha = 0.0
            for alpha in tqdm(np.linspace(0,1,20)):
                net.imm_mean(sess,alpha)
                _avg_acc = eval(net,sess,testsets,writer,test_init,test_accs,set_active_output,task_labels)
                if _avg_acc > best_imm_acc:
                    best_imm_acc = _avg_acc
                    best_imm_alpha = alpha
            net.imm_mean(sess,best_imm_alpha)
            print(best_imm_alpha)
            time.sleep(10)
        if stereo:
            net.store_gauss()
            best_stereo_n = 0.0
            best_stereo_acc = 0.0
            if dp:
                #for n in np.linspace(0.001,1 / len(trainset),20):
                for n in [0.0]:
                    net.restore_gauss()
                    net.em_stereo(n_component=len(trainset),dp=dp,thresh_hold=n)
                    stereo_acc = eval(net,sess,testsets,writer,test_init,test_accs,set_active_output,task_labels,stereo=True)
                    if stereo_acc > best_stereo_acc:
                        best_stereo_acc = stereo_acc
                        best_stereo_n = n
            #print(best_stereo_n)
            #time.sleep(10)
            else:
                #for n in range(1,len(trainset)+1):
                for n in [2]:
                    print('component is set to %d'%n)
                    time.sleep(10)
                    net.restore_gauss()
                    net.em_stereo(n_component=n,dp=dp)
                    stereo_acc = eval(net,sess,testsets,writer,test_init,test_accs,set_active_output,task_labels,stereo=True)
                    if stereo_acc > best_stereo_acc:
                        best_stereo_acc = stereo_acc
                        best_stereo_n = n
            print(best_stereo_n)
            time.sleep(10)
            eval(net,sess,testsets,writer,test_init,test_accs,set_active_output,task_labels,stereo=True)
        else:
            eval(net,sess,testsets,None,test_init,test_accs,set_active_output,task_labels)
        writer.close()





