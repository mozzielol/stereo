import tensorflow as tf
import sonnet as snt
import numpy as np
from bnn.utils import mkdir
import matplotlib.pyplot as plt
from IPython import display
from tqdm import tqdm
import time
from bnn.model_utils import plot_accs

def reduce_entropy(X, axis=-1):
    """
    calculate the entropy over axis and reduce that axis
    :param X:
    :param axis:
    :return:
    """
    return -1 * np.sum(X * np.log(X+1E-12), axis=axis)

def calc_risk(preds, labels=None):
    """
    Calculates the parameters we can possibly use to examine risk of a neural net
    :param preds: preds in shape [num_runs, num_batch, num_classes]
    :param labels:
    :return:
    """
    #if isinstance(preds, list):
    #    preds = np.stack(preds)
    # preds in shape [num_runs, num_batch, num_classes]
    num_runs, num_batch = preds.shape[:2]

    ave_preds = np.mean(preds, axis=0)
    pred_class = np.argmax(ave_preds, axis=1)

    # entropy of the posterior predictive
    entropy = reduce_entropy(ave_preds, axis=1)

    # Expected entropy of the predictive under the parameter posterior
    entropy_exp = np.mean(reduce_entropy(preds, axis=2), axis=0)
    mutual_info = entropy - entropy_exp  # Equation 2 of https://arxiv.org/pdf/1711.08244.pdf

    # Average and variance of softmax for the predicted class
    variance = np.std(preds[:, range(num_batch), pred_class], 0)
    ave_softmax = np.mean(preds[:, range(num_batch), pred_class], 0)

    # And calculate accuracy if we know the labels
    if labels is not None:
        correct = np.equal(pred_class, labels)
    else:
        correct = None
    return entropy, mutual_info, variance, ave_softmax, correct

def em_predict(predictions,y):
    avg_pred = np.mean(predictions,axis=0)
    risk = calc_risk(predictions)
    acc = np.mean(np.equal(np.argmax(y,1), np.argmax(avg_pred, axis=-1)))
    return acc,risk[2]

def eval(net,sess,num_task,writer,test_init,test_accs,params_idx=None,disp=True,record=True):
    avg_acc_all = 0.0
    for test_idx in range(num_task):
        sess.run(test_init[test_idx])
        if params_idx is not None:
            #improving
            net.set_task_params(sess,params_idx[test_idx])
        avg_acc = 0.0
        for _ in range(10):
        #while True:
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


def em_eval(net,sess,num_task,writer,testsets,test_accs,disp=True,record=True,num_runs=200):
    def make_prediction(data,label):
        predictions = []
        total_acc = 0.0
        for _ in range(num_runs):
            pred, em_acc = sess.run([net.predictions,net.em_accuracy],feed_dict={net.x_placeholder:data,net.y_placeholder:label})
            predictions.append(pred)
            total_acc += em_acc
        return np.array(predictions),total_acc/num_runs

    avg_acc_all = 0.0
    params_idx_list = []
    for test_idx in range(num_task):
        avg_acc = []
        avg_uncertainty = []
        for params_idx in net.params_mean.keys():
            net.set_task_params(sess,params_idx)
            avg_acc.append(0.0)
            avg_uncertainty.append(0.0)

            for iters in range(1):
                test_data = testsets[test_idx][0][iters*128:(iters+1)*128]
                test_label = testsets[test_idx][1][iters*128:(iters+1)*128]
            #while True:
                try:
                    #predictions = np.stack(make_prediction(),axis=0)
                    predictions,acc = make_prediction(test_data,test_label)
                    step = sess.run(net.gstep)#,feed_dict={x:batch[0],y_:batch[1]})
                    acc,uncertainty = em_predict(predictions,test_label)
                except tf.errors.OutOfRangeError:
                    pass
                    #sess.run(test_init)
                    #acc,summaries,step = sess.run([net.accuracy,net.summary_op,net.gstep])#,feed_dict={x:batch[0],y_:batch[1]})
                avg_acc[params_idx] += acc/10
                avg_uncertainty[params_idx] += uncertainty
        params_idx_list.append(np.argmin(np.mean(avg_uncertainty,axis=1)))
    return params_idx_list,np.array(avg_uncertainty)*200
    """
        print(np.mean(avg_uncertainty,axis=1))
        print(avg_acc)
        #print(avg_uncertainty)
        avg_acc = avg_acc[np.argmin(np.mean(avg_uncertainty,axis=1))]
        print(avg_acc)
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
    """


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
def em_train(model,sess,num_epoch,disp_freq,trainset,testsets,train_init,test_init,lams=[0],num_component=2,num_thresh_hold=0.4,drop_out=False,
    BATCH_SIZE=128,sequential=False,terminal_out=False):
    model.initialize_default_params(sess)
    dp = False
    test_accs = {}
    test_accs['avg'] = []
    for t in range(len(testsets)):
        test_accs[t] = []
    print('Training start ...')
    graph_path = './graph/split/em_smooth'
    mkdir(graph_path)
    #writer = tf.summary.FileWriter(graph_path,tf.get_default_graph())
    writer = None
    num_task = len(trainset)
    sess.run(model.lams.assign(lams[0]))
    for idx in range(num_task):
        '''
        if idx == 0:
            try:
                model.restore_first_params(sess,clean=True)
                print(' ****** Restoring ... ****** ')
                continue
            except KeyError:
                print('First Training Start ...')
        '''
        #Sequential Bayesian Inference
        if idx > 0 and sequential:
            model.set_prior(sess,idx-1)

        #  Training Start
        for e in range(num_epoch):
            #print('Training Epoch {}/{} ...'.format(e,num_epoch))
            sess.run(train_init[idx])
            try:
                while True:
                    #_, summaries,step = sess.run([model.train_op,model.summary_op,model.gstep])
                    _,step = sess.run([model.apply_dropout(drop_out,idx),model.gstep])
                    #writer.add_summary(summaries,global_step = step)
            except tf.errors.OutOfRangeError:
                #print('End Epoch {}/{} ...'.format(e,num_epoch))
                avg_acc = eval(model,sess,num_task,writer,test_init,test_accs)

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
    for n_component in [num_component]:
        #for thresh_hold in np.linspace(0.1,0.45,3):
        for thresh_hold in [num_thresh_hold]:
            #model.restore_params_from_backup()
            model.st_smooth(n_component,dp=dp,thresh_hold=thresh_hold)
            #param_idx = eval(model,sess,num_task,None,testsets,test_accs,record=False,disp=False)
            avg_acc = eval(model,sess,num_task,None,test_init,test_accs,record=False,disp=False)
            if avg_acc > best_acc:
                best_acc = avg_acc
                best_component = n_component
                best_thresh_hold = thresh_hold
    #model.st_smooth(best_component,dp,best_thresh_hold)
    param_idx,avg_uncertainty = em_eval(model,sess,num_task,None,testsets,test_accs,record=False,disp=False)
    eval(model,sess,num_task,None,test_init,test_accs,params_idx=param_idx,record=True,disp=True)

    filename = 'em'
    np.savetxt('results/{}_lam={}.csv'.format(filename,lams[0]),[p for p in zip([best_thresh_hold],[best_component],[best_acc])],delimiter=', ', fmt='%.4f')

    plt.savefig('./images/{}_l={}.png'.format(filename,lams[0]))


def imm_train(model,sess,num_epoch,disp_freq,trainset,testsets,train_init,test_init,lams=[0],BATCH_SIZE=128,sequential=False,drop_out=False,
                                    imm=False,imm_mode=False,terminal_out=False):
    if (imm and imm_mode) or (not imm and not imm_mode):
        raise ValueError('only imm or imm_mode')
    model.initialize_default_params(sess)
    test_accs = {}
    test_accs['avg'] = []
    for t in range(len(testsets)):
        test_accs[t] = []
    print('Training start ...')
    graph_path = './graph/split/imm'
    mkdir(graph_path)
    #writer = tf.summary.FileWriter(graph_path,tf.get_default_graph())
    writer = None
    num_task = len(trainset)
    sess.run(model.lams.assign(lams[0]))
    for idx in range(num_task):
        '''
        if idx == 0:
            try:
                #model.restore_first_params(sess,clean=True)
                print(' ****** Restoring ... ****** ')
                if imm_mode:
                    model.compute_fisher(trainset[idx][0],sess)
                    model.store_fisher(idx)
                continue
            except KeyError:
                print('First Training Start ...')
        '''
        #Sequential Bayesian Inference
        if idx > 0 and sequential:
            model.set_prior(sess,idx-1)

        #  Training Start
        for e in range(num_epoch):
            sess.run(train_init[idx])
            try:
                while True:
                    #_, summaries,step = sess.run([model.train_op,model.summary_op,model.gstep])
                    _,step = sess.run([model.apply_dropout(drop_out,idx),model.gstep])
                    #writer.add_summary(summaries,global_step = step)
            except tf.errors.OutOfRangeError:
                avg_acc = eval(model,sess,num_task,writer,test_init,test_accs)
            
            if terminal_out: 
                print('Training {}th task, Epoch: {}, Accuracy: {:.4f}'.format(idx,e,avg_acc),end='\r')
        model.store_params(idx)
        if imm_mode:
            model.compute_fisher(trainset[idx][0],sess)
            model.store_fisher(idx)

    if imm:
        method_name = 'imm_mean'
    elif imm_mode:
        method_name = 'imm_mode'

    best_acc = 0.0
    best_alpha = 0.0
    disp = False
    func = getattr(model,method_name)
    for alpha in tqdm(np.linspace(0,1,20),ascii=True,desc='{} Smooth Process'.format(method_name)):
    #for alpha in [1.0]:
        func(sess,alpha)
        avg_acc = eval(model,sess,num_task,None,test_init,test_accs,record=False,disp=False)
        print('alpha :{} Accuracy:{}'.format(alpha,avg_acc))
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_alpha = alpha
    func(sess,best_alpha)
    eval(model,sess,num_task,None,test_init,test_accs,record=True,disp=True)
    if not disp:
        print('{} best alpha is:{}, best accuracy is {}'.format(method_name,best_alpha,best_acc))
    np.savetxt('results/'+method_name+'_lam={}.csv'.format(lams[0]),[p for p in zip([best_alpha],[best_acc])],delimiter=', ', fmt='%.4f')

    plt.savefig('./images/{}_l={}.png'.format(method_name,lams[0]))

def bayes_imm_train(model,sess,num_epoch,disp_freq,trainset,testsets,train_init,test_init,lams=[0],BATCH_SIZE=128,sequential=False,drop_out=False,
                                    imm=False,imm_mode=False,terminal_out=False):
    model.initialize_default_params(sess)
    if imm and imm_mode:
        raise ValueError('only imm or imm_mode')
    test_accs = {}
    test_accs['avg'] = []
    for t in range(len(testsets)):
        test_accs[t] = []
    print('Training start ...')
    graph_path = './graph/split/imm'
    mkdir(graph_path)
    #writer = tf.summary.FileWriter(graph_path,tf.get_default_graph())
    writer = None
    num_task = len(trainset)
    sess.run(model.lams.assign(lams[0]))
    for idx in range(num_task):
        '''
        if idx == 0:
            try:
                model.restore_first_params(sess,clean=True)
                print(' ****** Restoring ... ****** ')
                if imm_mode:
                    model.compute_fisher(trainset[idx][0],sess)
                    model.store_fisher(idx)
                continue
            except KeyError:
                print('First Training Start ...')
        '''
        #Sequential Bayesian Inference
        if idx > 0 and sequential:
            model.set_prior(sess,idx-1)

        #  Training Start
        for e in range(num_epoch):
            sess.run(train_init[idx])
            try:
                while True:
                    _, summaries,step = sess.run([model.apply_dropout(drop_out,idx),model.summary_op,model.gstep])
                    #writer.add_summary(summaries,global_step = step)
            except tf.errors.OutOfRangeError:
                avg_acc = eval(model,sess,num_task,writer,test_init,test_accs)
            
            if terminal_out: 
                print('Training {}th task, Epoch: {}, Accuracy: {:.4f}'.format(idx,e,avg_acc),end='\r')
        model.store_params(idx)
        if imm_mode:
            model.compute_fisher(trainset[idx][0],sess)
            model.store_fisher(idx)

    if imm:
        method_name = 'bayes_imm_mean'
    elif imm_mode:
        method_name = 'bayes_imm_mode'

    best_acc = 0.0
    best_alpha = 0.0
    disp = False
    func = getattr(model,method_name)
    model.back_up_params()
    for alpha in tqdm(np.linspace(0,1,20),ascii=True,desc='{} Smooth Process'.format(method_name)):
    #for alpha in [1.0]:
        model.restore_params_from_backup()
        func(sess,alpha)
        avg_acc = eval(model,sess,num_task,None,test_init,test_accs,record=False,disp=False)
        print('alpha :{} Accuracy:{}'.format(alpha,avg_acc))
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_alpha = alpha
    func(sess,best_alpha)
    eval(model,sess,num_task,None,test_init,test_accs,record=True,disp=True)
    if not disp:
        print('{} best alpha is:{}, best accuracy is {}'.format(method_name,best_alpha,best_acc))
    np.savetxt('results/'+method_name+'_lam={}.csv'.format(lams[0]),[p for p in zip([best_alpha],[best_acc])],delimiter=', ', fmt='%.4f')

    plt.savefig('./images/{}_l={}.png'.format(method_name,lams[0]))

def bayes_imm_kl_train(model,sess,num_epoch,disp_freq,trainset,testsets,train_init,test_init,lams=[0],BATCH_SIZE=128,sequential=False,drop_out=False,
                                    imm=False,imm_mode=False,terminal_out=False):
    model.initialize_default_params(sess)
    if imm and imm_mode:
        raise ValueError('only imm or imm_mode')
    test_accs = {}
    test_accs['avg'] = []
    for t in range(len(testsets)):
        test_accs[t] = []
    print('Training start ...')
    graph_path = './graph/split/imm'
    mkdir(graph_path)
    #writer = tf.summary.FileWriter(graph_path,tf.get_default_graph())
    writer = None
    num_task = len(trainset)
    sess.run(model.lams.assign(lams[0]))
    for idx in range(num_task):
        '''
        if idx == 0:
            try:
                model.restore_first_params(sess,clean=True)
                print(' ****** Restoring ... ****** ')
                if imm_mode:
                    model.compute_fisher(trainset[idx][0],sess)
                    model.store_fisher(idx)
                continue
            except KeyError:
                print('First Training Start ...')
        '''
        #Sequential Bayesian Inference
        if idx > 0 and sequential:
            model.set_prior(sess,idx-1)

        #  Training Start
        for e in range(num_epoch):
            sess.run(train_init[idx])
            try:
                while True:
                    _, summaries,step = sess.run([model.apply_dropout(drop_out,idx),model.summary_op,model.gstep])
                    #writer.add_summary(summaries,global_step = step)
            except tf.errors.OutOfRangeError:
                avg_acc = eval(model,sess,num_task,writer,test_init,test_accs)
            
            if terminal_out: 
                print('Training {}th task, Epoch: {}, Accuracy: {:.4f}'.format(idx,e,avg_acc),end='\r')
        model.store_params(idx)
        if imm_mode:
            model.compute_fisher(trainset[idx][0],sess)
            model.store_fisher(idx)

    if imm:
        method_name = 'bayes_imm_mean_kl'
    elif imm_mode:
        method_name = 'bayes_imm_mode_kl'

    best_acc = 0.0
    best_alpha = 0.0
    disp = False
    func = getattr(model,method_name)
    model.back_up_params()
    for alpha in tqdm(np.linspace(0,1,20),ascii=True,desc='{} Smooth Process'.format(method_name)):
    #for alpha in [1.0]:
        model.restore_params_from_backup()
        func(sess,alpha)
        avg_acc = eval(model,sess,num_task,None,test_init,test_accs,record=False,disp=False)
        print('alpha :{} Accuracy:{}'.format(alpha,avg_acc))
        if avg_acc > best_acc:
            best_acc = avg_acc
            best_alpha = alpha
    func(sess,best_alpha)
    eval(model,sess,num_task,None,test_init,test_accs,record=True,disp=True)
    if not disp:
        print('{} best alpha is:{}, best accuracy is {}'.format(method_name,best_alpha,best_acc))
    np.savetxt('results/'+method_name+'_lam={}.csv'.format(lams[0]),[p for p in zip([best_alpha],[best_acc])],delimiter=', ', fmt='%.4f')

    plt.savefig('./images/{}_l={}.png'.format(method_name,lams[0]))

def common_train(model,sess,num_epoch,disp_freq,trainset,testsets,train_init,test_init,lams=[0],BATCH_SIZE=128,sequential=False,terminal_out=False,
    drop_out=False):
    model.initialize_default_params(sess)
    test_accs = {}
    test_accs['avg'] = []
    for t in range(len(testsets)):
        test_accs[t] = []
    print('Training start ...')
    graph_path = './graph/split/common'
    mkdir(graph_path)
    #writer = tf.summary.FileWriter(graph_path,tf.get_default_graph())
    writer = None
    num_task = len(trainset)
    sess.run(model.lams.assign(lams[0]))
    for idx in range(num_task):
        '''
        if idx == 0:
            try:
                model.restore_first_params(sess,clean=True)
                print(' ****** Restoring ... ****** ')
                continue
            except KeyError:
                print('First Training Start ...')
        '''
        #Sequential Bayesian Inference
        if idx > 0 and sequential:
            model.set_prior(sess,idx-1)

        #  Training Start
        for e in range(num_epoch):
            sess.run(train_init[idx])
            try:
                while True:
                    _, summaries,step = sess.run([model.apply_dropout(drop_out,idx),model.summary_op,model.gstep])
                    #writer.add_summary(summaries,global_step = step)
            except tf.errors.OutOfRangeError:
                avg_acc = eval(model,sess,num_task,writer,test_init,test_accs)
            
            if terminal_out: 
                print('Training {}th task, Epoch: {}, Accuracy: {:.4f}'.format(idx,e,avg_acc),end='\r')
        model.store_params(idx)


    #np.savetxt('results/'+'common'+'_lam={}.csv'.format(lams[0]),[p for p in zip([best_alpha],[best_acc])],delimiter=', ', fmt='%.4f')

    plt.savefig('./images/{}_l={}.png'.format('common',lams[0]))

