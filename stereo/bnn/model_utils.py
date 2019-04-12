import tensorflow as tf
import sonnet as snt
import numpy as np
from bnn.utils import mkdir
import matplotlib.pyplot as plt
from IPython import display

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
    data = tf.data.Dataset.from_tensor_slices(data)
    data = data.shuffle(10000)
    data = data.batch(BATCH_SIZE)
    init = iterator.make_initializer(data)
    return init


def plot_accs(x_axis,y_axis,idx,accs,step,disp_freq,last=False):
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
    net.summary()
    return iterator


def train_permute(net,sess,num_epoch,disp_freq,trainset,testsets,x,y_,lams=[0],BATCH_SIZE=128,sequential=False):
    #Initialize lam for every test
    iterator = initialize_model(net,trainset)
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
            train_init = make_data_initializer(trainset[idx],iterator)
            for e in range(num_epoch):
                sess.run(train_init)
                try:
                    while True:
                        #batch = sess.run([X_train,y_train])
                        _, summaries,step = sess.run([net.train_op,net.summary_op,net.gstep])#,feed_dict={x:batch[0],y_:batch[1]})
                        writer.add_summary(summaries,global_step = step)
                
                    #if step % disp_freq == 0:
                except tf.errors.OutOfRangeError:
                        avg_acc_all = 0.0
                        for test_idx in range(len(testsets)):
                            #plt.subplot(l+1,len(testsets)+1,test_idx+1)
                            test_init = make_data_initializer(testsets[test_idx],iterator)
                            sess.run(test_init)
                            avg_acc = 0.0
                            for _ in range(10):
                                #batch = sess.run([X_test,y_test])
                                acc,summaries,step = sess.run([net.accuracy,net.summary_op,net.gstep])#,feed_dict={x:batch[0],y_:batch[1]})
                                avg_acc += acc
                                writer.add_summary(summaries,global_step = step)
                            test_accs[test_idx].append(avg_acc / 10)
                            avg_acc_all += avg_acc / 10
                            plot_accs((len(testsets)+1) // 3 + 1,3,test_idx+1,test_accs[test_idx],step,disp_freq)
                            #plt.plot(range(1,step+2,disp_freq),test_accs[test_idx])
                            #plt.ylim(0,1)
                            #plt.title('Task %d'%test_idx)
                            #display.display(plt.gcf())
                            #display.clear_output(wait=True)
                            #plt.gcf().set_size_inches(len(testsets)*5, 3.5)
                        test_accs['avg'].append(avg_acc_all / len(testsets))
                        plot_accs((len(testsets)+1) // 3 + 1,3,len(testsets)+1,test_accs['avg'],step,disp_freq,last=True)
                #except tf.errors.OutOfRangeError:
                 #   pass
            if sequential:
                net.store()
                net.set_prior(sess)





