import tensorflow as tf
import sonnet as snt

import numpy as np

import matplotlib.pyplot as plt
from cycler import cycler

def permute(task, seed):
    np.random.seed(seed)
    perm = np.random.permutation(task.train._images.shape[1])
    permuted = deepcopy(task)
    permuted.train._images = permuted.train._images[:, perm]
    permuted.test._images = permuted.test._images[:, perm]
    permuted.validation._images = permuted.validation._images[:, perm]
    return permuted

def make_tf_data_batch(x, y, shuffle=True):
    # create Dataset objects using the data previously downloaded
    dataset_train = tf.data.Dataset.from_tensor_slices((x, y.astype(np.int32)))

    if shuffle:
        dataset_train = dataset_train.shuffle(100000)

    # we shuffle the data and sample repeatedly batches for training
    batched_dataset_train = dataset_train.repeat().batch(BATCH_SIZE)
    # create iterator to retrieve batches
    iterator_train = batched_dataset_train.make_one_shot_iterator()
    # get a training batch of images and labels
    (batch_train_images, batch_train_labels) = iterator_train.get_next()

    return batch_train_images, batch_train_labels


def get_training_params(train_set,train_label,test_set,test_label):
    out, log_probs, nll = net(train_set, targets=train_label, sample=True, n_samples=1, 
                              loss_function=lambda y, y_target: tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_target, logits=y))

    num_batches = (len(mnist.train.labels)//BATCH_SIZE)
    loss = 0.01*log_probs/num_batches + nll
    optim = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optim.minimize( loss )

    out_test_deterministic, _, _ = net(test_set, sample=False, loss_function=None)
    prediction = tf.cast(tf.argmax(out_test_deterministic, 1), tf.int32)
    equality = tf.equal(prediction, test_label)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))
    
    return train_op, loss,accuracy


def train(net,loss,train_op,accuracy,loss2=None,accuracy2=None):
    for i in range(TRAINING_STEPS):
        l, _ = sess.run([loss, train_op])
        l2 = None
        if loss2 is not None:
            l2 = sess.run([loss2])
        if i>=1000 and i%1000==0:
            # Test accuracy
            avg_acc = 0.0
            avg_acc2 = 0.0
            num_iters = len(mnist.test.labels)//BATCH_SIZE
            for test_iter in range(num_iters):
                acc = sess.run(accuracy)
                avg_acc += acc
                if accuracy2 is not None:
                    acc2 = sess.run(accuracy2)
                    avg_acc2 += acc2

            avg_acc /= num_iters
            avg_acc2 /= num_iters
            print("Iteration ", i, "loss: ", l, "accuracy: ", avg_acc, "loss2: ", l2,"accuracy2: ", avg_acc2)
            

            ## Histogram of standard deviations (w and b)
            all_stds = []
            for l in net.layers:
                w_sigma = np.reshape( sess.run(l.w_sigma), [-1] ).tolist()
                b_sigma = np.reshape( sess.run(l.b_sigma), [-1] ).tolist()
                all_stds += w_sigma + b_sigma

            n = TRAINING_STEPS//1000
            plt.rc('axes', prop_cycle=cycler('color', [plt.get_cmap('inferno')(1. * float(i)/n) for i in range(n)]))
            lbl = ""
            if i==1000:
                lbl = "t=1000"
            elif i==1000*(TRAINING_STEPS//1000):
                lbl = "t="+str(1000*(TRAINING_STEPS//1000))
            plt.hist(all_stds, 100, alpha=0.3, label=lbl)

def test(net,loss,accuracy,loss2,accuracy2):
    for i in range(10):
        l, l2 = sess.run([loss, loss2])
        
        # Test accuracy
        avg_acc = 0.0
        avg_acc2 = 0.0
        num_iters = len(mnist.test.labels)//BATCH_SIZE
        for test_iter in range(num_iters):
            acc = sess.run(accuracy)
            avg_acc += acc

            acc2 = sess.run(accuracy2)
            avg_acc2 += acc2

        avg_acc /= num_iters
        print("Iteration ", i, "loss: ", l, "accuracy: ", avg_acc, "loss2: ", l2,"accuracy2: ", avg_acc2)