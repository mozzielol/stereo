import tensorflow as tf
import sonnet as snt

import numpy as np


def save_model_params(model,sess):

    layers = model.layers
    params = {}
    for idx in range(len(layers)):
        params['w%d'%idx] = (sess.run(layers[idx].w_mean),sess.run(layers[idx].w_sigma))
        params['b%d'%idx] = (sess.run(layers[idx].b_mean),sess.run(layers[idx].b_sigma))
    
    return params


def set_new_params(model,params1, params2,sess):
    layers = model.layers
    new_params = {}

    def kalman(name,i):
        kal_gain = params1['%s%d'%(name,idx)][1]**2 / (params1['%s%d'%(name,idx)][1]**2 + params2['%s%d'%(name,idx)][1]**2)
        new_mean = params1['%s%d'%(name,idx)][0] + (params2['%s%d'%(name,idx)][0] - params1['%s%d'%(name,idx)][0]) * kal_gain
        new_sigma = np.sqrt((1-kal_gain) * params1['%s%d'%(name,idx)][1]**2)

        mean_value = getattr(layers[i],name+'_mean')
        sigma_value = getattr(layers[i],name+'_sigma')
        print(mean_value)
        print(sigma_value)
        sess.run(tf.assign(mean_value,new_mean))
        sess.run(tf.assign(sigma_value,new_sigma))

        return (new_mean,new_sigma)
    for idx in range(len(layers)):
        new_params['w%d'%idx] = kalman('w',idx)
        new_params['b%d'%idx] = kalman('b',idx)

    return new_params

def set_prior_dist(model,params,sess):
    layers = model.layers

    def set_prior(name,i):

        mean_value = getattr(layers[i],name+'_prior_mean')
        sigma_value = getattr(layers[i],name+'_prior_sigma')
        new_mean = sess.run(getattr(layers[i],name+'_mean'))
        new_sigma = sess.run(getattr(layers[i],name+'_sigma'))
        print(sigma_value)
        sess.run(tf.assign(mean_value,new_mean))
        sess.run(tf.assign(sigma_value,new_sigma))

    for idx in range(len(layers)):
        set_prior('w',idx)
        set_prior('b',idx)
