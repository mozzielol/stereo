import tensorflow as tf

def set_vanilla_loss(out, log_probs, nll):

	loss = lams*log_probs + nll
	optim = tf.train.AdamOptimizer(learning_rate=0.001)
    train_op = optim.minimize( loss )