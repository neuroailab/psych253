import time
import numpy as np
import tensorflow as tf


def tf_optimize(loss,
				optimizer_class,
				target,
				training_data,
				num_iterations,
				optimizer_args=(),
				optimizer_kwargs=None,
				sess=None
			   ):
                           
    if sess is None:
        sess = tf.Session()
        
    if optimizer_kwargs is None:
        optimizer_kwargs = {}
    
    #construct the optimizer
    optimizer = optimizer_class(*optimizer_args, 
                                **optimizer_kwargs)
    optimizer_op = optimizer.minimize(loss)
    
    #initialize variables
    init_op = tf.global_variables_initializer()
    sess.run(init_op)

    targetvals = []
    losses = []
    times = []
    for i in range(num_iterations):
        t0 = time.time()
        output = sess.run({'opt': optimizer_op,
                           'target': target,
                           'loss': loss}, 
                           feed_dict=training_data)
        times.append(time.time() - t0)
        targetvals.append(output['target'])
        losses.append(output['loss'])
    
    print('Average time per iteration --> %.5f' % np.mean(times))
    return np.array(losses), targetvals