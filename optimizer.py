import numpy as np

class BatchReader(object):
    
    def __init__(self,
                 data_dict,
                 batch_size,
                 shuffle=True,
                 shuffle_seed=0,
                 pad=True,
                 processors=None):
        self.data_dict = data_dict        
        self.batch_size = batch_size
        _k = data_dict.keys()[0]
        self.data_length = data_dict[_k].shape[0]
        self.total_batches = (self.data_length - 1) // self.batch_size + 1
        self.curr_batch_num = 0
        self.curr_epoch = 1
        self.pad = pad
        self.shuffle = shuffle
        self.shuffle_seed = shuffle_seed        
        if self.shuffle:
            self.rng = np.random.RandomState(seed=self.shuffle_seed)
            self.perm = self.rng.permutation(self.data_length)
        if processors is None:
            self.processors = {}
        
    def __iter__(self):
        return self

    def next(self):
        return self.get_next_batch()
        
    def get_next_batch(self):
        data = self.get_batch(self.curr_batch_num)
        self.increment_batch_num()
        return data

    def increment_batch_num(self):
        m = self.total_batches
        if (self.curr_batch_num >= m - 1):
            self.curr_epoch += 1
            if self.shuffle:
                self.perm = self.rng.permutation(self.data_length)
        self.curr_batch_num = (self.curr_batch_num + 1) % m

    def get_batch(self, cbn):
        data = {}
        startv = cbn * self.batch_size
        endv = (cbn + 1) * self.batch_size
        if self.pad and endv > self.data_length:
            startv = self.data_length - self.batch_size
            endv = startv + self.batch_size
        for k in self.data_dict:
            if self.shuffle:
                batch_inds = np.sort(self.perm[startv: endv])
            else:
            	batch_inds = slice(startv, endv)
            data[k] = self.get_data(k, batch_inds)
        return data

    def get_data(self, k, inds):
        if k in self.processors:
            processor = self.processors[k]
            return processor(self.data_dict[k], inds)
        else:
            return self.data_dict[k][inds]

    

class TF_Optimizer(object):
    """Make the tensorflow SGD-style optimizer into a scikit-learn compatible class
       Uses BatchReader for stochastically getting data batches.
       
       model_func: function which returns tensorflow nodes for
                     predictions, data_input
        
       loss_func: function which takes model_func prediction output node and 
                  returns tensorflow nodes for
                     loss, label_input
                     
       optimizer_class: which tensorflow optimizer class to when learning the model parameters
       
       batch_size: which batch size to use in training
       
       train_iterations: how many iterations to run the optimizer for 
           --> this should really be picked automatically by like when the training
               error plateaus
               
        model_kwargs:  dictionary of additional arguments for the model_func
        
        loss_kwargs: dictionary of additional arguments for the loss_func
        
        optimizer_args, optimizer_kwargs: additional position and keyword args for the
         optimizer class
         
        sess: tf session to use (will be constructed if not passed) 
        
        train_shuffle: whether to shuffle example order during training
       
    """ 
    
    def __init__(self, 
                 model_func,
                 loss_func, 
                 optimizer_class,
                 batch_size,
                 train_iterations,
                 model_kwargs=None,
                 loss_kwargs=None,
                 optimizer_args=(),
                 optimizer_kwargs=None,
                 sess=None,
                 train_shuffle=False,
                 data_processors=None
                 ):
                        
        self.model_func = model_func
        if model_kwargs is None:
            model_kwargs = {}
        self.model_kwargs = model_kwargs
        self.loss_func = loss_func
        if loss_kwargs is None:
            loss_kwargs = {}
        self.loss_kwargs = loss_kwargs
        self.train_shuffle=train_shuffle
        
        self.train_iterations = train_iterations
        self.batch_size = batch_size
        self.data_processors = data_processors
     
        if sess is None:
            sess = tf.Session()
        self.sess = sess
                
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
        self.optimizer = optimizer_class(*optimizer_args, 
                                         **optimizer_kwargs)
        
    def fit(self, train_data, train_labels):
        fshape = train_data.shape[1:]
        self.data_holder = tf.placeholder(shape=(None,) + fshape,
                                     dtype=tf.float32,
                                     name='data')
        lshape = train_labels.shape[1:]
        self.label_holder = tf.placeholder(shape=(None,) + lshape,
                                     dtype=tf.float32,
                                     name='labels')
    
        self.model = self.model_func(self.data_holder,
        							 self.label_holder,
        							 **self.model_kwargs)
        self.loss = self.loss_func(self.model,
                                   self.label_holder,
                                   **self.loss_kwargs)
        
        self.optimizer_op = self.optimizer.minimize(self.loss)
            
        data_dict = {self.data_holder: train_data,
                     self.labels_holder: train_labels}
        data_processors={}
        if self.data_processors:
            if 'data' in self.data_processors:
                data_processors[self.data_holder] = self.data_processors['data']
            if 'labels' in self.data_processors:
                data_processors[self.labels_holder] = self.data_processors['labels']
        train_data = BatchReader(data_dict=data_dict,
                                 batch_size=self.batch_size,
                                 shuffle=self.train_shuffle,
                                 shuffle_seed=0,
                                 pad=True,
                                 processors=data_processors)
        
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        
        self.losses = []                      
        for i in range(self.train_iterations):
            data_batch = train_data.next()
            output = self.sess.run({'opt': self.optimizer_op,
                                    'loss': self.loss}, 
                                    feed_dict=data_batch)
            self.losses.append(output['loss'])
                
    def predict(self, test_data):
        data_dict = {self.data_holder: test_data}
        data_processors={}
        if self.data_processors:
            if 'data' in self.data_processors:
                data_processors[self.data_holder] = self.data_processors['data']
        test_data = BatchReader(data_dict=data_dict,
                             batch_size=self.batch_size,
                             shuffle=False,
                             pad=False,
                             processors=data_processors)                  
        preds = []
        for i in range(test_data.total_batches):
            data_batch = test_data.get_batch(i)
            pred_batch = self.sess.run(self.model, feed_dict=data_batch)
            preds.append(pred_batch)
        return np.row_stack(preds)
