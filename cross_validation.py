"""
library of routines for cross validation
"""
import numpy as np


def get_splits(metadata,
               split_by_func,
               num_splits,
               num_per_class_test,
               num_per_class_train,
               train_filter=None,
               test_filter=None,
               seed=0):
    """
    construct a consistent set of splits for cross validation
    
    arguments: 
        metadata: numpy.rec.array of metadata 
        split_by_func: callable, returns label for spliting data into balanced categories 
                       when applied to metadata
        num_per_class_test: number of testing examples for each unique 
                            split_by category
        num_per_class_train: number of train examples for each unique 
                            split_by category
        train_filter: callable (or None): specifying which subset of the data 
                 to use in training applied on a per-element basis to metadata
        test_filter: callable (or None): specifying which subset of the data 
                 to use in testing applied on a per-element basis to metadata
        seed: seed for random number generator
    """
    
    #define helper function for filtering metadata by desired filter
    def get_possible_inds(metadata, filter):
        inds = np.arange(len(metadata))
        if filter is not None:
            subset = np.array(map(filter, metadata)).astype(np.bool)
            inds = inds[subset]
        return inds
    
    #filter the data by train and test filters
    train_inds = get_possible_inds(metadata, train_filter)
    test_inds = get_possible_inds(metadata, test_filter)
    
    #construct possibly category labels for balancing data
    labels = split_by_func(metadata)
    #for later convenience, get unique values of splitting labels in train and test data
    unique_train_labels = np.unique(labels[train_inds])
    unique_test_labels = np.unique(labels[test_inds])
    
    #seed the random number generator
    rng = np.random.RandomState(seed=seed)
    
    #construct the splits one by one
    splits = []
    for _split_ind in range(num_splits):
        #first construct the testing data
        actual_test_inds = []
        #for each possible test label
        for label in unique_test_labels: 
            #look at all possible stimuli with this label
            possible_test_inds_this_label = test_inds[labels[test_inds] == label]
            #count how many there are
            num_possible_test_inds_this_label = len(possible_test_inds_this_label)
            #make sure there are enough
            err_msg = 'You requested %s per test class but there are only %d available' % (
                    num_per_class_test, num_possible_test_inds_this_label)
            assert num_possible_test_inds_this_label >= num_per_class_test, err_msg
            #select num_per_class_test random examples
            perm = rng.permutation(num_possible_test_inds_this_label)
            actual_test_inds_this_label = possible_test_inds_this_label[
                                                      perm[ :num_per_class_test]]
            actual_test_inds.extend(actual_test_inds_this_label)
        actual_test_inds = np.sort(actual_test_inds)
        
        #now, since the pools of possible train and test data overlap, 
        #but since we don't want the actual train and data examples to overlap at all,
        #remove the chosen test examples for this split from the pool of possible 
        #train examples for this split
        remaining_available_train_inds = np.unique(list(set(
                           train_inds).difference(actual_test_inds)))
        
        #now contruct the train portion of the split
        #basically the same way as for the testing examples
        actual_train_inds = []
        for label in unique_train_labels:
            _this_label = labels[remaining_available_train_inds] == label
            possible_train_inds_this_label = remaining_available_train_inds[_this_label]
            num_possible_train_inds_this_label = len(possible_train_inds_this_label)
            err_msg = 'You requested %s per train class but there are only %d available' % (
                  num_per_class_train, num_possible_train_inds_this_label)
            assert num_possible_train_inds_this_label >= num_per_class_train, err_msg
            perm = rng.permutation(num_possible_train_inds_this_label)
            actual_train_inds_this_label = possible_train_inds_this_label[
                                                      perm[ :num_per_class_train]]
            actual_train_inds.extend(actual_train_inds_this_label)
        actual_train_inds = np.sort(actual_train_inds)
        
        split = {'train': actual_train_inds, 'test': actual_test_inds}
        splits.append(split)
        
    return splits


def validate_splits(splits, labels):
    train_classes = np.unique(labels[splits[0]['train']])
    for split in splits:
        train_inds = split['train']
        test_inds = split['test']
        assert set(train_inds).intersection(test_inds) == set([])
        train_labels = labels[split['train']]
        test_labels = labels[split['test']]
        assert (np.unique(train_labels) == train_classes).all()
        assert set(test_labels) <= set(train_classes)
    return train_classes


