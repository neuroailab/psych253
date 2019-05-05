"""
implementations of basic methods for finding eigenvalues and eigenvectors, and PCA
"""
import numpy as np


def norm(v):
    return v / np.linalg.norm(v)


def power_method(A, n=200):
    """
    #A = square metrics of shape (k, k)
    #n = number of times we want to iterate the power method
    returns top eigenvalue and corresponding eigenvector
    """
    
    #initial guess for eigenvector
    k = A.shape[1]  #number of dimensions
    v = norm(np.random.rand(k))

    #iterate
    for i in range(n):
        #calculate v --> Av
        v = np.dot(A, v)
        #normalize
        v = norm(v)
        
    #compute corresponding eigenvalue
    l = np.dot(v.T, np.dot(A, v)) / np.dot(v, v)

    return l, v
    

def get_eigenvalues(A, n=200):
    """A is a square matrix
       n = number iterations for power method
        
       returns eigenvals, eigenvecs
             eigenvals in order from biggest to smallest
             eigenvecs in same order as column rectors
    """
    
    k = A.shape[1]
    vals = []
    vecs = []
    for i in range(k):
        l, v = power_method(A, n=n)   #power method
        vals.append(l); vecs.append(v) 
        A = A - l * np.outer(v.T, v)  #deflation
    return np.array(vals), np.array(vecs).T


class PCA(object):
    def __init__(self, n_components):
        self.n_components = n_components
        
    def fit(self, train_data):
        data, self.fmean, self.fvar = featurewise_norm(train_data)
        cov = np.dot(data.T, data)
        eigenvals, eigenvecs = get_eigenvalues(cov)
        self.eigenvals = eigenvals
        self.eigenvecs = eigenvecs
        
    def transform(self, test_data):
        data, _ig, _ig = featurewise_norm(test_data, fmean=self.fmean, fvar=self.fvar)
        Xproj = np.dot(data, self.eigenvecs)
        return Xproj[:, :self.n_components]