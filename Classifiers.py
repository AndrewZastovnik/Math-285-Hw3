"""
=============================================
Classification functions
=============================================

Contains the following classifiers
=============================================
Instance based Classifiers
=============================================
KNN                     K Nearest Neighbors
local_kmeans_class      Local Kmeans

=============================================
Probabilistic Classifiers
=============================================
nvb                     Naive Bayes


=============================================
Misc Functions
=============================================
class_error_rate
inboth
weight_function
mfoldX

"""



import numpy as np
import math

def KNN(I, L, x, k,metric='euclidean',weights = 1):
    """
    I is the matrix of obs
    L are the labels
    x is what we are trying to classify
    k are how many neighbors to look at or whatever
    first we want to create a matrix of distances from each object
    we want to classify to every object in our training set
    """
    from scipy import stats
    from scipy.spatial.distance import cdist
    sizex = len(np.atleast_2d(x))
    label = np.zeros((k,sizex))
    nearest = np.zeros((sizex,k+1))
    for rowsx in range(0, sizex):
        dists = cdist(I, np.atleast_2d(x[rowsx]), metric=metric)
        # Now we should have all our distances in our dist array
        # Next find the k closest neighbors of x
        k_smallest = np.argpartition(dists,tuple(range(1,k+1)),axis=None)
        nearest[rowsx] = k_smallest[:k+1]
        # The next step is to use this info to classify each unknown obj
        # if we don't want to use weights weights should equal 1
        if weights == 1:
            for i in range(0,k):
                label[i,rowsx] = stats.mode(L[k_smallest[:i+1]])[0]
        else:
            labs = np.unique(L)
            for i in range(k):
                lab_weighted = np.zeros(np.unique(L).shape[0])
                d = dists[k_smallest[:i+2]][:,0]
                weights = weight_function(d)
                for p in range(0,labs.shape[0]):
                    indices = inboth(np.arange(0,L.shape[0])[L == labs[p]],k_smallest[:i+2])
                    lab_weighted[p]= np.sum(np.multiply(weights,indices))
                label[i,rowsx] = labs[np.argmax(lab_weighted)]
        if rowsx % 1000 == 1:
            print(rowsx)
    return label, nearest

def local_kmeans_class(I, L, x, k):
    # A local kmeans function
    # takes training set I and training labels L
    # and uses them to classify x for 1:k nearest neighbors
    # Returns the predicted labels for x
    from scipy.spatial.distance import cdist
    sizex = len(np.atleast_2d(x)) # the number of obs in I
    columns = I.shape[1] # Number of factors in I
    label = np.zeros((sizex,k)) # place to put our labels
    #nearest = np.zeros((sizex,10,k,columns))
    for rowsx in range(0, sizex): # loop through every row of I
        dists = cdist(I, np.atleast_2d(x[rowsx]), metric='euclidean') # gets distances
        center = np.zeros((10,k,columns)) # place to put the centeres for each label
        label_order = np.unique(L)
        labs = np.unique(L)
        thing = np.zeros((k,columns)) # place to store the total sums
        for l in np.unique(L).shape[0]:
            indices = L == labs[l] # finds the indices in L that are labs
            k_smallest = np.argpartition(dists[indices],tuple(range(1,k)),axis=None) # sorts the thing
            for i in range(0,k):
                M = I[indices] #matrix with only labs
                if i == 0:
                    thing[i] = M[k_smallest[i+1]]
                else:
                    thing[i] = thing[i-1] + M[k_smallest[i+1]]
            center[l,:,:] = np.divide(thing,np.repeat(np.arange(1,11).reshape(10,1),columns ,axis=1))
            #Now we have the local averages for every lable and k
        for i in range(k): # now we need to find the closed center basically knn again
            #print(cdist(center[:,i,:], np.atleast_2d(x[rowsx]), metric='euclidean'))
            dists2center = cdist(center[:,i,:], np.atleast_2d(x[rowsx]), metric='euclidean')
            k_smallest = np.argpartition(dists2center,tuple(range(1)),axis=None)
            label[rowsx,i] = label_order[k_smallest[0]]
        #nearest[rowsx] = center
        if rowsx % 1000 == 1: #keep track of where we are
            print(rowsx)
    return label#, nearest

class nvb:
    # A Naive Bayes Classifier
    # Has atributes fit distribution and predict
    # fit creates a dictionary of likelihood functions
    def fit(self,train_data,train_labs):
        # Creates the distributions associated with each factor and class
        self.train_labs = train_labs
        self.likelihood = {} #  a dictionary to store our dist functions
        lab = np.unique(train_labs)
        for l in range(lab.shape[0]):
            for factors in range(train_data.shape[1]):
                # Get the indices for a specific class
                indices = np.nonzero(np.array(train_labs == lab[l]))
                # Create likelihood functions
                self.likelihood[str(factors)+'l'+str(lab[l])] = self.distribution(train_data[indices,factors])

    def distribution(self,factor):
        # Creates a distribution give a factor
        mu = np.mean(factor) #  Finds the mean
        s = np.var(factor)  #  Findes the variance
        def normal(x):
            if s == 0: # what do we do if variance is 0?
                # Create a very small box with length 2e-10 around the mean
                if x <=mu +1e-10 and x >= mu - 1e-10:
                    return (0.5e10) # The height of our box +1
                else:
                    return 1e-100  # if it is outside our range likelihood is 0 + 1
                # ok can't add 1 to the likelihood functions
            else:
                # Now our normal distributions
                l = (1/(2*3.14*s)**.5)*2.71828**(-((x-mu)**2)/(2*s))
                if l ==0:
                    # to handel underflow
                    l=1e-100
                return l
        return normal

    def predict(self,test_data):
        # The thing the gives us predictions give test data
        labels = np.zeros(test_data.shape[0])
        for i in range(test_data.shape[0]):
            pofl = np.zeros(np.unique(self.train_labs).shape[0])
            for l in range(np.unique(self.train_labs).shape[0]):
                for factors in range(test_data.shape[1]):
                    pofl[l] += math.log(self.likelihood[str(factors)+'l'+str(l)](test_data[i,factors]))
            labels[i] = np.unique(self.train_labs)[np.argmax(pofl)]
            print(i)
        return labels

    def count_unique(self):
        # Maybe I should use this to get the prior probabilities but I'm not
        labs = np.unique(self.train_labs)
        labs_count = np.zeros(labs.shape) #this probabily will need to be checked
        for l in self.train_labs.shape[0]:
            labs_count[l] = sum(self.train_labs == labs[l])
        return labs, labs_count

def weight_function(d):
    #takes a distance vector d and computes the associated linear weights
    weights = np.add(np.divide(d, np.subtract(np.min(d),np.max(d))),1-np.min(d)/np.subtract(np.min(d),np.max(d)))
    return weights

def inboth(list1,list2):
    # returns a list of 1's and 0's the same length as list2 where 1's mean that index is also in list1
    index = np.zeros(list2.shape)
    for i in range(list2.shape[0]):
        if list2[i] in list1:
            index[i] = 1
    return index

def class_error_rate(pred_labels,true_labels):
    # for calculating the error rate
    error = np.zeros(pred_labels.shape[0])
    for i in range(pred_labels.shape[0]):
        error[i] = sum(pred_labels[i] != true_labels)/pred_labels.shape[1]
    return error

def mfoldX(I, L, m, maxk):
    # I is the trainset
    # L is the Training Labels
    # m is the number of folds
    # maxk is the largest value of k we wish to test
    # first thing to acomplish is to randomly divide the data into m parts
    indices = np.random.permutation(I.shape[0]) # Creates a randomized index vector
    jump = round(len(L) / m) # Calculates the number of rows to jump for each fold
    # The following code cuts up our indices vector into m parts
    # I intended it to handle cases were m % I != 0 but it doesn't so rows(I) needs to be divisible by m
    I_index = indices[:jump]
    L_index = indices[:jump]
    for n in range(1, m - 1): # Iterats through the folds
        # stacks fold into a third diminsion
        I_index = np.dstack((I_index, indices[n * jump:(n + 1) * jump])) # a random index for the images
        L_index= np.dstack((L_index, indices[n * jump:(n + 1) * jump])) # a random index for the labels
    I_index = np.dstack((I_index, indices[(m-1) * jump:]))
    L_index = np.dstack((L_index, indices[(m-1) * jump:]))
    # Yea I'm pretty sure that wasn't necessary. I could have just used jump and the indices
    # but I'm not changing it now
    #
    # now data should be all nice and divided up we need to do something else
    error = np.zeros(maxk) # Creates a array to store our error rates
    for n in range(0, m): # Loop through each fold
        mask = np.ones(m,dtype=bool)
        mask[n]=0
        notn = np.arange(0,m)[mask] # Creates a series of number except for the m we are currently on
        # Creates a Ipt variable that has all
        Ipt = I[I_index[:,:,notn].reshape(((m-1)*I_index.shape[1]))]
        Lpt = L[I_index[:,:,notn].reshape(((m-1)*I_index.shape[1]))]
        label,near = KNN(Ipt,Lpt ,I[I_index[:,:,n].reshape(I_index.shape[1])],10)
        for k in range(10):
            error[k] = error[k] + sum((label[k] != L[L_index[:,:,n]])[0])
    error = error / (len(L))
    return error

