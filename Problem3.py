from PCA import mnist, center_matrix_SVD
from Classifiers import nvb
import numpy as np
import pickle

import pylab as plt

def main():
    digits = mnist()
    mynvb = nvb()
    x = center_matrix_SVD(digits.train_Images)
    mynvb.fit(digits.train_Images,digits.train_Labels)
    """
    mynvb.fit(x.PCA[:,:50],digits.train_Labels)
    newtest = digits.test_Images -x.centers
    newtest=newtest@np.transpose(x.V[:50,:])
    """
    labels=mynvb.predict(digits.test_Images)
    pickle.dump(labels,open('NB_Full.p','wb'))
    z= (np.arange(1000)/1000)*10 - 4 #
    y0 = mynvb.likelihood[str(0)+'l'+str(0)](x)
    y1 = mynvb.likelihood[str(0)+'l'+str(1)](x)
    y2= mynvb.likelihood[str(0)+'l'+str(2)](x)
    y3 = mynvb.likelihood[str(0)+'l'+str(3)](x)
    y4 = mynvb.likelihood[str(0)+'l'+str(4)](x)
    y5 = mynvb.likelihood[str(0)+'l'+str(5)](x)
    y6 = mynvb.likelihood[str(0)+'l'+str(6)](x)
    y7 = mynvb.likelihood[str(0)+'l'+str(7)](x)
    y8 = mynvb.likelihood[str(0)+'l'+str(8)](x)
    y9 = mynvb.likelihood[str(0)+'l'+str(9)](x)
    plt.plot(z,y0, label='0')
    indices = np.nonzero(np.array(digits.test_labels == 1))
    #plt.plot(newtest[indices,0],np.zeros(len(indices)),marker='o')
    plt.hist(np.transpose(newtest[indices,0]),normed=True,stacked=True)
    plt.plot(z,y1, label='1')
    plt.plot(z,y2, label='2')
    plt.plot(z,y3, label='3')
    plt.plot(z,y4, label='4')
    plt.plot(z,y5, label='5')
    plt.plot(z,y6, label='6')
    plt.plot(z,y7, label='7')
    plt.plot(z,y8, label='8')
    plt.plot(z,y9, label='9')
    plt.legend(loc='upper right')
    plt.show()

main()
