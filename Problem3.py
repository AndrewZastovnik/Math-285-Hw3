from PCA import mnist, center_matrix_SVD,class_error_rate
from Classifiers import nvb
import numpy as np
import pickle

import pylab as plt

def main():
    digits = mnist()
    mynvb = nvb()
    x = center_matrix_SVD(digits.train_Images)
    mynvb.fit(digits.train_Images,digits.train_Labels)
    mynvb.fit(x.PCA[:,:50],digits.train_Labels)
    newtest = (digits.test_Images -x.centers)@np.transpose(x.V[:50,:])
    labels = mynvb.predict(newtest)
    errors_50, error_Full_index = class_error_rate(labels,digits.test_Labels)
    prob3_plots(mynvb,digits,newtest,pc=0)
    prob3_plots(mynvb,digits,newtest,pc=1)
    prob3_plots(mynvb,digits,newtest,pc=2)
    prob3_plots(mynvb,digits,newtest,pc=3)


def prob3_plots(mynvb,digits,newtest,pc):
    z= (np.arange(1000)/1000)*10 - 4 #
    y0 = mynvb.likelihood[str(pc)+'l'+str(0)](z)
    y1 = mynvb.likelihood[str(pc)+'l'+str(1)](z)
    y2= mynvb.likelihood[str(pc)+'l'+str(2)](z)
    y3 = mynvb.likelihood[str(pc)+'l'+str(3)](z)
    y4 = mynvb.likelihood[str(pc)+'l'+str(4)](z)
    y5 = mynvb.likelihood[str(pc)+'l'+str(5)](z)
    y6 = mynvb.likelihood[str(pc)+'l'+str(6)](z)
    y7 = mynvb.likelihood[str(pc)+'l'+str(7)](z)
    y8 = mynvb.likelihood[str(pc)+'l'+str(8)](z)
    y9 = mynvb.likelihood[str(pc)+'l'+str(9)](z)
    plt.plot(z,y0, label='0')
    indices = np.nonzero(np.array(digits.test_Labels == 1))
    #plt.plot(newtest[indices,0],np.zeros(len(indices)),marker='o')
    weights = np.ones_like(np.transpose(newtest[indices,pc]))/len(np.transpose(newtest[indices,pc]))
    plt.hist(np.transpose(newtest[indices,pc]),weights=weights)
    plt.plot(z,y1, label='1')
    plt.plot(z,y2, label='2')
    plt.plot(z,y3, label='3')
    plt.plot(z,y4, label='4')
    plt.plot(z,y5, label='5')
    plt.plot(z,y6, label='6',ls='--')
    plt.plot(z,y7, label='7',ls='--')
    plt.plot(z,y8, label='8',ls='--')
    plt.plot(z,y9, label='9',ls='--')
    plt.legend(loc='upper right')
    plt.show()

main()
