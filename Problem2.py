from PCA import mnist, center_matrix_SVD, class_error_rate
import numpy as np
import pickle
import pylab as plt
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as LDA

def main():
    digits = mnist()
    x = center_matrix_SVD(digits.train_Images)
    errors_154 = doQDA(x,digits,154)
    pickle.dump(errors_154,open('QDA_154.p','wb'))
    errors_50 = doQDA(x,digits,50)
    pickle.dump(errors_50,open('QDA_50.p','wb'))
    errors_10 = doQDA(x,digits,10)
    pickle.dump(errors_10,open('QDA_10.p','wb'))
    errors_60 = doQDA(x,digits,60)
    pickle.dump(errors_60,open('QDA_60.p','wb'))
    prob1_plots(digits)
    put_into_excel(digits)


def doQDA(x,digits,s):
    myLDA = LDA()
    myLDA.fit(x.PCA[:,:s],digits.train_Labels)
    newtest = digits.test_Images -x.centers
    newtest=newtest@np.transpose(x.V[:s,:])
    labels = myLDA.predict(newtest)
    errors = class_error_rate(labels.reshape(1,labels.shape[0]),digits.test_Labels)
    return errors

def prob1_plots(digits):
    labels_Full = pickle.load(open('KNN_Full','rb'))
    error_Full, error_Full_index = class_error_rate(labels_Full,digits.test_Labels)
    error_154,thing = pickle.load(open('QDA_154.p','rb'))
    error_50,thing = pickle.load(open('QDA_50.p','rb'))
    error_60,thing = pickle.load(open('QDA_60.p','rb'))
    plt.figure()
    plt.bar([0,1,2,3],[error_Full[2],error_154,error_50,error_60])
    plt.title('Bar Plot of Error Rates')
    plt.show()

def put_into_excel(digits):
    labels_Full = pickle.load(open('KNN_Full','rb'))
    error_Full, error_Full_index = class_error_rate(labels_Full,digits.test_Labels)
    error_154,thing = pickle.load(open('QDA_154.p','rb'))
    error_50,thing = pickle.load(open('QDA_50.p','rb'))
    error_60,thing = pickle.load(open('QDA_60.p','rb'))
    errors = np.hstack((error_Full[2],error_154,error_50,error_60))
    import pandas
    df = pandas.DataFrame(errors)
    df.to_excel('Errors_QDA.xls')
main()