from PCA import mnist, center_matrix_SVD
import numpy as np
from Classifiers import KNN
import LDA2D
import pickle
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

def main():
    digits = mnist()
    do_LDA2D_KNN(digits,7,7)
    do_LDA2D_KNN(digits,9,9)
    do_LDA2D_KNN(digits,10,10)
    do_LDA2D_KNN(digits,7,9)
    do_LDA2D_KNN(digits,9,11)
    do_LDA2D_KNN(digits,9,7)

def do_LDA2D_KNN(digits,p,q):
    l,r = LDA2D.iterative2DLDA(digits.train_Images, digits.train_Labels, p, q, 28, 28)

    new_train = np.zeros((digits.train_Images.shape[0],p*q))
    for i in range(digits.train_Images.shape[0]):
        new_train[i] = (np.transpose(l)@digits.train_Images[i].reshape(28,28)@r).reshape(p*q)
    new_test = np.zeros((digits.test_Images.shape[0],p*q))
    for i in range(digits.test_Images.shape[0]):
        new_test[i] = (np.transpose(l)@digits.test_Images[i].reshape(28,28)@r).reshape(p*q)
    myLDA = LDA()
    x = center_matrix_SVD(new_train)
    new_new_train = myLDA.fit_transform(new_train-x.centers,digits.train_Labels)
    new_new_test = myLDA.transform(new_test-x.centers)
    labels, nearest = KNN(new_new_train,digits.train_Labels,new_new_test,10,'euclidean')
    pickle.dump(labels, open('LDA2D'+ str(p) + 'x' + str(q) + '_EU.p','wb'))
    pickle.dump(nearest, open('LDA2D'+ str(p) + 'x' + str(q) + '_EU.p','wb'))
    labels, nearest = KNN(new_new_train,digits.train_Labels,new_new_test,10,'cityblock')
    pickle.dump(labels, open('LDA2D'+ str(p) + 'x' + str(q) + '_CB.p','wb'))
    pickle.dump(nearest, open('LDA2D'+ str(p) + 'x' + str(q) + '_CB.p','wb'))
    labels, nearest = KNN(new_new_train,digits.train_Labels,new_new_test,10,'cosine')
    pickle.dump(labels, open('LDA2D'+ str(p) + 'x' + str(q) + '_CO.p','wb'))
    pickle.dump(nearest, open('LDA2D'+ str(p) + 'x' + str(q) + '_CO.p','wb'))

main()