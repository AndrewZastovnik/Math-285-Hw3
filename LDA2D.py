import numpy as np

def lda2d(train_data,train_labels,classnumber,height,width):
    eachclass = np.zeros(1,classnumber)
    m = train_data.shape
    mean_train = np.mean(train_data,axis=1)
    mean_train1 = mean_train.reshape(height,width)
    sum0 = np.zeros(height,height)
    for i in range(m[1]):
        train1 = mean_train1[:i].reshape(height,width)
        sum0 = sum0 + (train1 - mean_train1)@np.transpose(train1 - mean_train1)
    sb = np.zeros(height,height)
    for i in range(classnumber):
        dd = np.where(train_labels == i)
        n = dd.shape
        eachclass[1,i] = n[0]
        train1 = train_data[:,dd]
        meantraintrain = np.mean(train1,axis= 1)
        traintrain1 = meantraintrain.reshape(height,width)
        sb = sb+n[0]*(traintrain1 - mean_train1)@np.transpose(traintrain1 - mean_train1)
    Sum1=sum0-sb
    t = np.rank(Sum1)
    s,u = np.linalg.eig(np.linalg.pinv(Sum1)@sb)
    tt = np.diagflat(s)
    ind = tt.argsort()[::-1]
    u1 = u[:,ind]
    return u1

def iterative2DLDA(Trainset, LabelTrain, p, q,r, c):
    m = Trainset.shape
    classnumber = max(LabelTrain)
    aa = {}
    for i in range(classnumber):
        temp= np.arange(LabelTrain.size).reshape(LabelTrain.shape)[np.where(LabelTrain==i)]
        temp1=temp
        m1 = temp1.shape
        Trainset1 = Trainset[:,temp1]
        aa[i] = np.mean(Trainset1,axis=1)
    bb = np.mean(Trainset,axis=1)
    bb1 = np.transpose(bb)
    R = np.vstack((np.eye(q,q),np.zeros((c-q,q))))
    for j in range(10):
        sb1=np.zeros((r,r))
        sw1=np.zeros((r,r))
        for i in range(classnumber):
            temp=np.where(LabelTrain==i)
            temp1=np.transpose(temp)
            m1 = temp1.shape
            Trainset1=Trainset[:,temp1]
            m2 = Trainset1.shape
            for s in range(m2[1]):
                sw1=sw1+Trainset1[:,s].reshape(r,c) - \
                    aa[i].reshape(r,c)@R@np.transpose(R)@Trainset1[:,s].reshape(r,c) - aa[i].reshape(r,c)
            sb1=sb1+m1[0]*aa[i].reshape(r,c) - bb1.reshape(r,c)@R@np.transpose(R)@aa[i].reshape(r,c)-bb1.reshape(r,c)
        s,u = np.linalg.eig(np.linalg.pinv(sw1)@sb1)
        tt = s
        ind = tt.argsort()[::-1]
        u11 = u[:,ind]
        L = u11[:,:p]
        #Seriously?
        sb2 = np.zeros((c,c))
        sw2 = np.zeros((c,c))
        for i in range(classnumber):
            temp= np.arange(LabelTrain.size).reshape(LabelTrain.shape)[np.where(LabelTrain==i)]
            temp1=temp
            m1 = temp1.shape
            Trainset1 = Trainset[:,temp1]
            m2 = Trainset1.shape
            for s in range(m2[1]):
                sw2=sw2+Trainset1[:,s].reshape(r,c) - \
                    np.transpose(aa[i].reshape(r,c))@L@np.transpose(L)@Trainset1[:,s].reshape(r,c) - aa[i].reshape(r,c)
            sb2=sb2+m1[0]*aa[i].reshape(r,c) - np.transpose(bb1.reshape(r,c))\
                                               @L@np.transpose(L)@aa[i].reshape(r,c)-bb1.reshape(r,c)
        s1,u1 = np.linalg.eig(np.linalg.pinv(sw2)@sb2)
        tt1 = s1
        ind1 = tt1.argsort()[::-1]
        u12 = u1[:,ind1]
        R = u12[:,:q]
        print(j)
    return L,R
