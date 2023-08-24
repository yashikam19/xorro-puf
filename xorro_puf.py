import numpy as np
import sklearn
from sklearn.svm import LinearSVC

def my_fit( Z_train ):

    p_column = Z_train[:,64:68]
    q_column = Z_train[:,68:72]
    model={}
    mux0 = np.sum(p_column*2**np.arange(3,-1,-1),axis=1)
    mux1 = np.sum(q_column*2**np.arange(3,-1,-1),axis=1)
    Z_train = np.delete(Z_train,np.s_[64:72],axis=1)
    Z_train = np.insert(Z_train,64,mux0,axis=1)
    Z_train = np.insert(Z_train,65,mux1,axis=1)
    
    for row in Z_train:
        if (row[64]>row[65]):
            temp = row[64]
            row[64]=row[65]
            row[65]=temp
            row[66] = 1- row[66]
    sorted_dataset = sorted(Z_train, key = lambda x: (x[64],x[65]))
    p = sorted_dataset[0][64]
    q = sorted_dataset[0][65]
    train_pq = []

    for row in sorted_dataset:
        if (row[64]==p and row[65]==q):
            train_pq.append(row)
        else:
            clf = LinearSVC(loss='squared_hinge',dual=False)
            clf.fit(np.array(train_pq)[:,:64],np.array(train_pq)[:,-1])
            model[(p,q)] = clf
            p = row[64]
            q = row[65]
            train_pq.clear()
            train_pq.append(row)

    clf = LinearSVC(loss='squared_hinge',dual=False)
    clf.fit(np.array(train_pq)[:,:64],np.array(train_pq)[:,-1])
    model[(p,q)] = clf
    return model					# Return the trained model


def my_predict( X_tst ,model  ):
    predictions = []
    for row in X_tst:
        p_row = row[64:68]
        q_row = row[68:72]
        p_mux = sum(x * 2**i for i, x in enumerate(p_row[::-1]))
        q_mux = sum(x * 2**i for i, x in enumerate(q_row[::-1]))
        if (p_mux < q_mux):
            clf = model[(p_mux,q_mux)]
            prediction = clf.predict(np.array(row[:64]).reshape(1, -1))
            predictions.append(prediction)
        else:
            clf = model[(q_mux,p_mux)]
            prediction = 1 - clf.predict(np.array(row[:64]).reshape(1, -1))
            predictions.append(prediction)
    return np.transpose(predictions)
