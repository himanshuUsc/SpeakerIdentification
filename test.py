import scipy.cluster.vq as sp
import numpy as np
import os, pickle
from myhmm_scaled import MyHmmScaled as HMM

def read_file(name):
    with open(name) as f:
        observation=f.readlines()
    s=[]
    sequences=[]
    for word in observation:
        word=word[:-1]
        s.append(word)
    sequences.append(s)
    return sequences

def train_machine(data,init_model):
    M=HMM(init_model)
    M.forward_backward_multi_scaled(data)
    return(M)

def predict(seq,hmms):
    prob=[0]*len(hmms)
    max_prob=0
    pos=0
    for hmm in range(len(hmms)):
        prob[hmm]=hmms[hmm].forward_scaled(seq)
    for pr in range(len(prob)):
        if prob[pr]>max_prob:
            max_prob=prob[pr]
            pos=pr+1
    return (max_prob,pr)
    
    
if __name__=="__main__":
    path = "mfcc_vectors_train\\"
    wavfiles=os.listdir(path)#all the training files have been listed
    
    trained_hmm=[0]*len(wavfiles)
    mfcc_data=[0]*len(wavfiles)
    
    for w in range(len(wavfiles)):
        mfcc_data[w]=read_file(path+wavfiles[w])
        
        #trained_hmm[w]=train_machine(mfcc_data[w],"initial.txt")
    path2="mfcc_vectors_test\\"
    wavfiles_test=os.listdir(path2)#all the testing files have been listed
    mfcc_data_test=[0]*len(wavfiles_test)

    for dat in range(len(wavfiles_test)):
        mfcc_data_test[dat]=read_file(path2+wavfiles_test[dat])
        print(mfcc_data_test[dat])
        prediction=predict(mfcc_data_test[dat],trained_hmm)
        print ("testing file number",dat,"has been found for person number",prediction[1],"in the training data with the probability of",prediction[0])
        
        
