import scipy.io.wavfile as wvf
import scipy.cluster.vq as sp
import numpy as np
import os
import re
import sys
from features import mfcc



path = r"C:\Users\ashish\Desktop\speaker_identification\audio_recordings_train\\"
path2=r"C:\Users\ashish\Desktop\speaker_identification\mfcc_vectors_train\\"
wavfiles=os.listdir(path)
#print(wavfiles)
rate,sig = wvf.read("abhijeet.wav")
mfcc_feat = mfcc(sig,rate)
codebook = sp.kmeans(mfcc_feat, 16)[0]

for wavfile in wavfiles:
    final = []
    rate,sig = wvf.read(path+wavfile)
    mfcc_feat = mfcc(sig,rate)
    data = sp.vq(mfcc_feat,codebook)
    #print(len(data[0]))
    for i in data[0]:
        final.append(i)
    f = open(path2+wavfile.split(".")[0]+"_vq.txt","w")
    for i in final:
        f.write(str(i)+"\n")
    f.close()
