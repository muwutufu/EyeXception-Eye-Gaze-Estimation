#!/usr/bin/env python3

import numpy as np

# Example of easy averaging script without looping
Path = '/home/peter/Documents/MP/calculatemode/'

# Calculates mode of distribution, based on the amount of members in the distribution
# Not used in our submission
n = 10 # Nr of members in distribution
def modefm(mi, ma, ar):
    eps = 0.0000001
    step = (ma-mi)/n
    ma = ma+eps
    mi = mi-eps
    ttot = 0
    tcount = 0
    begin = 2
    while True:
        for i in range(n):
            count = 0
            tot = 0
            for v in ar:
                if v >= mi+i*step and v < mi+(i+1)*step:
                    count += 1
                    tot += v
            if count >= begin:
                tcount += count
                ttot += tot

        if tcount != 0:
            return ttot/tcount
        else:
            begin -= 1

b1 = np.loadtxt(Path+'predictions1.txt.gz')
# b1 = b1[:,np.newaxis,:]
b2 = np.loadtxt(Path+'predictions2.txt.gz')
# b2 = b2[:,np.newaxis,:]
b3 = np.loadtxt(Path+'predictions3.txt.gz')
# b3 = b3[:,np.newaxis,:]
b4 = np.loadtxt(Path+'predictions4.txt.gz')
# b4 = b4[:,np.newaxis,:]
b5 = np.loadtxt(Path+'predictions5.txt.gz')
# b5 = b5[:,np.newaxis,:]
b6 = np.loadtxt(Path+'predictions6.txt.gz')
# b6 = b6[:,np.newaxis,:]
b7 = np.loadtxt(Path+'predictions7.txt.gz')
# b7 = b7[:,np.newaxis,:]
b8 = np.loadtxt(Path+'predictions8.txt.gz')
# b8 = b8[:,np.newaxis,:]
b9 = np.loadtxt(Path+'predictions9.txt.gz')
# b9 = b9[:,np.newaxis,:]
b10 = np.loadtxt(Path+'predictions10.txt.gz')
# b10 = b10[:,np.newaxis,:]

# fulll = np.concatenate([b1[:,:,0],b2[:,:,0],b3[:,:,0],b4[:,:,0],b5[:,:,0],b6[:,:,0],b7[:,:,0],b8[:,:,0],b9[:,:,0],b10[:,:,0]],1)
# fullr = np.concatenate([b1[:,:,1],b2[:,:,1],b3[:,:,1],b4[:,:,1],b5[:,:,1],b6[:,:,1],b7[:,:,1],b8[:,:,1],b9[:,:,1],b10[:,:,1]],1)
# print(fulll.shape)
# smode = np.zeros([len(fulll),2])
# for i in range(len(fulll)):
#    mil = np.min(fulll[i])
#    mal = np.max(fulll[i])
#    mir = np.min(fullr[i])
#    mar = np.max(fullr[i])
#    smode[i,0]=modefm(mil,mal,fulll[i])
#    smode[i,1]=modefm(mir,mar,fullr[i])


tot = (b1 + b2 + b3 + b4 + b5 + b6 + b7 + b8 + b9 + b10) / 10
np.savetxt('/home/peter/Documents/MP/calculatemode/predictions.txt.gz',tot)