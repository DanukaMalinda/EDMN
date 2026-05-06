from scipy.linalg import norm
from scipy.spatial.distance import euclidean
import numpy as np

_SQRT2 = np.sqrt(2)     # sqrt(2) with default precision np.float64

def hellinger1(p, q):
    return norm(np.sqrt(p) - np.sqrt(q)) / _SQRT2

def hellinger2(p, q):
    return euclidean(np.sqrt(p), np.sqrt(q)) / _SQRT2

def HD(p, q):
    return np.sqrt(np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)) / _SQRT2

def SE(p, q):
    return np.sum(np.square(p-q))

def MH(p,q):
    return np.sum(np.abs(p-q))

def PS(p,q):
    ps = 0
    for i in range(p.shape[0]):
        ps = ps + np.square(p[i]-q[i])/(p[i]+q[i])
    return 2*ps

def TS(p,q):
    ts = 0
    for i in range(p.shape[0]):
        ts = ts + p[i]*np.log(p[i]/(p[i]+q[i]+0.00001)) + q[i]*np.log(q[i]/(p[i]+q[i]+0.00001))
    return ts

def JD(p,q):
    jd = 0
    for i in range(p.shape[0]):
        jd = jd + (p[i]*np.log(p[i]) + q[i]*np.log(q[i]))/2 + ((p[i]+q[i])/2)*np.log((p[i]+q[i])/2)
    return jd

def TN(p,q):
    tn = 0
    for i in range(p.shape[0]):
        tn = tn + ((p[i]+q[i])/2)*np.log((p[i]+q[i])/(2*np.log(np.sqrt(p[i]*q[i]))))
    return tn

def DC(p,q):
    dc = 0
    numer = np.sum(np.square(p-q))
    denom = np.sum(np.square(p)) + np.sum(np.square(q))+0.00001
    dc = numer/denom
    return dc

def JC(p,q):
    jc = 0
    numer = np.sum(np.square(p-q))
    denom = np.sum(np.square(p)) + np.sum(np.square(q)) - np.sum(p*q)+0.00001
    jc = numer/denom
    return jc

def CB(p,q):
    return np.max(np.abs(p-q))

def IP(p,q):
    return np.sum(p*(q*np.sum(p*q)))

def HB(p,q):
    hb = 0
    numer = np.sum(p*q)
    denom = np.sum(np.square(p)) + np.sum(np.square(q)) - np.sum(p*q)+0.00001
    hb = numer/denom
    return hb

def CS(p,q):
    cs = 0
    numer = np.sum(p*q)
    denom = np.sqrt(np.sum(np.square(p))) + np.sqrt(np.sum(np.square(q)))+0.00001
    cs = numer/denom
    return cs

def HM(p,q):
    hm = 0
    for i in range(p.shape[0]):
        hm = hm + (p[i]*q[i])/(p[i]+q[i]+0.00001)
    return 2*hm