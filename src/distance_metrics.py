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
        if (p[i]==0 and q[i]==0):
            ps = ps
        else:
            ps = ps + np.square(p[i]-q[i])/(p[i]+q[i])
    return 2*ps

def TS(p,q):
    ts = 0
    for i in range(p.shape[0]):
        if (p[i]==0 or q[i]==0):
            ts = ts
        else:
            ts = ts + p[i]*np.log(2*p[i]/(p[i]+q[i]+0.00001)) + q[i]*np.log(2*q[i]/(p[i]+q[i]+0.00001))
    return ts

def JD_(p, q):
    jd = 0
    eps = 1e-10
    for i in range(p.shape[0]):
        pi = p[i]
        qi = q[i]
        if pi == 0 and qi == 0:
            continue
        term1 = 0
        if pi > 0:
            term1 += pi * np.log(pi + eps)
        if qi > 0:
            term1 += qi * np.log(qi + eps)
        m = (pi + qi) / 2
        term2 = m * np.log(m + eps)
        jd += 0.5 * term1 + term2
    return jd

def JD(p, q, eps=1e-10):
    p = p + eps
    q = q + eps
    p = p / np.sum(p)
    q = q / np.sum(q)
    m = 0.5 * (p + q)
    
    def kl_div(a, b):
        return np.sum(a * np.log(a / b))
    
    jsd = 0.5 * kl_div(p, m) + 0.5 * kl_div(q, m)
    return jsd

def TN(p, q):
    tn = 0
    eps = 1e-10  # To avoid log(0)
    for i in range(p.shape[0]):
        pi, qi = p[i], q[i]
        if pi == 0 and qi == 0:
            continue
        m = (pi + qi) / 2
        g = np.sqrt(pi * qi + eps)
        ratio = m / (2 * g + eps)
        tn += m * np.log(ratio + eps)
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