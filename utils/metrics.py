import numpy as np

def euc_dist(support,query):
    scores = np.linalg.norm(support[:, None, :] - query[None, :, :], 2, axis=-1)
    #This is commented out but the L1 norm performs slightly better, not sure why that is
    # scores = np.sum(np.abs(support[:, None, :] - query[None, :, :]),axis=2)
    pred = np.argmin(scores,axis=0)
    return pred, -scores

def cos_dist(support,query):
    #Normalization across features done prior to passing them to functions
    # support = support/np.linalg.norm(support,ord=2,axis=1,keepdims=1)
    # query = query/np.linalg.norm(query,ord=2,axis=1,keepdims=1)
    num = np.sum(support[:, None, :]*query[None, :, :],axis=2)
    den1 = np.sqrt(np.sum(support**2,axis=1))
    den2 = np.sqrt(np.sum(query**2,axis=1))
    scores = num/(den1[:, None]*den2[None,:])
    pred = np.argmax(scores,axis=0)
    return pred, scores

def mll_dist(support,query,mll_thresh):
    support[support < 1/mll_thresh] = 1/mll_thresh
    lam = 1/support
    lam[lam > mll_thresh] = mll_thresh
    scores = np.sum(np.log(lam[:, None, :]) - lam[:, None, :]*query[None, :, :],axis=2)
    pred = np.argmax(scores,axis=0)
    return pred, scores

def mll_transductive(support,n_way,query,clip_val,iterations,alpha_c):
    support[support < 1/clip_val] = 1/clip_val
    lam = 1/support
    lam[lam > clip_val] = clip_val
    mus = support
    c_support = np.zeros_like(mus)

    for iloop in range(iterations):
        cdfs = 1 - np.exp(-lam[None, :, :]*query[:, None, :])
        logprobs = np.sum(np.log(lam[:, None, :]) - lam[:, None, :]*query[None, :, :],axis=2)

        pred = np.argmax(logprobs,axis=0)

        for ic in range(n_way):
            temp_q = np.copy(query[pred==ic])
            weights = np.copy(cdfs[pred==ic])
            weights = weights[:,ic,:]
            if np.shape(weights)[0] > 0:
                c_support[ic,:] = np.mean(temp_q*weights,axis=0)
                mus[ic,:] = mus[ic,:]*(1-alpha_c)+alpha_c*c_support[ic,:]
        
        lam = 1/mus
        lam[lam > clip_val] = clip_val

    scores = np.copy(logprobs)
    return pred, scores