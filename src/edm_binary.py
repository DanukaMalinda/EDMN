import numpy as np
import distance_metrics
import itertools
from scipy.ndimage import gaussian_filter1d

def get_it(y_train, y_train_pred,y_test_pred, num_classes):
    cm = np.zeros((num_classes, num_classes))  
    for i in range(len(y_train)):
        cm[y_train_pred[i]][y_train[i]] += 1   

    norm_cm = cm.T/((np.sum(cm,axis=1))+1e-8).T
    return norm_cm



    
def get_coeficient_matrix(y_train, y_train_pred, num_classes):
    cm = np.zeros((num_classes, num_classes))  # intiate coeficient matrix
    for i in range(len(y_train)):
        cm[y_train_pred[i]][y_train[i]] += 1   # increment the matrix (pred, actual)
    
    # norm_cm = cm.T/((np.sum(cm,axis=1)).reshape(-1, 1)+1e-8).T                   # normalize confution matrix
    norm_cm = cm.T/((np.sum(cm,axis=1))+1e-8)
    return norm_cm.T

def basic_solution(norm_cm, pred_count):
    cm = norm_cm.T*pred_count
    return cm.T

def get_init_solution(norm_cm, y_test_pred, num_classes):

    #make sure no empty row or column in norm_cm
    cm_dummy = norm_cm.copy()
    for i,r in enumerate(cm_dummy):
        if np.sum(r) == 0:
            cm_dummy[i][i] = 1

    pred_count = np.zeros(num_classes)    # intiate predicted class distribution
    for i in range(len(y_test_pred)):
        pred_count[y_test_pred[i]] += 1   # populate predicted class distribution
    
    pred_prevalence = pred_count/np.sum(pred_count)
    total = np.sum(pred_count)

    bs = basic_solution(norm_cm, pred_count)

    if np.linalg.det(cm_dummy) != 0:
        init_sol = np.linalg.inv(cm_dummy) @ pred_prevalence       # inverse coeficient matrix
    else:
        init_sol = np.linalg.pinv(cm_dummy) @  pred_prevalence      # pseudo inverse coeficient matrix

    # init_count = np.round(init_sol*np.sum(pred_count)).astype(int)
    init_count = np.sum(bs, axis=0).astype(int)

    p_sum = np.sum(init_count[init_count>=0])
    n_sum = np.sum(init_count[init_count<0])

    if p_sum>2*(np.sum(pred_count)):
        print('adjustment created too large number for one class')
        init_count = pred_count

    elif n_sum>2*(np.sum(pred_count)):
        print('adjustment created too large number for one - class')
        init_count = pred_count
    
    elif n_sum<2*(-np.sum(pred_count)):
        print('adjustment created too small number for one class')
        init_count = pred_count

    else:
        # print('correction process started ...', n_sum, ' ',p_sum)
        last_index = 0
        if n_sum<0:
            for i,x in enumerate(init_count):
                if x>0:
                    init_count[i] = round(x*n_sum/p_sum) + init_count[i]
                    last_index = i
                else:
                    init_count[i] = 0
            
        remaining = np.sum(pred_count) - np.sum(init_count)

        if (remaining>0 or init_count[last_index]>(-remaining)):
            init_count[last_index] = init_count[last_index] + remaining
        else:
            counter = 0
            while(remaining<0):
                if(init_count[counter]>0):
                    init_count[counter] -= 1
                    remaining += 1
                counter = (counter+1)%num_classes

    # print('cm dummy: ',cm_dummy)
    init_estimate = np.array(np.round(cm_dummy.T*init_count).astype(int)).T

    # print('making 2 more different variations of initial guesses')
    counter_count = pred_count-init_count
    # print('ini count: ', init_count)
    # print('pred count: ', pred_count)
    # print('counter count: ', counter_count)

    # lower guess
    # lower guess
    if (((pred_count[0]+counter_count[0])>=0) and ((pred_count[1]-counter_count[0])>=0)):
        init_count_2 = np.array([pred_count[0]+counter_count[0], pred_count[1]-counter_count[0]])

    elif (((pred_count[0]+counter_count[0])>=0) and ((pred_count[1]-counter_count[0])<0)):
        adj = pred_count[1]
        init_count_2 = np.array([pred_count[0]+adj, pred_count[1]-adj])

    elif (((pred_count[0]+counter_count[0])<0) and ((pred_count[1]-counter_count[0])>=0)):
        adj = -pred_count[0]
        init_count_2 = np.array([pred_count[0]+adj, pred_count[1]-adj])

    else:
        adj = np.min(np.abs(pred_count[0]), np.abs(pred_count[1]))
        init_count_2 = np.array([pred_count[0]+adj, pred_count[1]-adj])
       


    # lower guess
    if (((pred_count[0]+counter_count[1])>=0) and ((pred_count[1]-counter_count[1])>=0)):
        init_count_3 = np.array([pred_count[0]+counter_count[1], pred_count[1]-counter_count[1]])

    elif (((pred_count[0]+counter_count[1])>=0) and ((pred_count[1]-counter_count[1])<0)):
        adj = pred_count[1]
        init_count_3 = np.array([pred_count[0]+adj, pred_count[1]-adj])

    elif (((pred_count[0]+counter_count[1])<0) and ((pred_count[1]-counter_count[1])>=0)):
        adj = -pred_count[0]
        init_count_3 = np.array([pred_count[0]+adj, pred_count[1]-adj])

    else:
        adj = np.min(np.abs(pred_count[0]), np.abs(pred_count[1]))
        init_count_3 = np.array([pred_count[0]+adj, pred_count[1]-adj])

    init_estimate_2 = np.array(np.round(cm_dummy.T*init_count_2).astype(int)).T
    init_estimate_3 = np.array(np.round(cm_dummy.T*init_count_3).astype(int)).T

    ini_c = [init_count, init_count_2, init_count_3]
    ini_est = [init_estimate, init_estimate_2, init_estimate_3]


    for r in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]:
        count_dummy = [r*total, (1-r)*total]
        dummy_est = np.array(np.round(cm_dummy.T*count_dummy).astype(int)).T
        ini_c.append(count_dummy)
        ini_est.append(dummy_est)
    

    # print('initial guesses done')
    # print(init_estimate)
    # print(init_estimate_2)
    # print(init_estimate_3)

    ini_c = [init_count, init_count_2, init_count_3]
    ini_est = [init_estimate, init_estimate_2, init_estimate_3]
    # ini_est = [init_estimate] #, init_estimate_2, init_estimate_3]

    for z in range(len(ini_est)):
        # print('choice: ', z)
        ini = np.sum(ini_est[z], axis=1)
        
        rem_vec = ini_c[z] - ini
        # print('ini count: ', ini_c[z], ' ini: ', ini, ' rem_vec: ', rem_vec)

        for i,k  in enumerate(rem_vec):
            if k>0:
                l = np.argmax(ini_est[z][i])
                ini_est[z][i][l] += k
            elif k<0:
                l = np.argmax(ini_est[z][i])
            
                if ini_est[z][i][l]>(-k):
                    ini_est[z][i][l] += k
                else:
                    counter = 0
                    rem = k
                    while(rem<0):
                        if(ini_est[z][i][counter]>0):
                            ini_est[z][i][counter] -= 1
                            rem += 1
                        counter = (counter+1)%num_classes
    
    # print('returned initial guesses!')
    return ini_est


def prob_examination(prob_train, num_classes):
    issues = 0
    for p in prob_train:
        if p<1/num_classes:
            issues += 1
        elif p<1/num_classes:
            issues += 1
    return issues

def getAccuracy(y_act,y_pred):
    correct = 0
    for i in range(len(y_act)):
        if y_act[i] == y_pred[i]:
            correct += 1

    return correct*100/len(y_act)


def _smooth_hist(raw_counts, n_bins, sigma=1.5, alpha=None):
    """
    Apply Gaussian smoothing and Laplace prior to a raw histogram count array.

    Steps
    -----
    1. Gaussian smoothing (sigma in bins): spreads sharp peaks so that
       adjacent bins share mass.  Prevents a narrow spike in a minor-class
       histogram from looking like a real signal in an empty region.
    2. Laplace (uniform) prior (alpha per bin): adds a small constant to
       every bin before normalising.  Guarantees every bin has non-zero
       mass, so the weighted mixture always has full support and minor
       components cannot "own" bins that the major component leaves empty.

    Parameters
    ----------
    raw_counts : np.ndarray  shape (n_bins,)  — output of np.histogram
    n_bins     : int
    sigma      : float — Gaussian std in bins (default 1.5; set 0 to skip)
    alpha      : float — Laplace prior per bin (default 1/n_bins)

    Returns
    -------
    np.ndarray  normalised to sum to 1
    """
    if alpha is None:
        alpha = 1.0 / n_bins          # weak uniform prior

    h = raw_counts.astype(float)
    if sigma > 0:
        h = gaussian_filter1d(h, sigma=sigma)   # smooth across neighbouring bins
    h = h + alpha                               # Laplace prior — no zero bins
    return h / h.sum()


def get_train_distributions(y_train, y_train_pred, probs_train, num_classes,
                            n_bins=100, sigma=1.5, alpha=None):
    prob_dictionary = {r: {c: [] for c in range(num_classes)} for r in range(num_classes)}
    hist_dictionary = {r: {c: [] for c in range(num_classes)} for r in range(num_classes)}

    for i in range(len(y_train)):
        prob_dictionary[y_train[i]][y_train_pred[i]].append(probs_train[i])

    for p in range(num_classes):
        for a in range(num_classes):
            raw, _ = np.histogram(np.array(prob_dictionary[p][a]),
                                  bins=n_bins, range=(0, 1), density=False)
            hist_dictionary[p][a] = _smooth_hist(raw, n_bins, sigma, alpha)

    return hist_dictionary, prob_dictionary


def get_test_distributions(y_test_pred, probs_test, num_classes,
                           n_bins=100, sigma=1.5, alpha=None):
    prob_dictionary = {r: [] for r in range(num_classes)}
    hist_dictionary = {r: [] for r in range(num_classes)}

    for i in range(len(y_test_pred)):
        prob_dictionary[y_test_pred[i]].append(probs_test[i])

    for p in range(num_classes):
        raw, _ = np.histogram(np.array(prob_dictionary[p]),
                              bins=n_bins, range=(0, 1), density=False)
        hist_dictionary[p] = _smooth_hist(raw, n_bins, sigma, alpha)

    return hist_dictionary, prob_dictionary

def get_steps(num_classes, current_combination=[]):
    # generate total neighborhood adjustments
    arr = [-1,0,1]
    if num_classes == 0:
        if sum(current_combination) == 0:
            yield tuple(current_combination)
        return
    
    for element in arr:
        new_combination = current_combination + [element]
        yield from get_steps(num_classes-1, new_combination)

def get_additional_neighbors(num_classes, step=2):
    directions = []
    index_pairs = list(itertools.combinations(range(num_classes), 2))
    
    for i, j in index_pairs:
        for sign in [-1, 1]:
            vec = [0] * num_classes
            vec[i] = -step * sign
            vec[j] = step * sign
            directions.append(tuple(vec))
    return np.array(directions)

def get_distance(estimated_hist, predicted_hist, dm):
    normalized_eh = estimated_hist/ (estimated_hist.sum() + 1e-8)
    normalized_ph = predicted_hist/ (predicted_hist.sum() + 1e-8)

    if (dm=='SE'):
        distance = distance_metrics.SE(normalized_eh, normalized_ph)
    elif(dm=='MH'):
        distance = distance_metrics.MH(normalized_eh, normalized_ph)
    elif(dm=='PS'):
        distance = distance_metrics.PS(normalized_eh, normalized_ph)
    elif(dm=='TS'):
        distance = distance_metrics.TS(normalized_eh, normalized_ph)
    elif(dm=='JD'):
        distance = distance_metrics.JD(normalized_eh, normalized_ph)
    elif(dm=='TN'):
        distance = distance_metrics.TN(normalized_eh, normalized_ph)
    elif(dm=='DC'):
        distance = distance_metrics.DC(normalized_eh, normalized_ph)
    elif(dm=='JC'):
        distance = distance_metrics.JC(normalized_eh, normalized_ph)
    elif(dm=='CB'):
        distance = distance_metrics.CB(normalized_eh, normalized_ph)
    elif(dm=='IP'):
        distance = distance_metrics.IP(normalized_eh, normalized_ph)
    elif(dm=='HB'):
        distance = distance_metrics.HB(normalized_eh, normalized_ph)
    elif(dm=='CS'):
        distance = distance_metrics.CS(normalized_eh, normalized_ph)
    elif(dm=='HM'):
        distance = distance_metrics.HM(normalized_eh, normalized_ph)
    else:
        distance = distance_metrics.HD(normalized_eh, normalized_ph)

    return distance


def make_distributions(train_hist_dict, init_estimate, num_classes):
    """
    Build the estimated probability histogram for one predicted class by
    forming a weighted mixture of the class-conditional training histograms.

    Each class a contributes its histogram train_hist_dict[a] weighted by
    its estimated count init_estimate[a].  Weights are normalised so the
    mixture always sums to 1 regardless of the absolute count values.

    Because the training histograms were already Gaussian-smoothed and
    given a Laplace prior (in get_train_distributions), every class has
    full-support mass across all bins.  This means:
      - minor components blend smoothly rather than spiking in empty bins
      - the mixture reflects true proportional contributions at every bin
    """
    weights = np.clip(init_estimate, 0, None).astype(float)
    total   = weights.sum() + 1e-8

    est_hist = np.zeros_like(train_hist_dict[0], dtype=float)
    for a in range(num_classes):
        est_hist += (weights[a] / total) * train_hist_dict[a]

    return est_hist / (est_hist.sum() + 1e-6)
    

def get_estimation(train_hist_dictionary, test_hist_dictionary, num_classes, neighborhood_steps, ini_estimates, dm):
    estimation = []
    best_distance = np.inf
    count = 0
    position = count

    for initial_estimate in ini_estimates:
        final_estimation = []
        total_distance = 0
        for p in range(num_classes):
            pred_hist = test_hist_dictionary[p]
            init_est_hist = make_distributions(train_hist_dictionary[p], initial_estimate[p], num_classes)
            init_distance = get_distance(init_est_hist, pred_hist, dm)

            min_distance = init_distance
            best_estimation = initial_estimate[p]
            # print('estimating ', p)

            while True:
                neighborhood = neighborhood_steps + best_estimation
                
                
                for current_neighbor in neighborhood:
                    # print('current neighbor: ',current_neighbor)
                    if not np.all(current_neighbor>= 0): continue

                    est_hist = make_distributions(train_hist_dictionary[p], current_neighbor, num_classes)
                    distance = get_distance(est_hist, pred_hist, dm)

                    if (distance < min_distance):
                        min_distance = distance
                        best_estimation = current_neighbor
            
                if min_distance < init_distance:
                    init_distance = min_distance
                else: break
            
            final_estimation.append(best_estimation)
            total_distance += min_distance
        
        if total_distance < best_distance:
            position = count
            estimation = final_estimation
            best_distance = total_distance

        count += 1

    return estimation, position

def get_actual_count(test_labels, num_classes):
    act = np.zeros(num_classes)
    for i in test_labels:
        act[i] += 1
    
    return act

def AE(act, est, num_classes):
    return np.sum(np.absolute(np.array(act)-np.array(est)))/(np.sum(act)*num_classes)


def KLD(act, est, num_classes):
    kld = 0
    for i in range(num_classes):
        if act[i]==0:
            kld = 0
        elif est[i]==0:
            log_p = np.log(act[i]/1e-3)
            kld += act[i]*log_p/np.sum(act)
        else:
            log_p = np.log(act[i]/est[i])
            kld += act[i]*log_p/np.sum(act)
    
    return kld

def NKLD(act, est, num_classes):
    kld = KLD(act, est, num_classes)
    return 2*((np.exp(kld))/(1+ np.exp(kld))) - 1
     

        
            

