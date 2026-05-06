import numpy as np
import distance_metrics

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
    return norm_cm

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

    if np.linalg.det(cm_dummy) != 0:
        init_sol = np.linalg.inv(cm_dummy) @ pred_prevalence       # inverse coeficient matrix
    else:
        init_sol = np.linalg.pinv(cm_dummy) @  pred_prevalence      # pseudo inverse coeficient matrix

    init_count = np.round(init_sol*np.sum(pred_count)).astype(int)

    p_sum = np.sum(init_count[init_count>=0])
    n_sum = np.sum(init_count[init_count<0])

    if p_sum>2*(np.sum(pred_count)):
        init_count = pred_count

    if n_sum>2*(np.sum(pred_count)):
        init_count = pred_count
    
    if n_sum<2*(-np.sum(pred_count)):
        init_count = pred_count

    print('correction process started ...', n_sum, ' ',p_sum)
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
    init_estimate = np.array(np.round(init_count*cm_dummy).astype(int))

    ini = np.sum(init_estimate, axis=0)
    rem_vec = init_count - ini

    for i,k  in enumerate(rem_vec):
        if k>0:
            l = np.argmax(init_estimate[i])
            init_estimate[i][l] += k
        elif k<0:
            l = np.argmax(init_estimate[i])
            if init_estimate[i][l]>(-k):
                init_estimate[i][l] += k
            else:
                counter = 0
                rem = k
                while(rem<0):
                    if(init_estimate[i][counter]>0):
                        init_estimate[i][counter] -= 1
                        rem += 1
                    counter = (counter+1)%num_classes
    

    return init_estimate


def prob_examination(prob_train, num_classes):
    issues = 0
    for p in prob_train:
        if p<1/num_classes:
            issues += 1
        elif p<1/num_classes:
            issues += 1
    return issues



def get_train_distributions(y_train, y_train_pred, probs_train, num_classes,n_bins=100):
    # 1. create 2D dictionary x 2 (for probs and for histograms)
    # 2. add probabilities to each combination (pred,actual)
    # 3. make histograms for each combinations
    # 4. store histograms in dictionary for histograms
    # 5. return the histogram distionary
    prob_dictionary = {r: {c: [] for c in range(num_classes)} for r in range(num_classes)}
    hist_dictionary = {r: {c: [] for c in range(num_classes)} for r in range(num_classes)}

    for i in range(len(y_train)):
        prob_dictionary[y_train[i]][y_train_pred[i]].append(probs_train[i])

    for p in range(num_classes):
        for a in range(num_classes):
            hist, bin_edges = np.histogram(np.array(prob_dictionary[p][a]), bins=n_bins, range=(0, 1), density=False)
            hist_dictionary[p][a] = hist / (hist.sum() + 1e-8)

    return hist_dictionary, prob_dictionary


def get_test_distributions(y_test_pred, probs_test, num_classes,n_bins=100):
    # 1. create 1D dictionary x 2 (for probs and for histograms)
    # 2. add probabilities to each combination (pred,actual)
    # 3. make histograms for each combinations
    # 4. store histograms in dictionary for histograms
    # 5. return the histogram distionary
    prob_dictionary = {r: [] for r in range(num_classes)}
    hist_dictionary = {r: [] for r in range(num_classes)}

    for i in range(len(y_test_pred)):
        prob_dictionary[y_test_pred[i]].append(probs_test[i])
    
    for p in range(num_classes):
        hist, bin_edges = np.histogram(np.array(prob_dictionary[p]), bins=n_bins, range=(0, 1), density=False)
        hist_dictionary[p] = hist / (hist.sum() + 1e-8)
    
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
    for a in range(num_classes):
        est_hist = train_hist_dict[a]*init_estimate[a]

    return est_hist / (est_hist.sum() + 1e-6)
    

def get_estimation(train_hist_dictionary, test_hist_dictionary, num_classes, neighborhood_steps, initial_estimate, dm):
    
    final_estimation = []
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

    return final_estimation

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
     

        
            

