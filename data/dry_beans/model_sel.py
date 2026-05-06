import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import prep as dtpr
import edm
import matplotlib.pyplot as plt
import csv

# Reproducibility
torch.manual_seed(42)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


# *****************************
# *****************************
# *****************************
# *****************************
print('seeting up the model ...........................')
# Model definition
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.001)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    

class Conv1DMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(Conv1DMLP, self).__init__()

        self.input_dim = input_dim

        # Conv1D expects input shape: [batch_size, channels, sequence_length]
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.pool = nn.AdaptiveAvgPool1d(16)  # reduce to fixed length

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 16, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn6 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, num_classes)

        self.dropout = nn.Dropout(0.1)

    def forward(self, x, temperature=1.0):
        # Reshape input from [B, D] to [B, 1, D] for Conv1D
        x = x.unsqueeze(1)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.bn3(self.fc1(x)))
        x = self.dropout(F.relu(self.bn4(self.fc2(x))))
        x = self.dropout(F.relu(self.bn5(self.fc3(x))))
        x = F.relu(self.bn6(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    

# *****************************
# *****************************
# *****************************
# *****************************
dataset = dtpr.prep_data(True)
print(dataset.head())

# dataset
X_n = dataset.iloc[:, :-1].values.astype(np.float32)
y_n = dataset.iloc[:, -1].values.astype(np.int64)

# Hyperparameters
# num_classes = len(np.unique(y_n))
# input_dim = X_n.shape[1]
# train_size = int(X_n.shape[0]*0.25)
hidden_dim = 64
epochs = 50
batch_size = 32
lr = 0.001
bins = 100
lambda_dist = 1.0  # Weight for distribution matching loss

print('training data shape: ', X_n.shape, ' target shape: ', y_n.shape)

# # Split into train+val and test
# print('splitting data ...............')
# X = torch.from_numpy(X_n) 
# y = torch.from_numpy(y_n)


# *****************************
# *****************************
# Ratios and prevalences
# *****************************
# *****************************
# split ratios
test_size_list = [0.3,0.5,0.7,0.9]

# train prevelances
train_3_prevalences = [[0.2  ,0.5 ,0.3],
                       [0.05 ,0.8 ,0.15],
                       [0.35 ,0.3 ,0.35]]

train_4_prevalences = [[0.5 ,0.3 ,0.1  ,0.1],
                       [0.7 ,0.1 ,0.1  ,0.1],
                       [0.25,0.25,0.25 ,0.25]]

train_5_prevalences = [[0.2  ,0.15 ,0.35 ,0.1  ,0.2],
                       [0.35 ,0.25 ,0.15 ,0.15 ,0.1],
                       [0.2  ,0.2  ,0.2  ,0.2  ,0.2]]

train_6_prevalences = [[0.1  ,0.2  ,0.1  ,0.1  ,0.25, 0.25],
                       [0.05 ,0.1  ,0.3  ,0.4  ,0.1  ,0.05],
                       [0.17 ,0.17 ,0.16 ,0.17 ,0.17 ,0.16]]

train_7_prevalences = [[0.2  ,0.3  ,0.2  ,0.15 ,0.05 ,0.05 ,0.05],
                       [0.05 ,0.1  ,0.05 ,0.05 ,0.25 ,0.3  ,0.2],
                       [0.15 ,0.14 ,0.14 ,0.15 ,0.14 ,0.14 ,0.14]]

train_10_prevalences = [[0.05 ,0.2  ,0.05 ,0.1  ,0.05 ,0.25 ,0.05 ,0.05 ,0.1  ,0.1],
                        [0.15 ,0.05 ,0.2  ,0.05 ,0.1  ,0.05 ,0.2  ,0.1  ,0.05 ,0.05],
                        [0.1  ,0.1  ,0.1  ,0.1  ,0.1  ,0.1  ,0.1  ,0.1  ,0.1  ,0.1]]

train_prev_dictionary = {3:train_3_prevalences, 
                         4:train_4_prevalences,
                         5:train_5_prevalences,
                         6:train_6_prevalences,
                         7:train_7_prevalences,
                         10:train_10_prevalences}

# test prevelances
test_3_prevalences = [[0.1  ,0.7  ,0.2],
                      [0.55 ,0.1  ,0.35],
                      [0.35 ,0.55 ,0.1],
                      [0.4  ,0.25 ,0.35],
                      [0.0  ,0.05 ,0.95]]

test_4_prevalences = [[0.65, 0.25 ,0.05 ,0.05],
                      [0.2,  0.25 ,0.3  ,0.25],
                      [0.45, 0.15 ,0.2  ,0.2],
                      [0.2,  0.0  ,0.0  ,0.8],
                      [0.3,  0.25 ,0.35 ,0.1]]

test_5_prevalences = [[0.15 ,0.1  ,0.65 ,0.1  ,0],
                      [0.45 ,0.1  ,0.3  ,0.05 ,0.1],
                      [0.2  ,0.25 ,0.25 ,0.1  ,0.2],
                      [0.35 ,0.05 ,0.05 ,0.05 ,0.5],
                      [0.05 ,0.25 ,0.15 ,0.15 ,0.4]]

test_6_prevalences = [[0.15 ,0.1  ,0.55 ,0.1  ,0.0  ,0.1],
                      [0.4  ,0.1  ,0.25 ,0.05 ,0.1  ,0.1],
                      [0.2  ,0.2  ,0.2  ,0.1  ,0.2  ,0.1],
                      [0.35 ,0.05 ,0.05 ,0.05 ,0.05 ,0.45],
                      [0.05 ,0.25 ,0.15 ,0.15 ,0.1  ,0.3]]

test_7_prevalences = [[0.1  ,0.1  ,0.1  ,0.5  ,0.1  ,0.0  ,0.1],
                      [0.4  ,0.1  ,0.2  ,0.05 ,0.1  ,0.1  ,0.05],
                      [0.15 ,0.2  ,0.15 ,0.1  ,0.2  ,0.1  ,0.1],
                      [0.3  ,0.05 ,0.05 ,0.05 ,0.05 ,0.05 ,0.45],
                      [0.05 ,0.25 ,0.1  ,0.15 ,0.1  ,0.3  ,0.05]]

test_10_prevalences = [[0.1  ,0.2  ,0.1  ,0.1  ,0.2  ,0.1  ,0    ,0.1  ,0.05 ,0.05],
                       [0.2  ,0.05 ,0.15 ,0.05 ,0.1  ,0.15 ,0.05 ,0.05 ,0.1  ,0.1],
                       [0    ,0.1  ,0.05 ,0.1  ,0.05 ,0.1  ,0.1  ,0.15 ,0.15 ,0.2],
                       [0.05 ,0.05 ,0.05 ,0.35 ,0.15 ,0.05 ,0    ,0.1  ,0.1  ,0.1],
                       [0.05 ,0.1  ,0.1  ,0.15 ,0.1  ,0.15 ,0.05 ,0.1  ,0.1  ,0.1]]


test_prev_dictionary = {3:test_3_prevalences, 
                         4:test_4_prevalences,
                         5:test_5_prevalences,
                         6:test_6_prevalences,
                         7:test_7_prevalences,
                         10:test_10_prevalences}


for k,l in train_prev_dictionary.items():
    print(k," sum:", np.sum(np.array(l)))

for k,l in test_prev_dictionary.items():
    print(k," sum:", np.sum(np.array(l)))

seed_list = [11,42,73,34,25,96,57,88,50,31]
bin_list = [10,20,30,40,50,60,70,80,90,100]
# -------------------------------------------
# 1. get the size of the dataset
# 2 .group data based on class 
# 3. for each split
# 4. get the size of train and test
# 5. determine the number of data point per class for train
# 6. determine the number of data point per class for test
# 7. create a seed list
# 8. randomly pick data from groups create train data
# 9. randomly pick data from groups create test data



# 1
total_size = len(X_n)
print("total data size: ",total_size)

# 2
num_classes = len(np.unique(y_n))
print("group data based on class: ", num_classes)

print('y_cts =', np.unique(y_n, return_counts=True))

group_dictionary = {k: [] for k in range(num_classes)}
print("group_dictionary ",group_dictionary)

for i in range(total_size):
    group_dictionary[y_n[i]].append(i)

for k in np.unique(y_n):
    print("number of data point in class ", k, " = ", len(group_dictionary[k]))

group_sizes = [len(group_dictionary[k]) for k in range(num_classes)]


import math

def get_draw_size(y_cts, dt, train_distr, test_distr, C=None):
    if len(train_distr) != len(test_distr):
        raise ValueError("training and test distributions are not the same length")

    if C is None:
        C = sum(y_cts)

    constraints = [C] + [y_cts[i] / (dt[0] * train_distr[i] + dt[1] * test_distr[i])
                         for i in range(len(y_cts))]

    return np.floor(min(constraints))

def synthetic_draw(n_y, n_classes, y_cts, y_idx, dt_distr, train_distr, test_distr, seed=4711):
    if len(train_distr) != len(test_distr):
        raise ValueError("training and test distributions are not the same length")

    if len(y_cts) != len(train_distr):
        raise ValueError("Length of training distribution does not match number of classes")

    if len(dt_distr) != 2:
        raise ValueError("Length of train/test-split has to equal 2")

    if not math.isclose(np.sum(dt_distr), 1.0):
        raise ValueError("Elements of train/test-split do not sum to 1")

    if not math.isclose(np.sum(train_distr), 1.0):
        raise ValueError("Elements of train distribution do not sum to 1")

    if not math.isclose(np.sum(test_distr), 1.0):
        raise ValueError("Elements of test distribution do not sum to 1")

    n = get_draw_size(y_cts, dt_distr, train_distr, test_distr, C=n_y)

    train_cts = (n * dt_distr[0] * train_distr).astype(int)
    if min(train_cts) == 0:
        raise ValueError("Under given input distributions a class would miss in training")

    test_cts = (n * dt_distr[1] * test_distr).astype(int)

    # fix seed for reproducibility
    np.random.seed(seed)

    train_list = [np.random.choice(y_idx[i], size=train_cts[i], replace=False) for i in range(n_classes)]
    y_idx = [np.setdiff1d(y_idx[i], train_list[i]) for i in range(n_classes)]
    test_list = [np.random.choice(y_idx[i], size=test_cts[i], replace=False) if np.size(y_idx[i]) > 0 else [] for i in
                 range(n_classes)]

    train_index = np.concatenate(train_list)
    test_index = np.concatenate(test_list).astype(int)

    n_train = train_index.shape[0]
    n_test = test_index.shape[0]
    M = n_train + n_test
    r_train = n_train * 1.0 / M
    r_test = n_test * 1.0 / M

    train_ratios = train_cts * 1.0 / n_train
    test_ratios = test_cts * 1.0 / n_test

    stats_vec = np.concatenate(
        [np.array([M, n_train, n_test, r_train, r_test]), train_cts, train_ratios, test_cts, test_ratios])

    return train_index, test_index, stats_vec        

seed = 42
n_classes = num_classes
N = 13611
y_cts = np.array([1322,  522, 1630, 3546, 1928, 2027, 2636])
Y = np.array([0,1,2,3,4,5,6])
y_idx = [np.where(y_n == l)[0] for l in Y]

train_test_ratios = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]
train_test_ratios = [np.array(d) for d in train_test_ratios]

dt_ratios = train_test_ratios

train_ds = np.array(train_prev_dictionary[num_classes])
test_ds = np.array(test_prev_dictionary[num_classes])

results = []

for dt_distr in dt_ratios:
    for train_distr in train_ds:
        for test_distr in test_ds:
            for seed in seed_list:
                for bins in bin_list:

                    # print('Training and Test dists:')
                    # print(dt_distr)
                    # print(train_distr)
                    # print(test_distr)

                    train_index, test_index, stats_vec = synthetic_draw(N, n_classes, y_cts, y_idx, dt_distr, train_distr, test_distr, seed)

                    X_train, y_train = X_n[train_index], y_n[train_index]
                    X_test, y_test = X_n[test_index], y_n[test_index]

                    input_dim = X_train.shape[1]
                    print('train size: ', X_train.shape,y_train.shape, ' test size: ', X_test.shape,y_test.shape)

                    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                    y_train_tensor = torch.tensor(y_train, dtype=torch.long)
                    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                    y_test_tensor = torch.tensor(y_test, dtype=torch.long)

                    trainval_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)


                    train_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True)
                    test_loader = DataLoader(test_dataset, batch_size=batch_size)

                    # model = SimpleMLP(input_dim, hidden_dim, num_classes).to(device)
                    model = Conv1DMLP(input_dim, num_classes).to(device)
                    # model = torch.nn.DataParallel(model, device_ids=[0, 1])

                    optimizer = optim.Adam(model.parameters(), lr=lr)
                    criterion = nn.CrossEntropyLoss(label_smoothing=0.0)

                    best_model = model
                    min_loss = np.inf
                    train_loss_ss = 1
                    epoch_ss = 1
                    # Training loop
                    for epoch in range(epochs):
                        model.train()
                        total_loss = 0
                        v_loss = 0

                        for inputs, labels in train_loader:
                            inputs, labels = inputs.to(device), labels.to(device)
                            optimizer.zero_grad()

                            logits, probs = model(inputs)
                            ce_loss = criterion(logits, labels)    #get the training loss


                            # Combine losses using learned weights
                            loss = ce_loss
                            loss.backward()
                            optimizer.step()

                            total_loss += loss.item()

                        avg_loss = total_loss / len(train_loader)

                        for inputs, labels in test_loader:
                            model.eval()
                            inputs, labels = inputs.to(device), labels.to(device)

                            logits, probs = model(inputs)
                            ce_loss = criterion(logits, labels)    #get the training loss

                            # Combine losses using learned weights
                            loss = ce_loss
                            v_loss += loss.item()

                        avg_vloss = v_loss / len(test_loader)

                        if avg_vloss<min_loss:
                            best_model = model
                            min_loss = avg_vloss
                            train_loss_ss = avg_loss
                            epoch_ss = epoch

                    print(f'Epoch {epoch_ss+1}/{epochs} | Loss: {train_loss_ss:.4f} | test Loss: {min_loss:.4f}')

                    model = best_model
                    # collecting trining data predistions after model is trained
                    all_labels = []
                    all_preds = []
                    all_probs = []

                    for inputs, labels in train_loader:
                        model.eval()
                        inputs, labels = inputs.to(device), labels.to(device)

                        logits, probs = model(inputs)
                        # ce_loss = criterion(logits, labels)    #get the training loss
                        # # Combine losses using learned weights
                        # loss = ce_loss

                        # Store labels and predictions
                        all_labels.append(labels.detach().cpu())
                        # predicted = torch.argmax(probs, dim=1)
                        max_probs, predicted = torch.max(probs, dim=1)
                        all_preds.append(predicted.detach().cpu())
                        all_probs.append(max_probs.detach().cpu())


                    # collecting testing data predistions after model is trained
                    all_labels_test = []
                    all_preds_test  = []
                    all_probs_test  = []

                    for inputs, labels in test_loader:
                        model.eval()
                        inputs, labels = inputs.to(device), labels.to(device)

                        logits, probs = model(inputs)
                        # ce_loss = criterion(logits, labels)    #get the training loss
                        # # Combine losses using learned weights
                        # loss = ce_loss

                        # Store labels and predictions
                        all_labels_test.append(labels.detach().cpu())
                        # predicted = torch.argmax(probs, dim=1)
                        max_probs, predicted = torch.max(probs, dim=1)
                        all_preds_test.append(predicted.detach().cpu())
                        all_probs_test.append(max_probs.detach().cpu())

                    # print('all labels: ', all_labels[:10])
                    # print('all preds: ', all_preds[:10])
                    norm_cm = edm.get_coeficient_matrix(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), num_classes)  
                    # print('norm_cm: ',norm_cm)
                    

                    # print('***********************************************************test set size: ', torch.cat(all_preds_test).numpy().shape)
                    initial_solution = edm.get_init_solution(norm_cm, torch.cat(all_preds_test).numpy(), num_classes)    
                    # print('initial solution: ',initial_solution)

                    # issues = edm.prob_examination(torch.cat(all_probs).numpy(), num_classes)
                    # print('issues: ', issues)

                    train_hist_dictionary, train_prob_dictionary = edm.get_train_distributions(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy(), num_classes, bins)
                    
                    # fig, axes = plt.subplots(num_classes, num_classes, figsize=(num_classes, num_classes))
                    # bin_edges = np.linspace(0, 1, bins + 1)  # assuming you know the range
                    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    # print(train_hist_dictionary[3][3])
                    # print(train_prob_dictionary[3][3])
                    # for p in range(num_classes):
                    #     for a in range(num_classes):
                    #         axes[p][a].bar(bin_centers, train_hist_dictionary[p][a], width=bin_edges[1] - bin_edges[0])
                    #         axes[p][a].set_xlim(0, 1)
                    #         axes[p][a].set_ylim(0, 1)
                    # plt.show()

                    test_hist_dictionary, test_prob_dictionary = edm.get_test_distributions(torch.cat(all_preds_test).numpy(), torch.cat(all_probs_test).numpy(), num_classes, bins)

                    # fig, axes = plt.subplots(1, num_classes, figsize=(3, num_classes))
                    # bin_edges = np.linspace(0, 1, bins + 1)  # assuming you know the range
                    # bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
                    # print(train_hist_dictionary[3])
                    # print(train_prob_dictionary[3])
                    # for p in range(num_classes):
                    #     axes[p].bar(bin_centers, test_hist_dictionary[p], width=bin_edges[1] - bin_edges[0])
                    #     axes[p].set_xlim(0, 1)
                    #     axes[p].set_ylim(0, 1)
                    # plt.show()

                    neighborhood_steps = np.array(list(edm.get_steps(num_classes)))
                    # print('shape of neighborhood: ', neighborhood_steps)

                    final_estimation = edm.get_estimation(train_hist_dictionary, test_hist_dictionary, num_classes, neighborhood_steps, initial_solution)
                    # print('final estimation: ', final_estimation)

                    act = edm.get_actual_count(torch.cat(all_labels_test).numpy(), num_classes)
                    prd = edm.get_actual_count(torch.cat(all_preds_test).numpy(), num_classes)
                    est = np.sum(final_estimation, axis=1)
                    ini = np.sum(initial_solution, axis=0)
                    absolute_error = edm.AE(act, est, num_classes)
                    nkld = edm.NKLD(act, est, num_classes)

                    print('act: ', act)
                    print('prd: ', prd)
                    print('ini: ', ini)
                    print('est: ', est)
                    print('mae: ',absolute_error)
                    print('nkld: ', nkld)

                    results.append([dt_distr, train_distr, test_distr, seed, bins, act, prd, est, absolute_error, nkld])

output_file_path = f'results2.csv'

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['dt_distr', 'train_distr','test_distr', 'seed', 'bins', 'act', 'prd', 'est','absolute_error','nkld'])  # Write header

    for row in results:
        csv_writer.writerow(row)