import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import data_loader
import edm
import matplotlib.pyplot as plt
import csv
import argparse
import multiclass_models
from data_prevalence import synthetic_draw

# Reproducibility
torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

parser = argparse.ArgumentParser(description="Example script with argparse")

# Add arguments
parser.add_argument('--dataset', type=str, required=True, help='Dataset name')
parser.add_argument('--load_model', type=bool, default=False, help='Load Initial weights')
parser.add_argument('--distance_metric', type=str, default='JD', help='Distance Metric')
parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
parser.add_argument('--bins', type=int, default=100, help='Number of bins in histogram')
parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning Rate')

args = parser.parse_args()


# Hyperparameters
hidden_dim = 64
epochs = args.epochs
batch_size = args.batch_size
lr = args.learning_rate
bins = args.bins
lambda_dist = 1.0  # Weight for distribution matching loss
dm = args.distance_metric

# Load dataset
X_n, y_n = data_loader.load_data(args.dataset)
print('X_n shape', X_n.shape, 'y_n shape', y_n.shape)

# split ratios
test_size_list = [0.3,0.5,0.7,0.9]

# train prevelances
train_2_prevalences = [[0.05  ,0.95],
                       [0.1   ,0.9],
                       [0.3   ,0.7],
                       [0.5   ,0.5],
                       [0.7   ,0.3],
                       [0.9   ,0.1]]

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

train_prev_dictionary = {2:train_2_prevalences,
                         3:train_3_prevalences, 
                         4:train_4_prevalences,
                         5:train_5_prevalences,
                         6:train_6_prevalences,
                         7:train_7_prevalences,
                         10:train_10_prevalences}
# test prevelances
test_2_prevalences = [ [0.1   ,0.9],
                       [0.2   ,0.8],
                       [0.3   ,0.7],
                       [0.4   ,0.6],
                       [0.5   ,0.5],
                       [0.6   ,0.4],
                       [0.7   ,0.3],
                       [0.8   ,0.2],
                       [0.9   ,0.1]]

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


test_prev_dictionary = { 2:test_2_prevalences,
                         3:test_3_prevalences, 
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
    

Y = np.unique(y_n)
print('******************************************************************************************* Y: ', Y)
# Y = np.array([0,1,2,3,4,5,6])
y_idx = [np.where(y_n == l)[0] for l in Y]

# 1
total_size = len(X_n)
print("total data size: ",total_size)

# 2
num_classes = len(np.unique(y_n))
print("group data based on class: ", num_classes)

print('y_cts =', np.unique(y_n, return_counts=True))

group_dictionary = {k: [] for k in Y}
print("group_dictionary ",group_dictionary)

for i in range(total_size):
    group_dictionary[y_n[i]].append(i)

for k in np.unique(y_n):
    print("number of data point in class ", k, " = ", len(group_dictionary[k]))

group_sizes = [len(group_dictionary[k]) for k in Y] 

seed = 42
n_classes = num_classes
N = len(X_n)
y_cts = np.array(group_sizes)


train_test_ratios = [[0.1, 0.9], [0.3, 0.7], [0.5, 0.5], [0.7, 0.3]]
train_test_ratios = [np.array(d) for d in train_test_ratios]

dt_ratios = train_test_ratios



train_ds = np.array(train_prev_dictionary[num_classes])
test_ds = np.array(test_prev_dictionary[num_classes])

results = []

distance_metrics = ["SE", "MH", "PS", "TS", "JD", "TN", "DC", "JC", "CB", "IP", "HB", "CS", "HM", "HD"]

# for dist_metric in distance_metrics:

output_file_path = f'../results/results_edm_{args.dataset}_{args.distance_metric}.csv'
output_file_path = f'../results/results_edm_{args.dataset}_all_metrics.csv'

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['dt_distr', 'train_distr','test_distr', 'seed', 'bins', 'act', 'prd', 'est','absolute_error','nkld', 'acc','dm'])  # Write header


avg_AE = 0
avg_NKLD = 0
simulations = 0

avg_AE_acc = 0
avg_NKLD_acc = 0
simulations_acc = 0

neighborhood_steps = np.array(list(edm.get_steps(num_classes)))
print('neighborhood_steps_init len: ', len(neighborhood_steps))

for i in range(num_classes):
    additional_steps = edm.get_additional_neighbors(num_classes, step=(i+2))
    neighborhood_steps = np.vstack([neighborhood_steps, additional_steps])
# print('shape of neighborhood: ', neighborhood_steps)
print('neighborhood_steps len: ', len(neighborhood_steps))

for dt_distr in dt_ratios:
    for train_distr in train_ds:
        for test_distr in test_ds:
            for seed in seed_list:
                for bins in bin_list:

                    try:

                        # print('Training and Test dists:')
                        # print(dt_distr)
                        # print(train_distr)
                        # print(test_distr)

                        train_index, test_index, stats_vec = synthetic_draw(N, n_classes, y_cts, y_idx, dt_distr, train_distr, test_distr, seed)

                        X_train, y_train = X_n[train_index], y_n[train_index]
                        X_test, y_test = X_n[test_index], y_n[test_index]

                        # Get validation indices (rest of the data not in train or test)
                        all_indices = np.arange(len(X_n)) 
                        val_index = np.setdiff1d(all_indices, np.union1d(train_index, test_index))

                        # Create validation sets
                        X_val, y_val = X_n[val_index], y_n[val_index]

                        input_dim = X_train.shape[1]
                        print('train size: ', X_train.shape, ' test size: ', X_test.shape, ' seed: ', seed)

                        X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
                        y_train_tensor = torch.tensor(y_train, dtype=torch.long)
                        X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
                        y_test_tensor = torch.tensor(y_test, dtype=torch.long)
                        X_val_tensor = torch.tensor(X_val, dtype=torch.float32)
                        y_val_tensor = torch.tensor(y_val, dtype=torch.long)

                        trainval_dataset = TensorDataset(X_train_tensor, y_train_tensor)
                        test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
                        val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

                        train_loader = DataLoader(trainval_dataset, batch_size=batch_size, shuffle=True)
                        test_loader = DataLoader(test_dataset, batch_size=batch_size)
                        val_loader = DataLoader(val_dataset, batch_size=batch_size)

                        model = multiclass_models.getModel(args.dataset, input_dim, num_classes).to(device) 
                        if args.load_model:
                            model.load_state_dict(torch.load(f'../models/{args.dataset}_multiclass_model_weights.pth', map_location=device))  

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

                            for inputs, labels in val_loader:
                                model.eval()
                                inputs, labels = inputs.to(device), labels.to(device)

                                logits, probs = model(inputs)
                                ce_loss = criterion(logits, labels)    #get the validation loss

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

                        model.eval()
                        with torch.no_grad():
                            for inputs, labels in train_loader:
                                
                                inputs, labels = inputs.to(device), labels.to(device)

                                logits, probs = model(inputs)

                                # Store labels and predictions
                                all_labels.append(labels.detach().cpu())
                                max_probs, predicted = torch.max(probs, dim=1)
                                all_preds.append(predicted.detach().cpu())
                                all_probs.append(max_probs.detach().cpu())


                        # collecting testing data predistions after model is trained
                        all_labels_test = []
                        all_preds_test  = []
                        all_probs_test  = []

                        model.eval()
                        with torch.no_grad():
                            for inputs, labels in test_loader:
                                
                                inputs, labels = inputs.to(device), labels.to(device)
                                logits, probs = model(inputs)

                                # Store labels and predictions
                                all_labels_test.append(labels.detach().cpu())
                                max_probs, predicted = torch.max(probs, dim=1)
                                all_preds_test.append(predicted.detach().cpu())
                                all_probs_test.append(max_probs.detach().cpu())

                        acc = edm.getAccuracy(torch.cat(all_labels_test).numpy(), torch.cat(all_preds_test).numpy())

                        norm_cm = edm.get_coeficient_matrix(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), num_classes)  
                    
                        initial_solution, _ = edm.get_init_solution(norm_cm, torch.cat(all_preds_test).numpy(), num_classes)    

                        train_hist_dictionary, train_prob_dictionary = edm.get_train_distributions(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy(), num_classes, bins)

                        test_hist_dictionary, test_prob_dictionary = edm.get_test_distributions(torch.cat(all_preds_test).numpy(), torch.cat(all_probs_test).numpy(), num_classes, bins)

                        for dm in distance_metrics:
                            print('Distance metric: ',dm)
                            final_estimation = edm.get_estimation(train_hist_dictionary, test_hist_dictionary, num_classes, neighborhood_steps, initial_solution, dm)

                            act = edm.get_actual_count(torch.cat(all_labels_test).numpy(), num_classes)
                            prd = edm.get_actual_count(torch.cat(all_preds_test).numpy(), num_classes)
                            est = np.sum(final_estimation, axis=0)
                            ini = np.sum(initial_solution, axis=1)
                            absolute_error = edm.AE(act, est, num_classes)
                            nkld = edm.NKLD(act, est, num_classes)

                            print('act: ', act)
                            print('prd: ', prd)
                            print('ini: ', ini)
                            print('mae: ',absolute_error)
                            print('nkld: ', nkld)
                            print('model accuracy:',acc)

                            avg_AE += absolute_error
                            avg_NKLD += nkld
                            simulations += 1
                            

                            if acc>55:
                                avg_AE_acc += absolute_error
                                avg_NKLD_acc += nkld
                                simulations_acc += 1

                            print(args.dataset, ' MAE: ',avg_AE/simulations, ' NKLD: ', avg_NKLD/simulations, ' MAE acc: ',avg_AE_acc/simulations_acc, ' NKLD acc: ', avg_NKLD_acc/simulations_acc)

                            result = [dt_distr, train_distr, test_distr, seed, bins, act, prd, est, absolute_error, nkld, acc, dm]
                            results.append([dt_distr, train_distr, test_distr, seed, bins, act, prd, est, absolute_error, nkld, acc, dm])
                            
                            with open(output_file_path, 'a', newline='') as csvfile:
                                csv_writer = csv.writer(csvfile)
                                csv_writer.writerow(result)

                    except Exception as e:
                        print(f"[dt_distr {dt_distr}, train_distr {train_distr}, test_distr {test_distr}, seed {seed}, bins {bins}] Error occurred: {e}")
                        continue  # Skip to next experiment

output_file_path = f'../results/results_edm_{args.dataset}_all_metrics_final.csv'

with open(output_file_path, 'w', newline='') as csvfile:
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(['dt_distr', 'train_distr','test_distr', 'seed', 'bins', 'act', 'prd', 'est','absolute_error','nkld', 'acc','dm'])  # Write header

    for row in results:
        csv_writer.writerow(row)