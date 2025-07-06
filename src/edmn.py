import edm
import torch
import numpy as np

def get_confusion_matrix(y_train, y_train_pred, num_classes):
    cm = np.zeros((num_classes, num_classes))  # intiate coeficient matrix
    for i in range(len(y_train)):
        cm[y_train_pred[i]][y_train[i]] += 1   # increment the matrix (pred, actual)
    
    return cm

def train_edm(train_loader, val_loader, model, device, num_classes, bins, dm):
    all_labels = []
    all_preds = []
    all_probs = []

    model.eval()
    with torch.no_grad():
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

    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            logits, probs = model(inputs)

            # Store labels and predictions
            all_labels_test.append(labels.detach().cpu())
            # predicted = torch.argmax(probs, dim=1)
            max_probs, predicted = torch.max(probs, dim=1)
            all_preds_test.append(predicted.detach().cpu())
            all_probs_test.append(max_probs.detach().cpu())

    
    train_hist_dictionary, _ = edm.get_train_distributions(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(), torch.cat(all_probs).numpy(), num_classes, bins)
    test_hist_dictionary, _ = edm.get_test_distributions(torch.cat(all_preds_test).numpy(), torch.cat(all_probs_test).numpy(), num_classes, bins)
    cm = get_confusion_matrix(torch.cat(all_labels).numpy(), torch.cat(all_preds).numpy(),num_classes)

    edm_cost = 0
    for i in range(num_classes):
        est_hist = edm.make_distributions(train_hist_dictionary[i], cm[i], num_classes)
        dist = edm.get_distance(est_hist, test_hist_dictionary[i], dm)
        edm_cost += dist

    return edm_cost


