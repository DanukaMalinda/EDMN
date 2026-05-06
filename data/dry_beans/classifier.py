import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import KFold, train_test_split
from torch.utils.data import TensorDataset, DataLoader, Subset
import numpy as np
import prep as dtpr

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

dataset = dtpr.prep_data(True)
print(dataset.head())


# dataset
X_n = dataset.iloc[:, :-1].values.astype(np.float32)
y_n = dataset.iloc[:, -1].values.astype(np.int64)


# Hyperparameters
num_classes = len(np.unique(y_n))
input_dim = X_n.shape[1]
train_size = int(X_n.shape[0]*0.75)
hidden_dim = 64
epochs = 10
batch_size = 128
lr = 0.001
k_folds = 5
bins = 20
lambda_dist = 1.0  # Weight for distribution matching loss


# Split into train+val and test
X = torch.from_numpy(X_n) 
y = torch.from_numpy(y_n)

X_trainval, X_test, y_trainval, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

trainval_dataset = TensorDataset(X_trainval, y_trainval)
test_dataset = TensorDataset(X_test, y_test)

# Model definition
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
# Loss weight learning
class LossWeightLearner(nn.Module):
    def __init__(self):
        super(LossWeightLearner, self).__init__()
        self.log_sigma1 = nn.Parameter(torch.zeros(1))  # for loss1
        self.log_sigma2 = nn.Parameter(torch.zeros(1))  # for loss2

    def forward(self, loss1, loss2):
        weighted_loss = (torch.exp(-self.log_sigma1) * loss1 + self.log_sigma1) + \
                        (torch.exp(-self.log_sigma2) * loss2 + self.log_sigma2)
        return weighted_loss

# Full model wrapper
class FullModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_classes):
        super(FullModel, self).__init__()
        self.mlp = SimpleMLP(input_dim, hidden_dim, num_classes)
        self.loss_weights = LossWeightLearner()

    def forward(self, x, temperature=1.0):
        return self.mlp(x, temperature)
    


# Hellinger distance
def hellinger_distance(p, q):
    return torch.sqrt(0.5 * torch.sum((torch.sqrt(p) - torch.sqrt(q)) ** 2))

# Distribution matching loss
def distribution_matching_loss(val_histograms, train_histograms, confusion_weights, num_classes):
    distances = []
    for pred in range(num_classes):
        combined_train_hist = torch.zeros_like(next(iter(train_histograms.values())))
        for actual in range(num_classes):
            weight = confusion_weights[actual, pred]
            combined_train_hist += weight * train_histograms[(actual, pred)]

        # Normalize combined_train_hist so it becomes a proper histogram
        combined_train_hist /= (combined_train_hist.sum() + 1e-8)

        # Normalize validation histogram for fair distance computation
        val_hist = val_histograms[pred] / (val_histograms[pred].sum() + 1e-8)

        dist = hellinger_distance(val_hist, combined_train_hist)
        distances.append(dist)

    return torch.max(torch.stack(distances))

def collect_train_histograms(model, dataloader, num_classes, bins):
    model.eval()
    histograms = {(pred, actual): torch.zeros(bins, device=device)
                  for pred in range(num_classes)
                  for actual in range(num_classes)}

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, probs = model(inputs)

            for i in range(probs.size(0)):
                prob_vector = probs[i]
                pred_class = torch.argmax(prob_vector).item()
                actual_class = labels[i].item()

                prob_value = prob_vector[pred_class].item()  # or prob_vector[actual_class].item() depending on what you want
                bin_idx = min(int(prob_value * bins), bins - 1)
                histograms[(pred_class, actual_class)][bin_idx] += 1

    # Normalize histograms
    for key in histograms:
        histograms[key] /= (histograms[key].sum() + 1e-8)

    return histograms

def collect_val_histograms(probs, num_classes, bins):
    histograms = {pred: torch.zeros(bins, device=probs.device) for pred in range(num_classes)}
    for i in range(probs.size(0)):
        prob_vector = probs[i]
        pred_class = torch.argmax(prob_vector).item()
        prob_value = prob_vector[pred_class].item()
        bin_idx = min(int(prob_value * bins), bins - 1)
        histograms[pred_class][bin_idx] += 1
    return histograms  # <- no normalization

def compute_confusion_weights(model, dataloader, num_classes):
    model.eval()
    conf_matrix = torch.zeros((num_classes, num_classes), device=device)
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, probs = model(inputs)
            preds = torch.argmax(probs, dim=1)
            for i in range(labels.size(0)):
                actual = labels[i].item()
                predicted = preds[i].item()
                conf_matrix[actual, predicted] += 1

    return conf_matrix  # <- raw integer counts



# K-Fold Cross-Validation
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

while(True):
    if(batch_size<train_size):
        for fold, (train_ids, val_ids) in enumerate(kfold.split(trainval_dataset)):
            print(f'FOLD {fold + 1}')
            
            train_subsampler = Subset(trainval_dataset, train_ids)
            val_subsampler = Subset(trainval_dataset, val_ids)

            train_loader = DataLoader(train_subsampler, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_subsampler, batch_size=batch_size)

            model = FullModel(input_dim, hidden_dim, num_classes).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

            # Training loop
            for epoch in range(epochs):
                model.train()
                total_loss = 0

                for inputs, labels in train_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()

                    logits, probs = model(inputs)
                    ce_loss = criterion(logits, labels)

                    # Distribution matching loss on validation data (use small val batch for efficiency)
                    model.eval()
                    val_inputs, _ = next(iter(val_loader))
                    val_inputs = val_inputs.to(device)
                    with torch.no_grad():
                        val_logits, val_probs = model(val_inputs)
                    
                    train_histograms = collect_train_histograms(model, train_loader, num_classes, bins)
                    # print(train_histograms)
                    conf_matrix = compute_confusion_weights(model, train_loader, num_classes)
                    print(conf_matrix)

                    val_histograms = collect_val_histograms(val_probs, num_classes, bins)
                    dist_loss = distribution_matching_loss(val_histograms, train_histograms, conf_matrix, num_classes)

                    model.train()


                    # Combine losses using learned weights
                    total_loss = model.loss_weights(ce_loss, dist_loss)

                    loss = total_loss
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()

                avg_loss = total_loss / len(train_loader)
                print(f'Epoch {epoch+1}/{epochs} | Loss: {avg_loss.item():.4f}')
        batch_size = batch_size*2
        
    else:
        break

# Final test set inference (optional)
model.eval()
test_loader = DataLoader(test_dataset, batch_size=batch_size)
all_probs, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs = inputs.to(device)
        logits, probs = model(inputs)
        all_probs.append(probs.cpu())
        all_labels.append(labels)

all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)

print("Test Set Probabilities Shape:", all_probs.shape)
print("Test Set Labels Shape:", all_labels.shape)
print(all_probs[:5])
print(all_labels[:5])

# Save test set probabilities and labels for later use
# np.save("test_probs.npy", all_probs.numpy())
# np.save("test_labels.npy", all_labels.numpy())



# saving training set predictions and targets to get histograms
train_loader = DataLoader(trainval_dataset, batch_size=batch_size)
all_probs, all_labels = [], []

with torch.no_grad():
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        logits, probs = model(inputs)
        all_probs.append(probs.cpu())
        all_labels.append(labels)

all_probs = torch.cat(all_probs)
all_labels = torch.cat(all_labels)

# Save test set probabilities and labels for later use
# np.save("train_probs.npy", all_probs.numpy())
# np.save("train_labels.npy", all_labels.numpy())
