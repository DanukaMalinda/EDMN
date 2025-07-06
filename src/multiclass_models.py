import torch.nn as nn
import torch.nn.functional as F



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
    
class yeast_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(yeast_model, self).__init__()
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
    
class drugs_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(drugs_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 1024)
        self.bn2 = nn.BatchNorm1d(1024)
        self.fc3 = nn.Linear(1024, 256)
        self.bn3 = nn.BatchNorm1d(256)
        self.fc4 = nn.Linear(256, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 8)
        self.bn6 = nn.BatchNorm1d(8)
        self.fc7 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.001)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        x = self.dropout(F.relu(self.bn5(self.fc5(x))))
        x = F.relu(self.bn6(self.fc6(x)))

        logits = self.fc7(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    

class vgame_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(vgame_model, self).__init__()
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

class wine_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(wine_model, self).__init__()
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
    
class turk_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(turk_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    

class beans_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(beans_model, self).__init__()
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



class bike_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(bike_model, self).__init__()
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
    
class blog_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(blog_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 96)
        self.bn2 = nn.BatchNorm1d(96)
        self.fc3 = nn.Linear(96, 32)
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
    

class conc_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(conc_model, self).__init__()
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

class contra_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(contra_model, self).__init__()
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
    
class diam_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(diam_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 64)
        self.bn3 = nn.BatchNorm1d(64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.BatchNorm1d(32)
        self.fc5 = nn.Linear(32, 16)
        self.bn5 = nn.BatchNorm1d(16)
        self.fc6 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.001)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))

        logits = self.fc6(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class ener_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ener_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 16)
        self.bn1 = nn.BatchNorm1d(16)
        self.fc2 = nn.Linear(16, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 48)
        self.bn3 = nn.BatchNorm1d(48)
        self.fc4 = nn.Linear(48, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 16)
        self.bn6 = nn.BatchNorm1d(16)
        self.fc7 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.001)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))

        logits = self.fc7(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    

class fifa_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(fifa_model, self).__init__()
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
    
class gasd_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(gasd_model, self).__init__()
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
    

class gest_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(gest_model, self).__init__()
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

class hars_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(hars_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 16)
        self.bn6 = nn.BatchNorm1d(16)
        self.fc7 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.001)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))

        logits = self.fc7(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs

class micro_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(micro_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 250)
        self.bn2 = nn.BatchNorm1d(250)
        self.fc3 = nn.Linear(250, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.fc5 = nn.Linear(64, 32)
        self.bn5 = nn.BatchNorm1d(32)
        self.fc6 = nn.Linear(32, 16)
        self.bn6 = nn.BatchNorm1d(16)
        self.fc7 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.001)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))
        x = F.relu(self.bn5(self.fc5(x)))
        x = F.relu(self.bn6(self.fc6(x)))

        logits = self.fc7(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs

class news_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(news_model, self).__init__()
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



class nurse_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(nurse_model, self).__init__()
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
    
class optd_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(optd_model, self).__init__()
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
    

class pend_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(pend_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.fc4 = nn.Linear(32, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.01)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs

    
class rice_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(rice_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.01)

    def forward(self, x, temperature=1.0):
        x = F.leaky_relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.leaky_relu(self.bn2(self.fc2(x))))


        logits = self.fc3(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class thrm_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(thrm_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 96)
        self.bn3 = nn.BatchNorm1d(96)
        self.fc4 = nn.Linear(96, 16)
        self.bn4 = nn.BatchNorm1d(16)
        self.fc5 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    

def getModel(name, input_dim, num_classes):
    if name == 'yeast':
        return yeast_model(input_dim, num_classes)
    
    elif name == 'drugs':
        return drugs_model(input_dim, num_classes)
    
    elif name == 'vgame':
        return vgame_model(input_dim, num_classes)
    
    elif name == 'wine':
        return wine_model(input_dim, num_classes)
    
    elif name == 'turk':
        return turk_model(input_dim, num_classes)
    
    elif name == 'beans':
        return beans_model(input_dim, num_classes)
    
    elif name == 'bike':
        return bike_model(input_dim, num_classes)
    
    elif name == 'blog':
        return blog_model(input_dim, num_classes)
    
    elif name == 'conc':
        return conc_model(input_dim, num_classes)
    
    elif name == 'contra':
        return contra_model(input_dim, num_classes)
    
    elif name == 'diam':
        return diam_model(input_dim, num_classes)
    
    elif name == 'ener':
        return ener_model(input_dim, num_classes)
    
    elif name == 'fifa':
        return fifa_model(input_dim, num_classes)
    
    elif name == 'gasd':
        return gasd_model(input_dim, num_classes)
    
    elif name == 'gest':
        return gest_model(input_dim, num_classes)
    
    elif name == 'hars':
        return hars_model(input_dim, num_classes)
    
    elif name == 'micro':
        return micro_model(input_dim, num_classes)
    
    elif name == 'news':
        return news_model(input_dim, num_classes)
    
    elif name == 'nurse':
        return nurse_model(input_dim, num_classes)
    
    elif name == 'optd':
        return optd_model(input_dim, num_classes)
    
    elif name == 'pend':
        return pend_model(input_dim, num_classes)
    
    elif name == 'rice':
        return rice_model(input_dim, num_classes)
    
    elif name == 'thrm':
        return thrm_model(input_dim, num_classes)
    
    else:
        return yeast_model(input_dim, num_classes)
    

