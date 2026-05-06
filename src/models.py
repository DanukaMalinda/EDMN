import torch.nn as nn
import torch.nn.functional as F

class adult_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(adult_model, self).__init__()
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

class adult_model__(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(adult_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.02)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class adult_model_(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(adult_model_, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 16)
        self.bn2 = nn.BatchNorm1d(16)
        self.fc3 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(0.01)  # Increased dropout slightly

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        logits = self.fc3(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class avila_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(avila_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 32)
        self.bn1 = nn.BatchNorm1d(32)
        self.fc2 = nn.Linear(32, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

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
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

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
    
class bc_cont_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(bc_cont_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class cars_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(cars_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

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
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

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
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class cappl_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(cappl_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

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
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    

class dota_model_(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(dota_model_, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.2)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class dota_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(dota_model, self).__init__()
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
    
class drugs_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(drugs_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class ener_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ener_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class fifa_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(fifa_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class flare_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(flare_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class grid_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(grid_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class ads_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(ads_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class magic_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(magic_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class boone_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(boone_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class vgame_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(vgame_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class mush_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(mush_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

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
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class thrm_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(thrm_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class telco_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(telco_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class cond_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(cond_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class study_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(study_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class spam_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(spam_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class craft_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(craft_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class alco_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(alco_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class phish_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(phish_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class musk_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(musk_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

    def forward(self, x, temperature=1.0):
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = F.relu(self.bn4(self.fc4(x)))

        logits = self.fc5(x)
        probs = F.softmax(logits / temperature, dim=1)
        return logits, probs
    
class music_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(music_model, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.fc3 = nn.Linear(64, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.fc4 = nn.Linear(16, 8)
        self.bn4 = nn.BatchNorm1d(8)
        self.fc5 = nn.Linear(8, num_classes)
        self.dropout = nn.Dropout(0.0000)

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
    
class occup_model(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(occup_model, self).__init__()
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
    


def getModel(name, input_dim, num_classes):
    if name == 'adult':
        print("loading Adult model")
        return adult_model(input_dim, num_classes)
    
    elif name == 'avila':
        return avila_model(input_dim, num_classes)
    
    elif name == 'alco':
        return alco_model(input_dim, num_classes)
    
    elif name == 'bike_b':
        return bike_model(input_dim, num_classes)
    
    elif name == 'blog_b':
        return blog_model(input_dim, num_classes)
    
    elif name == 'bc_cont':
        return bc_cont_model(input_dim, num_classes)
    
    elif name == 'cars_b':
        return cars_model(input_dim, num_classes)
    
    elif name == 'conc_b':
        return conc_model(input_dim, num_classes)
    
    elif name == 'contra_b':
        return contra_model(input_dim, num_classes)
    
    elif name == 'cappl':
        return cappl_model(input_dim, num_classes)
    
    elif name == 'diam_b':
        return diam_model(input_dim, num_classes)
    
    elif name == 'dota_b':
        return dota_model(input_dim, num_classes)
    
    elif name == 'drugs_b':
        return drugs_model(input_dim, num_classes)
    
    elif name == 'ener_b':
        return ener_model(input_dim, num_classes)
    
    elif name == 'fifa_b':
        return fifa_model(input_dim, num_classes)
    
    elif name == 'flare_b':
        return flare_model(input_dim, num_classes)
    
    elif name == 'grid_b':
        return grid_model(input_dim, num_classes)
    
    elif name == 'ads':
        return ads_model(input_dim, num_classes)
    
    elif name == 'magic':
        return magic_model(input_dim, num_classes)
    
    elif name == 'boone':
        return boone_model(input_dim, num_classes)
    
    elif name == 'mush':
        return mush_model(input_dim, num_classes)
    
    elif name == 'vgame_b':
        return vgame_model(input_dim, num_classes)
    
    elif name == 'turk_b':
        return turk_model(input_dim, num_classes)
    
    elif name == 'thrm_b':
        return thrm_model(input_dim, num_classes)
    
    elif name == 'telco_b':
        return telco_model(input_dim, num_classes)
    
    elif name == 'cond_b':
        return cond_model(input_dim, num_classes)
    
    elif name == 'study_b':
        return study_model(input_dim, num_classes)
    
    elif name == 'spam_b':
        return spam_model(input_dim, num_classes)
    
    elif name == 'craft_b':
        return craft_model(input_dim, num_classes)
    
    elif name == 'phish':
        return phish_model(input_dim, num_classes)
    
    elif name == 'musk':
        return musk_model(input_dim, num_classes)
    
    elif name == 'music':
        return music_model(input_dim, num_classes)
    
    elif name == 'wine_b':
        return wine_model(input_dim, num_classes)
    
    elif name == 'occup':
        return occup_model(input_dim, num_classes)
    
    else:
        return diam_model(input_dim, num_classes)