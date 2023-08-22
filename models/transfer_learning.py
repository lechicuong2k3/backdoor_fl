import torchvision.models as models
import torch.nn as nn
import torch

def weights_init(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight.data)
        
class Head(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(Head, self).__init__()
        self.fc1 = nn.Linear(in_channels, 512)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x
    
class VGG16(nn.Module):
    def __init__(self, num_classes):
        super(VGG16, self).__init__()
        # Load pre-trained VGG16 model
        self.base_model = models.vgg16(weights="DEFAULT")
        for param in self.base_model.parameters():
            param.requires_grad = False
        for param in self.base_model.classifier.parameters():
            param.requires_grad = False
        self.base_model.classifier[6] = nn.Linear(in_features=4096, out_features=10)
    
    def __str__(self):
        return "VGG16"
        
    def forward(self, x):
        x = self.base_model(x)
        return x
    
    def reset_weights(self):
        self.base_model.classifier.apply(weights_init)

class DenseNet(nn.Module):
    def __init__(self, num_classes):
        super(DenseNet, self).__init__()
        # Load pre-trained DenseNet121 model
        self.base_model = models.densenet121(weights="DEFAULT")
        for param in self.base_model.parameters():
            param.requires_grad = False
        num_ftrs = self.base_model.classifier.in_features
        self.base_model.classifier = nn.Identity()
        self.head_model = Head(in_channels=num_ftrs, num_classes=num_classes) 
    
    def __str__(self):
        return "DenseNet121"
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.head_model(x)
        return x
    
    def reset_weights(self):
        self.head_model.apply(weights_init)
        
class ResNet(nn.Module):
    def __init__(self, num_classes):
        super(ResNet, self).__init__()
        # Load pre-trained ResNet50 model
        self.base_model = models.resnet50(weights="DEFAULT")
        for param in self.base_model.parameters():
            param.requires_grad = False   
        num_ftrs = self.base_model.fc.in_features
        self.base_model.fc = nn.Identity()
        self.head_model = Head(in_channels=num_ftrs, num_classes=num_classes) 
    
    def __str__(self):
        return "ResNet50"
        
    def forward(self, x):
        x = self.base_model(x)
        x = self.head_model(x)
        return x
    
    def reset_weights(self):
        self.head_model.apply(weights_init)