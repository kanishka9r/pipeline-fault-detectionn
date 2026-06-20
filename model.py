import torch.nn as nn

# Define the CNN model for classification
class CNNClassifier(nn.Module):
    def __init__(self,  num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(2, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),

            nn.Conv1d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),

            nn.Conv1d(128, 256, kernel_size=3, padding=1), 
            nn.BatchNorm1d(256),
            nn.ReLU(),

            nn.AdaptiveAvgPool1d(1) ) # take avg of each channel on time (batch size , 256 , 1) 
        
        self.classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.4), 
            nn.Linear(64, num_classes))

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.features(x)
        x = x.squeeze(-1) # (batch size , 256)
        return self.classifier(x)