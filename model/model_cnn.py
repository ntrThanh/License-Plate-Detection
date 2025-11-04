from torch import nn

def make_blocks(input_channels, output_channels, use_dropout=False):
    layers = [
        nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channels),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2, stride=2),
    ]

    if use_dropout:
        layers.append(nn.Dropout(0.5))

    return nn.Sequential(*layers)


class ClassifierNumber(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        # 🔹 Sửa lại đầu vào thành 1 kênh (ảnh trắng đen)
        self.cnn1 = make_blocks(1, 64, use_dropout=True)
        self.cnn2 = make_blocks(64, 128, use_dropout=True)
        self.cnn3 = make_blocks(128, 256, use_dropout=True)
        self.cnn4 = make_blocks(256, 512, use_dropout=True)

        self.cnn5 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )

        self.flatten = nn.Flatten()
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.cnn1(x)
        x = self.cnn2(x)
        x = self.cnn3(x)
        x = self.cnn4(x)
        x = self.cnn5(x)
        x = self.flatten(x)
        x = self.fc(x)
        return x
