import os

import tqdm

import torch.utils.data
from sklearn.metrics import accuracy_score
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from model.model_cnn import ClassifierNumber


def train(epochs = 30):
    print('Training classifier characters model')
    path_to_dataset = './dataset/CNN letter Dataset/'
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])

    dataset = ImageFolder(root=path_to_dataset, transform=transform)

    # split train and test data
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size

    # create train, test dataset, and loader data
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=True, drop_last=True, num_workers=4)

    # create model
    model = ClassifierNumber(num_classes=len(dataset.classes))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # handle checkpoint
    check_point_path = './checkpoint'
    os.makedirs(check_point_path, exist_ok=True)

    last_check_point = os.path.join(check_point_path, 'last_checkpoint.pth')
    best_model_path = os.path.join(check_point_path, 'best_character_classification.pth')

    start_epoch = 0
    best_accuracy = 0


    # handle resume training
    if os.path.exists(last_check_point):
        print(f'[INFO] Found checkpoint at {last_check_point}. Resuming training from checkpoint.]')
        checkpoint = torch.load(last_check_point)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_accuracy = checkpoint['best_accuracy']
        print(f'[INFO] Resuming from epoch {start_epoch} and best accuracy {best_accuracy}.')

    for epoch in range(start_epoch, epochs):
        model.train()

        progress = tqdm.tqdm(train_loader)
        for iteration, (image, label) in enumerate(progress):
            output = model(image)
            loss = criterion(output, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            progress.set_description(f'[INFO]: Epoch: {epoch} / {epochs} Iteration: {iteration}/{len(train_loader)} Loss: {loss.item():.4f}')

        model.eval()

        all_predictions = []
        all_labels = []

        with torch.no_grad():
            progress = tqdm.tqdm(test_loader)
            for iteration, (image, label) in enumerate(progress):
                output = model(image)
                loss = criterion(output, label)
                preds = torch.argmax(output, dim=1)

                all_predictions.extend(preds.cpu().numpy())
                all_labels.extend(label.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_predictions)
        print(f'[INFO]: Epoch: {epoch} / {epochs} Accuracy: {accuracy:.4f}')

        checkpoint = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_accuracy': best_accuracy,
            'best_model_state_dict': model.state_dict(),
        }

        torch.save(checkpoint, last_check_point)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(checkpoint, best_model_path)
            print(f'[INFO] Saving best model at epoch {epoch} and accuracy {best_accuracy}.]')


if __name__ == '__main__':
    train()