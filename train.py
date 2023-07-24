import torch
from tqdm import tqdm
from torch import nn, optim

BS = 4
LR = 0.001
N_EPOCHS = 25
N_CLASS = 7

def get_dataloader(dataset, train=True):
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [0.8, 0.2])

    train_loader = torch.utils.data.DataLoader(
        dataset = train_dataset,
        batch_size = BS,
        shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        dataset = test_dataset,
        batch_size = BS
    )
    return train_loader if train else test_loader

def run_train_loop(model, dataset, device):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=LR)
    train_loader = get_dataloader(dataset, train=True)

    model.train()
    for e in range(N_EPOCHS):
        print(f"Epoch {e+1}/{N_EPOCHS}")
        for image, label in tqdm(train_loader):
            image = image.to(device)
            label = label.to(device)
            # forward
            outputs = model(image)
            loss = criterion(outputs, label)
            # backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Loss at the end of epoch: {loss.item()}")

def run_eval_loop(model, dataset, device):

    test_loader = get_dataloader(dataset, train=False)
    with torch.no_grad():
        n_correct = 0
        n_samples = 0
        n_class_correct = N_CLASS*[0]
        n_class_samples = N_CLASS*[0]
        for images, labels in tqdm(test_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            
            # values, indexes
            _, predictions = torch.max(outputs, 1)
            n_samples += labels.size(0)
            n_correct += (predictions == labels).sum().item()
            
            for i in range(4):
                if i < len(labels):
                    label = labels[i]
                    pred = predictions[i]
                    if(label == pred):
                        n_class_correct[label]+=1
                    n_class_samples[label] += 1
                    
    classes = list(dataset.label_encoder.keys())

    acc = 100.0 * n_correct/n_samples
    print(f'Accuracy = {acc}')

    for i in range(N_CLASS):
        acc = 100.0*n_class_correct[i]/n_class_samples[i]
        print(f'Accuracy of {classes[i]}: {acc}')
