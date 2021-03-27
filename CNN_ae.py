import numpy as np
import random

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


SEQUENCE = 50
N_FEATURES = 5


class AutoEncoder(nn.Module):

    def __init__(self, code_size):
        super().__init__()
        self.code_size = code_size

        self.enc_cnn_1 = nn.Conv1d(5, 10, kernel_size=3)
        self.enc_cnn_2 = nn.Conv1d(10, 20, kernel_size=3)
        self.enc_linear_1 = nn.Linear(920, 50)
        self.enc_linear_2 = nn.Linear(50, self.code_size)

        self.dec_linear_1 = nn.Linear(self.code_size, 160)
        self.dec_linear_2 = nn.Linear(160, SEQUENCE * N_FEATURES)

    def forward(self, sample):
        code = self.encode(sample)
        out = self.decode(code)
        return out, code

    def encode(self, sample):
        code = self.enc_cnn_1(sample)
        code = F.selu(code)
        code = self.enc_cnn_2(code)
        code = F.selu(code)
        code = code.view([sample.size(0), -1])
        code = F.selu(code)
        code = self.enc_linear_1(code)
        code = self.enc_linear_2(code)
        return code

    def decode(self, code):
        out = F.selu(self.dec_linear_1(code))
        out = torch.sigmoid(self.dec_linear_2(out))
        out = out.view([code.size(0), N_FEATURES, SEQUENCE])
        return out


class AssetDataset(torch.utils.data.Dataset):
    """Face Landmarks dataset."""

    def __init__(self, data, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = np.swapaxes(data, 1, 2)
        self.transform = transform

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = self.data[idx].astype('float64')

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    # [ time, low, high, open, close, volume ]
    data = np.genfromtxt('eggs.csv', delimiter=',')
    data = np.flip(data, axis=0)
    data = data[:, 1:]
    # normalize
    min_points, max_points = np.min(data, axis=0), np.max(data, axis=0)
    data = (data - min_points) / (max_points - min_points)
    # split into equal sequences
    # total_rows = data.shape[0]
    # PARTS = int(total_rows / SEQUENCE)
    # missing = (data.shape[0]) % PARTS
    # splits = np.array(np.array_split(data[missing:], PARTS))
    
    # window
    splits = np.array([np.array(data[i:i+SEQUENCE])
                       for i in range(len(data)-SEQUENCE)])

    # Hyperparameters
    code_size = 20
    num_epochs = 10
    batch_size = 128
    lr = 0.001
    optimizer_cls = optim.Adam

    dataset = AssetDataset(splits)
    train_loader = torch.utils.data.DataLoader(
        dataset, shuffle=True, batch_size=batch_size, num_workers=8)

    autoencoder = AutoEncoder(code_size)
    loss_fn = nn.MSELoss(reduction='sum')
    optimizer = optimizer_cls(autoencoder.parameters(), lr=lr)

    autoencoder = torch.load('model.pt')  # autoencoder.load('model.pt')
    autoencoder.train()

    for epoch in range(num_epochs):
        print("Epoch %d" % epoch)

        for i, batch in enumerate(train_loader):
            batch = batch.float()
            out, code = autoencoder(batch)

            optimizer.zero_grad()
            loss = loss_fn(out, batch)
            loss.backward()
            optimizer.step()

            if (i+1) % 1000 == 0:
                print(f"Batch: {i}, loss: {loss.item()}")

        print("Loss = %.3f" % loss.item())

    torch.save(autoencoder, "model.pt")

    autoencoder.eval()

    for i in range(5):
        test_image = random.choice(dataset)
        test_image = torch.tensor(test_image).view([1, 5, SEQUENCE]).float()
        test_reconst, _ = autoencoder(test_image)

        import matplotlib.pyplot as plt
        test_image = test_image.detach().numpy()
        test_reconst = test_reconst.detach().numpy()

        # print(test_image.detach().numpy().shape)
        # print(test_reconst.detach().numpy().shape)

        plt.plot(test_image[0, 1, :], label=f'{i}_org')
        plt.plot(test_reconst[0, 1, :], label=f'{i}_pred')
        
    plt.legend()
    plt.show()
    # print(test_reconst)

    # exit()

    # # Load data
    # train_data = datasets.MNIST(
    #     '~/data/mnist/', train=True, transform=transforms.ToTensor())
    # test_data = datasets.MNIST(
    #     '~/data/mnist/', train=False, transform=transforms.ToTensor())

    # # Training loop
    # for epoch in range(num_epochs):
    #     print("Epoch %d" % epoch)

    #     for i, (images, _) in enumerate(train_loader):    # Ignore image labels
    #         out, code = autoencoder(Variable(images))

    #         optimizer.zero_grad()
    #         loss = loss_fn(out, images)
    #         loss.backward()
    #         optimizer.step()

    #     print("Loss = %.3f" % loss.data[0])

    # # Try reconstructing on test data
    # test_image = random.choice(test_data)
    # test_image = Variable(test_image.view([1, 1, IMAGE_WIDTH, IMAGE_HEIGHT]))
    # test_reconst, _ = autoencoder(test_image)

    # torchvision.utils.save_image(test_image.data, 'orig.png')
    # torchvision.utils.save_image(test_reconst.data, 'reconst.png')
