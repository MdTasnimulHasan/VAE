import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from mpl_toolkits.axes_grid1 import ImageGrid
from torchvision.utils import save_image, make_grid
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import ast
import os
import argparse
from torch.nn import functional as F
from torch import nn, optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
# hyperparameters
batch_size = 2
latent_size = 2
epochs = 50
xdim = 768

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def get_args():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(
        prog='cVAE',
        description='using cVAE for 768 dimensional data',
        epilog='Example: python cVAE.py --src_filename --output_filename')
    
    parser.add_argument('--src_filename', type = str, default = 'input_data_cvae.tsv', help='source .tsv file')
    parser.add_argument('--model_savepath', type = str, default = 'E://Tasnim//SonyCSL//cVAE', help='best model save folder')
    parser.add_argument('--output_filename', type = str, default = 'output.txt', help='output_filepath')
    

    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        
        self.id_name = dataframe['id']
        self.vector = dataframe['embedding']
        self.label = dataframe['document_type']
    def __getitem__(self, index):
        
        vector_data = np.array(ast.literal_eval(self.vector[index])).astype(np.float32)
        # print(vector_data.dtype)
        
        return self.id_name[index], vector_data, self.label[index]

    def __len__(self):
        return len(self.vector)

     

def one_hot(x, unique_labels):
    # Create a mapping from labels to indices
    label_to_index = {label: idx for idx, label in enumerate(unique_labels)}
    
    # Convert the list of string labels to a list of indices
    indices = torch.tensor([label_to_index[label] for label in x])
    
    # One-hot encode the indices
    max_index = len(unique_labels) - 1
    return torch.eye(max_index + 1)[indices]


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, xdim), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # criterion = nn.BCEWithLogitsLoss()
    # BCE = criterion(recon_x, x)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    reconstruction_loss_fn = nn.MSELoss()
    reconstruction_loss = reconstruction_loss_fn(recon_x, x)
   
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print("BCE: ", BCE, " KLD: ", KLD)
    # print("reconstruction_loss: ", reconstruction_loss, " KLD: ", KLD)
    
    # return reconstruction_loss + KLD
    return BCE + KLD

def train(model, optimizer, epoch, device, batch_size, unique_labels, train_loader, x_dim=xdim):
    model.train()
    train_loss = 0
    for batch_idx, xf in enumerate(train_loader):
        id_tag = xf[0]
        data = xf[1]
        # print(data)
        labels = xf[2]
        # print(labels)
        data = data.to(device)
        # labels = labels.to(device)
        labels = one_hot(labels, unique_labels)
        labels = labels.to(device)
        recon_batch, mu, logvar = model(data, labels)
        optimizer.zero_grad()
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.detach().cpu().numpy()
        optimizer.step()
        
        # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #     epoch, batch_idx * len(data), len(train_loader.dataset),
        #     100. * batch_idx / len(train_loader),
        #     loss.item() / len(data)))

    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return train_loss / len(train_loader.dataset)

def test(model, optimizer, epoch, device, batch_size, unique_labels, test_loader, x_dim=xdim):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, xf in enumerate(test_loader):
            
            id_tag = xf[0]
            data = xf[1]
            # print(data)
            labels = xf[2]
            # print(labels)
            data = data.to(device)
            # labels = labels.to(device)
            labels = one_hot(labels, unique_labels)
            labels = labels.to(device)
            
            recon_batch, mu, logvar = model(data, labels)
            test_loss += loss_function(recon_batch, data, mu, logvar).detach().cpu().numpy()

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))
    return test_loss 

def inference(model, optimizer, epoch, device, batch_size, unique_labels, test_loader, x_dim=xdim):
    model.eval()
    with torch.no_grad():
        for i, xf in enumerate(test_loader):
            
            id_tag = xf[0]
            data = xf[1]
            # print(data)
            labels = xf[2]
            # print(labels)
            data = data.to(device)
            # labels = labels.to(device)
            labels = one_hot(labels, unique_labels)
            labels = labels.to(device)
            
            mu, logvar = model(data, labels)
            
    return [mu, logvar] 

def main():
    
    args = get_args()
    
    df_load = pd.read_csv(args.src_filename, sep="\t", header=None)
    
    # column_names =  ['id','embedding','document_type']
    
    headers = df_load.iloc[0]
    df  = pd.DataFrame(df_load.values[1:], columns=headers)
    
    df_unique_labels = df['document_type'].unique()
    class_size = len(df_unique_labels)
    
    dataset = CustomDataset(dataframe=df)

    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # create train and test dataloaders
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    model = CVAE(xdim, latent_size, len(df_unique_labels)).to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    train_loss = []
    test_loss = []
    for epoch in range(1, epochs + 1):
        
        current_train_loss = train(model, optimizer, epoch, device, batch_size, df_unique_labels, train_loader)
        train_loss.append(current_train_loss)
        current_test_loss = test(model, optimizer, epoch, device, 1, df_unique_labels, test_loader)
        test_loss.append(current_test_loss)
        
        
        epoch_count = range(1, len(train_loss) + 1)

        # Visualize loss history
        plt.plot(epoch_count, train_loss, 'r-')
        plt.plot(epoch_count, test_loss, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        
        
        if current_train_loss == min(train_loss):
            model_savename = 'model_best.pth'
            torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': train_loss,
            }, os.path.join(args.model_savepath, model_savename))
    
    
    # inference(args, model, device, test_loader)
    

    return
if __name__ == '__main__':
    main()
