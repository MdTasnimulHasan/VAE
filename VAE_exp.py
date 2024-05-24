

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_args():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(
        prog='VAE',
        description='using VAE for 768 dimensional data',
        epilog='Example: python VAE_exp.py --src_filename --output_filename')
    
    parser.add_argument('--src_filename', type = str, default = 'metadata_vectorised_split_cleansed_prospectus_investment_ovjective.txt', help='source txt file')
    parser.add_argument('--output_filename', type = str, default = 'output.txt', help='output_filepath')
    

    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        self.vector = dataframe[3]
        self.metadata = dataframe[0]
        self.idx = dataframe[1]
        self.sentence = dataframe[2]
    def __getitem__(self, index):
        
        vector_data = np.array(ast.literal_eval(self.vector[index])).astype(np.float32)
        # print(vector_data.dtype)
        
        return vector_data, self.metadata[index], self.idx[index], self.sentence[index]

    def __len__(self):
        return len(self.vector)

     

class VAE(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=400, latent_dim=2, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(latent_dim, hidden_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(hidden_dim, input_dim),
            nn.Sigmoid()
            )
     
    def encode(self, x):
        x = self.encoder(x)
        mean, logvar = self.mean_layer(x), self.logvar_layer(x)
        return mean, logvar, x

    def reparameterization(self, mean, var):
        epsilon = torch.randn_like(var).to(device)      
        z = mean + var*epsilon
        return z

    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        mean, logvar, _ = self.encode(x)
        z = self.reparameterization(mean, logvar)
        x_hat = self.decode(z)
        return x_hat, mean, logvar
        

     

def loss_function(x, x_hat, mean, log_var):
    
    
    # The block below results in negative loss
    
    # reproduction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
    # KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    # loss = reproduction_loss + KLD
    
    reconstruction_loss_fn = nn.MSELoss()
    reconstruction_loss = reconstruction_loss_fn(x_hat, x)

    loss = reconstruction_loss 
    return loss
     

def train(model, optimizer, epochs, device, batch_size, train_loader, test_loader, x_dim=768):
    model.train()
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, xf in enumerate(train_loader):
            x = xf[0]
            x = x.view(batch_size, x_dim).to(device)

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            train_loss += loss.item()
            
            loss.backward()
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Train Loss: ", train_loss/(batch_idx*batch_size))
        
        
        model.eval()        
        test_loss = 0
        test_batch_size = 1
        for batch_idx, xf in enumerate(test_loader):
            x = xf[0]
            x = x.view(test_batch_size, x_dim).to(device)
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            test_loss += loss.item()
            

        print("\tEpoch", epoch + 1, "\tAverage Test Loss: ", test_loss/(batch_idx*batch_size))
        
        
    return train_loss

def inference(args, model, device, test_loader, x_dim=768):
    
    model.eval()
    inference_size = 1
    for batch_idx, xf in enumerate(test_loader):
     
        x = xf[0]
        x = x.view(inference_size, x_dim).to(device)
        mean, log_var, x2 = model.encode(x)
        
        outout_vector = x2
        str_output = str(outout_vector.detach().cpu().numpy())
        print(str_output)
        
        test_metadata_name = ' '.join(str(value) for value in xf[1])
        print(test_metadata_name)
        
        test_metadata_idx = ' '.join(str(value) for value in (xf[2].detach().cpu().numpy()))
        print(test_metadata_idx)
        
        test_metadata_str = ' '.join(str(value) for value in xf[3])
        print(test_metadata_str)
        
        
        write_str = test_metadata_name + '\t' + test_metadata_idx + '\t' + test_metadata_str + '\t' + str_output
        
        with open(args.output_filename, 'a') as file:
            file.write(write_str + '\n')
        
    return x2

def main():
    
    args = get_args()
    
    df = pd.read_csv(args.src_filename, sep="\t", header=None)
    dataset = CustomDataset(dataframe=df)

    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    # create train and test dataloaders
    batch_size = 2
    test_batch_size = 1
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=test_batch_size, shuffle=False)
    
    
    model = VAE().to(device)
   
    optimizer = Adam(model.parameters(), lr=1e-5)
    
    epochs= 50000
    train(model, optimizer, epochs, device, batch_size, train_loader, test_loader)
    inference(args, model, device, test_loader)
    

    return
if __name__ == '__main__':
    main()
