# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 18:11:21 2024

@author: mewkh
"""

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
batch_size = 32
# latent_size = 2
# xout_dim = 2
epochs = 1500
xdim = 768
dropout_prob = 0.2
# SEED = 42
# np.random.seed(SEED)
# torch.manual_seed(SEED)
# torch.cuda.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

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
    parser.add_argument('--saved_model_path', type = str, default = 'E://Tasnim//SonyCSL//cVAE//cvae_model_best.pth', help='saved best model path')
    parser.add_argument('--mode', type = str, default = 'train', help='train for training and inference for inference')
    parser.add_argument('--output_filename', type = str, default = 'cvae_output.txt', help='output_filepath')
    

    return parser.parse_args()


class CustomDataset(Dataset):
    def __init__(self, dataframe):
        
        self.id_name = dataframe['id']
        self.vector = dataframe['embedding']
        self.label = dataframe['document_type']
    def __getitem__(self, index):
        
        # vector_data = np.array(ast.literal_eval(self.vector[index])).astype(np.float32)
        vector_data = self.vector[index]
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
    def __init__(self, feature_size = 768, hidden_dim=[600, 500, 400, 300, 200, 100, 50], latent_size=16, class_size=3 ):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc11  = nn.Linear(feature_size + class_size, hidden_dim[0])
        self.fc12  = nn.Linear(hidden_dim[0], hidden_dim[1])
        self.fc13  = nn.Linear(hidden_dim[1], hidden_dim[2])
        self.fc14  = nn.Linear(hidden_dim[2], hidden_dim[3])
        self.fc15  = nn.Linear(hidden_dim[3], hidden_dim[4])
        self.fc16  = nn.Linear(hidden_dim[4], hidden_dim[5])
        self.fc17  = nn.Linear(hidden_dim[5], hidden_dim[6])
        self.fc2  = nn.Linear(hidden_dim[6], latent_size)
        
        self.fc21 = nn.Linear(latent_size, 2)
        self.fc22 = nn.Linear(latent_size, 2)

        # decode
        self.fc31 = nn.Linear(2 + class_size, latent_size)
        self.fc32 = nn.Linear(latent_size, hidden_dim[6])
        self.fc33 = nn.Linear(hidden_dim[6], hidden_dim[5])
        self.fc34 = nn.Linear(hidden_dim[5], hidden_dim[4])
        self.fc35 = nn.Linear(hidden_dim[4], hidden_dim[3])
        self.fc36 = nn.Linear(hidden_dim[3], hidden_dim[2])
        self.fc37 = nn.Linear(hidden_dim[2], hidden_dim[1])
        self.fc38 = nn.Linear(hidden_dim[1], hidden_dim[0])
        
        self.fc4 = nn.Linear(hidden_dim[0], feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        
        e1 = self.elu(self.fc11(inputs))
        e2 = self.elu(self.fc12(e1))
        e3 = self.elu(self.fc13(e2))
        e4 = self.elu(self.fc14(e3))
        e5 = self.elu(self.fc15(e4))
        e6 = self.elu(self.fc16(e5))
        e7 = self.elu(self.fc17(e6))
        e8 = self.elu(self.fc2(e7))
        
        z_mu = self.fc21(e8)
        z_var = self.fc22(e8)
        return z_mu, z_var, e8

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
        
        d1 = self.elu(self.fc31(inputs))
        d2 = self.elu(self.fc32(d1))
        d3 = self.elu(self.fc33(d2))
        d4 = self.elu(self.fc34(d3))
        d5 = self.elu(self.fc35(d4))
        d6 = self.elu(self.fc36(d5))
        d7 = self.elu(self.fc37(d6))
        d8 = self.elu(self.fc38(d7))
        
        
        return self.sigmoid(self.fc4(d8))

    def forward(self, x, c):
        mu, logvar, x_com = self.encode(x.view(-1, xdim), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar


def loss_function(recon_x, x, mu, logvar):
    # criterion = nn.BCEWithLogitsLoss()
    # BCE = criterion(recon_x, x)
    # BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    
    reconstruction_loss_fn = nn.MSELoss()
    reconstruction_loss = reconstruction_loss_fn(recon_x, x)
   
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print("BCE: ", BCE, " KLD: ", KLD)
    # print("reconstruction_loss: ", reconstruction_loss, " KLD: ", KLD)
    wr = 1
    wk = 10
    
    loss = wr * reconstruction_loss + wk * KLD
    return loss
    # return BCE + KLD

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

    print('====> Epoch: {} Average loss: {:.10f}'.format(
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
    print('====> Test set loss: {:.10f}'.format(test_loss))
    return test_loss 

def inference(args, model, device, unique_labels, test_loader, x_dim=xdim):
    
    
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
            
            mu, logvar, x_com= model.encode(data, labels)
            z = model.reparameterize(mu, logvar)
            x_com = z
            print(x_com)
            
            outout_vector = x_com
            str_output = str(outout_vector.detach().cpu().numpy())
            print(str_output)
            
            test_metadata_name = ' '.join(str(value) for value in id_tag)
            print(test_metadata_name)
            
            test_metadata_label = ' '.join(str(value) for value in (labels.detach().cpu().numpy()))
            print(test_metadata_label)
            
            
            
            write_str = test_metadata_name + '\t' + test_metadata_label + '\t' + str_output
            
            with open(args.output_filename, 'a') as file:
                file.write(write_str + '\n')
            
    return 

def main():
    
    args = get_args()
    
    df_load = pd.read_csv(args.src_filename, sep="\t", header=None)
    
    # column_names =  ['id','embedding','document_type']
    
    headers = df_load.iloc[0]
    df  = pd.DataFrame(df_load.values[1:], columns=headers)
    # print(df['embedding'])
    
    str_vector = df['embedding']
    
    
    
    np_vector_list = []
    for i in range (0, len(df['embedding']), 1):
        np_vector_list.append(np.array(ast.literal_eval(str_vector[i])).astype(np.float32))
    np_vector = np.array(np_vector_list)
    # print(np_vector[0])
    
    
    
    # Choose normalization method: Standardization or Min-Max Scaling
    normalization_method = 'standardization'  # Choose 'standardization' or 'min-max'
    
    if normalization_method == 'standardization':
        # Standardization
        mean = np.mean(np_vector, axis=0)
        std = np.std(np_vector, axis=0)
        
        # Standardize the input vector
        x_standardized = (np_vector - mean) / std
        
        # Ensure no division by zero
        std[std == 0] = 1
        x_normalized = (np_vector - mean) / std
    elif normalization_method == 'min-max':
        # Compute min and max
        min_val = np.min(np_vector, axis=0)
        max_val = np.max(np_vector, axis=0)
        
        # Min-Max scale the input vector
        x_minmax = (np_vector - min_val) / (max_val - min_val)
        
        # Ensure no division by zero
        range_val = max_val - min_val
        range_val[range_val == 0] = 1
        x_normalized = (np_vector - min_val) / range_val
        
    # print(mean, std)
    # print(f"{normalization_method.capitalize()} Normalized DataFrame:")
    # print(x_normalized)
    # print(x_normalized.shape)
    
    for i in range (0, len(df['embedding']), 1):
        df['embedding'][i] = x_normalized[i]
    
    # print(df['embedding'])
    
    
    df_unique_labels = df['document_type'].unique()
    class_size = len(df_unique_labels)
    
    dataset = CustomDataset(dataframe=df)

    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

    # create train and test dataloaders
    
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    
    
    if args.mode == 'train':
        model = CVAE().to(device)
        optimizer = optim.Adam(model.parameters(), lr=1e-7)
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
                model_savename = 'cvae_model_best.pth'
                torch.save(model, os.path.join(args.model_savepath, model_savename))
                
    # elif args.mode == 'inference':
    # Load the model state dictionary
    model_path = args.saved_model_path  
    loaded_model = torch.load(args.saved_model_path)

    # Set the model to evaluation mode
    loaded_model.eval()
    
    inference(args, loaded_model, device, df_unique_labels, test_loader, x_dim=xdim)
        
        
    
    # inference(args, model, device, test_loader)
    

    return 
if __name__ == '__main__':
    main()
