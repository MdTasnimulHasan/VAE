

import torch
import numpy as np
import torch.nn as nn
from torch.optim import Adam, SGD
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
dropout_prob = 0.2

def get_args():
    """
    make parser to get parameters
    """

    parser = argparse.ArgumentParser(
        prog='VAE',
        description='using VAE for 768 dimensional data',
        epilog='Example: python VAE_exp.py --src_filename --output_filename')
    
    parser.add_argument('--src_filename', type = str, default = 'input_data_cvae.tsv', help='source .tsv file')
    parser.add_argument('--model_savepath', type = str, default = 'E://Tasnim//SonyCSL//cVAE', help='best model save folder')
    parser.add_argument('--saved_model_path', type = str, default = 'E://Tasnim//SonyCSL//cVAE//vae_model_best.pth', help='saved best model path')
    parser.add_argument('--mode', type = str, default = 'train', help='train for training and inference for inference')
    parser.add_argument('--output_filename', type = str, default = 'vae_output.txt', help='output_filepath')
    

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

     

class VAE(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=[600, 500, 400, 300, 200, 100, 50], latent_dim = 16, device=device):
        super(VAE, self).__init__()

        # encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[1], hidden_dim[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[2], hidden_dim[3]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[3], hidden_dim[4]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[4], hidden_dim[5]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[5], hidden_dim[6]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[6], latent_dim),
            nn.LeakyReLU(0.2)
            )
        
        # latent mean and variance 
        self.mean_layer = nn.Linear(latent_dim, 2)
        self.logvar_layer = nn.Linear(latent_dim, 2)
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(2, latent_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(latent_dim, hidden_dim[6]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[6], hidden_dim[5]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[5], hidden_dim[4]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[4], hidden_dim[3]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[3], hidden_dim[2]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[2], hidden_dim[1]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[1], hidden_dim[0]),
            nn.LeakyReLU(0.2),
            nn.Dropout(dropout_prob), 
            nn.Linear(hidden_dim[0], input_dim),
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
    KLD = - 0.5 * torch.sum(1+ log_var - mean.pow(2) - log_var.exp())
    # loss = reproduction_loss + KLD
    
    reconstruction_loss_fn = nn.MSELoss()
    reconstruction_loss = reconstruction_loss_fn(x_hat, x)
    wr = 1
    wk = 10
    
    loss = wr * reconstruction_loss + wk * KLD
    # loss = reproduction_loss 
    return loss
     

def train(args, model, optimizer, epochs, device, batch_size, train_loader, test_loader, x_dim=768):
    model.train()
    train_loss_list = []
    test_loss_list = []
    for epoch in range(epochs):
        train_loss = 0
        for batch_idx, xf in enumerate(train_loader):
            x = xf[1].to(device)
            

            optimizer.zero_grad()

            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            
            train_loss += loss.item()
            
            loss.backward()
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'Gradient norm for {name}: {param.grad.norm()}')
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        print("\tEpoch", epoch + 1, "\tAverage Train Loss: ", train_loss/(batch_idx*batch_size))
        
        current_train_loss = train_loss/(batch_idx*batch_size)
        train_loss_list.append(current_train_loss)
        
        if current_train_loss == min(train_loss_list):
            model_savename = 'vae_model_best.pth'
            torch.save(model, os.path.join(args.model_savepath, model_savename))
            
        model.eval()        
        test_loss = 0
        test_batch_size = 1
        for batch_idx, xf in enumerate(test_loader):
            x = xf[1]
            x = x.view(test_batch_size, x_dim).to(device)
            x_hat, mean, log_var = model(x)
            loss = loss_function(x, x_hat, mean, log_var)
            test_loss += loss.item()
            

        print("\tEpoch", epoch + 1, "\tAverage Test Loss: ", test_loss/(batch_idx*test_batch_size))
        test_loss_list.append(test_loss/(batch_idx*test_batch_size))
        
        epoch_count = range(1, len(train_loss_list) + 1)

        # Visualize loss history
        plt.plot(epoch_count, train_loss_list, 'r-')
        plt.plot(epoch_count, test_loss_list, 'b-')
        plt.legend(['Training Loss', 'Test Loss'])
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.show()
        
    return train_loss

def inference(args, device, test_loader, x_dim=768):
    
    model_path = args.saved_model_path  
    print('inference: ', model_path)
    model = torch.load(args.saved_model_path)

    model.eval()
    inference_size = 1
    for batch_idx, xf in enumerate(test_loader):
     
        x = xf[1]
        x = x.view(inference_size, x_dim).to(device)
        # mean, log_var, x2 = model.encode(x)
        mean, logvar, x2 = model.encode(x)
        z = model.reparameterization(mean, logvar)
        
        outout_vector = z
        str_output = str(outout_vector.detach().cpu().numpy())
        print(str_output)
        
        test_metadata_name = ' '.join(str(value) for value in xf[0])
        print(test_metadata_name)
        
        test_metadata_label = ' '.join(str(value) for value in xf[2])
        print(test_metadata_label)
        
        
        
        write_str = test_metadata_name + '\t' + test_metadata_label + '\t' + str_output
        
        with open(args.output_filename, 'a') as file:
            file.write(write_str + '\n')
        
    return x2

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
    
    
    dataset = CustomDataset(dataframe=df)

    train_size = int(0.8 * len(dataset)) 
    test_size = len(dataset) - train_size 
    train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])


    # create train and test dataloaders
    batch_size = 32
    test_batch_size = 1
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)
    
    
    model = VAE().to(device)
   
    optimizer = Adam(model.parameters(), lr=1e-7)
    # optimizer = SGD(model.parameters(), lr=1e-4)
    
    epochs= 1500
    train(args, model, optimizer, epochs, device, batch_size, train_loader, test_loader)
    inference(args, device, test_loader)
    

    return
if __name__ == '__main__':
    main()
