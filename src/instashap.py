import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader
from torch.distributions.categorical import Categorical
from tqdm.auto import tqdm
from copy import deepcopy

# use sian to get the interactions
from sian.utils import gettimestamp
from sian.data import Final_TabularDataset
from sian.models import TrainingArgs
from sian.fis import batchwise_FIS_Hyperparameters
from sian.interpret import masked_FID_Hyperparameters
from sian import initalize_the_explainer
from sian import train_mlp_final, do_the_fis_final

class ShapleySampler:
    '''
    For sampling player subsets from the Shapley distribution. 

    Args:
      num_features: number of features.
    '''

    def __init__(self, num_features):
        arange = torch.arange(1, num_features)
        w = 1 / (arange * (num_features - arange))
        w = w / torch.sum(w)
        self.categorical = Categorical(probs=w)
        self.num_features = num_features
        # main diagonal 
        self.tril = np.tril(
            np.ones((num_features - 1, num_features), dtype=np.float32), k=0
        )
        self.rng = np.random.default_rng()

    def sample(self, batch_size, paired_sampling):
        '''
        Generate sample.

        Args:
          batch_size: number of samples.
          paired_sampling: whether to use paired sampling.
        '''
        num_included = 1 + self.categorical.sample([batch_size])
        S = self.tril[num_included - 1]
        S = self.rng.permuted(S, axis=1)  # Note: permutes each row.
        if paired_sampling:
            S[1::2] = 1 - S[0:(batch_size - 1):2]  # Note: allows batch_size % 2 == 1.
        return torch.from_numpy(S)


class surrogate:
    '''
    Surrogate model for estimating Shapley values. implemented with SIAN

    Args:
      num_features: number of features.
    '''

    def __init__(self, mlp_args = None, dataset_obj = None, max_number_of_rounds = 20, order=2, output_type="regression"):
        super().__init__()
        self.mlp_args = mlp_args
        self.dataset_obj = dataset_obj
        self.max_number_of_rounds = max_number_of_rounds
        self.order = order
        self.output_type = output_type

    def get_mlp(self):
        #TODO: need to be mlp.arg
        mlp_results = train_mlp_final(self.dataset_obj, self.mlp_args)
        self.trained_mlp = mlp_results['trained_mlp']
        self.val_tensor = mlp_results['val_tensor']

    def get_FID_hyper(self, output_type = "regression", device = 'cpu'):

        if self.val_tensor is None:
            self.get_mlp()
        
        grouped_features_dict = self.dataset_obj.get_grouped_feature_dict()

        fid_masking_style = "masking_based"
        score_type_name = "new_arch_inter_sobol_score"
        inc_rem_pel_list = ['inc_inter_sobol_score', 'rem_inter_sobol_score', 'new_arch_inter_sobol_score',] #NOTE: only for batchwise plots
        self.fis_valX = self.val_tensor

        my_FID_hypers = masked_FID_Hyperparameters(fid_masking_style, output_type, score_type_name, inc_rem_pel_list,
                                                grouped_features_dict)
        self.FID_hyper = my_FID_hypers
    
    def get_FIS_hyper(self, device = 'cpu'): # MAX_K: maximum interaction order

        MAX_K = self.order
        max_number_of_rounds = self.max_number_of_rounds
        inters_per_round = 1
        tau_tup=(1.0,0.5,0.33)
        
        tau_thresholds = {}
        for k in range(MAX_K): 
            tau_thresholds[k+1] = tau_tup[k]
        
        my_FIS_hypers = batchwise_FIS_Hyperparameters(MAX_K, tau_thresholds, max_number_of_rounds, inters_per_round,
                    # jam_arch, 
                    None, 
                    tuples_initialization=None,pick_underlings=False,fill_underlings=False,PLOTTING=True)

        self.FIS_hyper = my_FIS_hypers

    def get_interactions(self, device = 'cpu'):

        output_type = self.output_type if hasattr(self, 'output_type') else "regression"

        if not hasattr(self, 'trained_mlp'):
            self.get_mlp()
        if not hasattr(self, 'FID_hyper'):
            self.get_FID_hyper(output_type=output_type)
        if not hasattr(self, 'FIS_hyper'):
            self.get_FIS_hyper()

        jam_arch = initalize_the_explainer(self.trained_mlp, self.FID_hyper)
        self.FIS_hyper.add_the_explainer(jam_arch)

        FIS_interactions = do_the_fis_final(self.FIS_hyper, self.fis_valX, AGG_K=100)

        self.interactions = FIS_interactions

        return self.interactions

    def get_transform_matrix(self): # use the interaction to get the transform matrix, so that we can get the transformed dataset and SHAP sampler to run the InstaSHAP model by batch

        transform_matrix = torch.zeros((self.dataset_obj.get_D(), len(self.interactions)-1))
        for i in range(1, len(self.interactions)):
            interaction = self.interactions[i]
            for feature_idx in interaction:
                transform_matrix[feature_idx, i-1] = 1.0
        self.transform_matrix = transform_matrix

        return self.transform_matrix
    

class phi(nn.Module): # samll network to estimate phi values

    def __init__(self, num_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        '''
        Forward pass.

        Args:
          x: input tensor of shape (batch_size, num_features)
        '''
        return self.net(x)


class InstaSHAP(nn.Module):
    '''
    InstaSHAP model for estimating Shapley values.

    Args:
      num_features: number of features.
      surrogate_model: surrogate model for estimating Shapley values.
    '''

    def __init__(self, interactions, transform_matrix, device='cpu'):
        super().__init__()
        self.model = nn.ModuleList(
            [phi(num_features=len(interactions[i])) for i in range(1, len(interactions))]
        )
        self.model.to(device)
        self.interactions = interactions
        self.len_interactions = [len(interactions[i]) for i in range(1, len(interactions))]
        self.transform_matrix = transform_matrix.to(device)
        self.device = device

    def forward(self, x, S):
        '''
        Forward pass.

        Args:
          x: input tensor of shape (batch_size, num_features)
          S: subset tensor of shape (batch_size, num_features)
        '''
        batch_size = x.shape[0]

        transformed_S = S @ self.transform_matrix  # shape: (batch_size, num_interactions)
        len_interactions_tensor = torch.tensor(self.len_interactions, device=self.device).unsqueeze(0)  # shape: (1, num_interactions)
        is_active = (transformed_S >= len_interactions_tensor - 0.01).float()  # shape: (batch_size, num_interactions), prevent precision errors
        
        inter_outputs = []
        for i, phi_model in enumerate(self.model):
            feature_selected = self.transform_matrix[:, i].squeeze() 
            inter_input = x[:, feature_selected == 1]  # shape: (batch_size, num_features_in_interaction)
            inter_output = phi_model(inter_input)  # shape: (batch_size, 1)
            inter_outputs.append(inter_output)
        
        inter_outputs = torch.cat(inter_outputs, dim=1)  # shape: (batch_size, num_interactions)
        output = torch.sum(inter_outputs * is_active, dim=1, keepdim=True)

        return output
    
    def train_instaSHAP(self, train_loader, num_epochs=10, lr=5e-3, device='cpu'):
        '''
        Train InstaSHAP model.

        Args:
          train_loader: DataLoader for training data.
          num_epochs: number of epochs.
          lr: learning rate.
          device: device to use.
        '''
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()

        self.train()

        x_batch, _ = next(iter(train_loader))
        num_features = x_batch.shape[1]
        sampler = ShapleySampler(num_features=num_features)
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for x_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):

                x_batch = x_batch.to(device)
                y_batch = y_batch.to(device)
                S_batch = sampler.sample(batch_size=x_batch.shape[0], paired_sampling=True).to(device)

                optimizer.zero_grad()
                outputs = self.forward(x_batch, S_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item() * x_batch.size(0)
            epoch_loss /= len(train_loader.dataset)
            print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}')

    def get_shapley_values(self, x):
        '''
        Get Shapley values for input tensor.

        Args:
          x: input tensor of shape (batch_size, num_features)
        '''
        transform_x = x @ self.transform_matrix  # shape: (batch_size, num_interactions)
        shapley_values = torch.zeros_like(transform_x)
        for i, phi_model in enumerate(self.model):
            feature_selected = self.transform_matrix[:, i].squeeze() 
            inter_input = x[:, feature_selected == 1]  # shape: (batch_size, num_features_in_interaction)
            inter_output = phi_model(inter_input)  # shape: (batch_size, 1)
            shapley_values[:, i] = inter_output.squeeze()

        return shapley_values

class phi_classifier(nn.Module):
    def __init__(self, num_features, out_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_features, 16),
            nn.ReLU(),
            nn.Linear(16, out_features),
        )

    def forward(self, x):
        return self.net(x)

class InstaSHAP_classifier(InstaSHAP):
    def __init__(self, interactions, transform_matrix, num_classes, device='cpu'):
        super().__init__(interactions, transform_matrix, device)
        self.num_classes = num_classes
        self.model = nn.ModuleList([
            phi_classifier(len(interactions[i]), num_classes) for i in range(1, len(interactions))
        ])
        self.to(device)

    def forward(self, x, S):
        batch_size = x.shape[0]

        transformed_S = S @ self.transform_matrix
        len_interactions_tensor = torch.tensor(self.len_interactions, device=self.device).unsqueeze(0)
        is_active = (transformed_S >= len_interactions_tensor - 0.01).float()
        
        inter_outputs = []
        for i, phi_model in enumerate(self.model):
            feature_selected = self.transform_matrix[:, i].squeeze() 
            inter_input = x[:, feature_selected == 1]
            inter_output = phi_model(inter_input)
            inter_outputs.append(inter_output)
        
        inter_outputs = torch.stack(inter_outputs, dim=1)
        is_active = is_active.unsqueeze(-1)
        output = torch.sum(inter_outputs * is_active, dim=1)

        return output

    def train_instaSHAP(self, train_loader, num_epochs=10, lr=5e-3):
        optimizer = optim.Adam(self.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        x_sample, _ = next(iter(train_loader))
        sampler = ShapleySampler(num_features=x_sample.shape[1])

        self.train()
        for epoch in range(num_epochs):
            total_loss = 0.0
            for x_batch, y_batch in tqdm(train_loader, desc=f'Epoch {epoch+1}'):
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device)
                S_batch = sampler.sample(x_batch.shape[0], paired_sampling=True).to(self.device)

                optimizer.zero_grad()
                logits = self.forward(x_batch, S_batch)
                
                if y_batch.dim() > 1:
                    y_batch = y_batch.squeeze()
                
                loss = criterion(logits, y_batch.long())
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * x_batch.size(0)
            
            avg_loss = total_loss / len(train_loader.dataset)
            print(f'Epoch {epoch+1}, Loss: {avg_loss:.4f}')

    def get_shapley_values(self, x):
        self.eval()
        x = x.to(self.device)
        with torch.no_grad():
            shaps = []
            for i, phi_model in enumerate(self.model):
                feature_selected = self.transform_matrix[:, i].squeeze() 
                inter_input = x[:, feature_selected == 1]
                shaps.append(phi_model(inter_input))
            return torch.stack(shaps, dim=1)