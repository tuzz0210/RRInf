from time import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import pickle, os
import torch

from matplotlib import pyplot as plt
import numpy as np


class IFEngine(object):
    def __init__(self):
        self.time_dict=defaultdict(list)
        self.hvp_dict=defaultdict(list)
        self.IF_dict=defaultdict(list)
        self.device='cuda'

    def preprocess_gradients(self, tr_grad_dict, val_grad_dict, noise_index):
        self.tr_grad_dict = tr_grad_dict
        self.val_grad_dict = val_grad_dict
        self.noise_index = noise_index

        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())
        self.compute_val_grad_avg()
        self.total_param_size=0
        for weight_name in self.val_grad_dict[0]:
            self.total_param_size=self.total_param_size+self.val_grad_dict[0][weight_name].numel()
        self.compute_design_matrix()  

    def compute_design_matrix(self):
        # Construct the gradient matrix for sampling a neuron in RRInf
        self.val_grad_avg = torch.cat([v for _, v in self.val_grad_avg_dict.items()], 0).reshape(-1).to(self.device)
        self.S=torch.zeros(self.n_train)
        self.Phi = torch.zeros((self.total_param_size, self.n_train)).to(self.device)
        for tr_id, grad_dict in self.tr_grad_dict.items():
            tmp_grad = torch.cat([grad_dict[k] for k, _ in self.val_grad_avg_dict.items()], 0)
            self.Phi[:, tr_id] = tmp_grad.reshape(-1)
            self.S[tr_id]=torch.mean(tmp_grad**2)

    def compute_val_grad_avg(self):
        # Compute the avg gradient on the validation dataset
        self.val_grad_avg_dict={}
        for weight_name in self.val_grad_dict[0]:
            self.val_grad_avg_dict[weight_name]=torch.zeros(self.val_grad_dict[0][weight_name].shape)
            for val_id in self.val_grad_dict:
                self.val_grad_avg_dict[weight_name] += self.val_grad_dict[val_id][weight_name] / self.n_val

    def compute_IF_baselines(self, lambda_const_param=10, compute_accurate=True):
        self.compute_IF_identity()
        self.compute_IF_DataInf(lambda_const_param=lambda_const_param)
        self.compute_IF_LiSSA(lambda_const_param=lambda_const_param)

        if compute_accurate:
            self.compute_hvp_accurate(lambda_const_param=lambda_const_param)

    def compute_IF_identity(self):
        start_time = time()
        self.hvp_dict['identity'] = self.val_grad_avg_dict.copy()
        if_tmp_dict = {}
        for tr_id in self.tr_grad_dict:
            if_tmp_value = 0
            for weight_name in self.val_grad_avg_dict:
                if_tmp_value += torch.sum(self.hvp_dict['identity'][weight_name]*self.tr_grad_dict[tr_id][weight_name])
            if_tmp_dict[tr_id]= -if_tmp_value

        self.IF_dict['identity'] = pd.Series(if_tmp_dict, dtype=float).to_numpy()
        self.time_dict['identity'] = time()-start_time

    def compute_IF_DataInf(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict={}
        for weight_name in self.val_grad_avg_dict:
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation
            hvp=torch.zeros(self.val_grad_avg_dict[weight_name].shape)
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                C_tmp = torch.sum(self.val_grad_avg_dict[weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                hvp += (self.val_grad_avg_dict[weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
            hvp_proposed_dict[weight_name] = hvp
        self.hvp_dict['DataInf'] = hvp_proposed_dict
        if_tmp_dict = {}
        for tr_id in self.tr_grad_dict:
            if_tmp_value = 0
            for weight_name in self.val_grad_avg_dict:
                if_tmp_value += torch.sum(self.hvp_dict['DataInf'][weight_name]*self.tr_grad_dict[tr_id][weight_name])
            if_tmp_dict[tr_id]= -if_tmp_value

        self.IF_dict['DataInf'] = pd.Series(if_tmp_dict, dtype=float).to_numpy()
        self.time_dict['DataInf'] = time()-start_time

    def compute_IF_accurate(self, lambda_const_param=10):
        start_time = time()
        hvp_accurate_dict={}
        for weight_name in self.val_grad_avg_dict:
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation (eigenvalue decomposition)
            AAt_matrix = torch.zeros(torch.outer(self.tr_grad_dict[0][weight_name].reshape(-1),
                                                 self.tr_grad_dict[0][weight_name].reshape(-1)).shape)
            for tr_id in self.tr_grad_dict:
                tmp_mat = torch.outer(self.tr_grad_dict[tr_id][weight_name].reshape(-1),
                                      self.tr_grad_dict[tr_id][weight_name].reshape(-1))
                AAt_matrix += tmp_mat

            L, V = torch.linalg.eig(AAt_matrix)
            L, V = L.float(), V.float()
            hvp = self.val_grad_avg_dict[weight_name].reshape(-1) @ V
            hvp = (hvp / (lambda_const + L/ self.n_train)) @ V.T

            hvp_accurate_dict[weight_name] = hvp.reshape(len(self.tr_grad_dict[0][weight_name]), -1)
            del tmp_mat, AAt_matrix, V # to save memory
        self.hvp_dict['accurate'] = hvp_accurate_dict
        if_tmp_dict = {}
        for tr_id in self.tr_grad_dict:
            if_tmp_value = 0
            for weight_name in self.val_grad_avg_dict:
                if_tmp_value += torch.sum(self.hvp_dict['accurate'][weight_name]*self.tr_grad_dict[tr_id][weight_name])
            if_tmp_dict[tr_id]= -if_tmp_value

        self.IF_dict['accurate'] = pd.Series(if_tmp_dict, dtype=float).to_numpy()
        self.time_dict['accurate'] = time()-start_time

    def compute_IF_LiSSA(self, lambda_const_param=10, n_iteration=10, alpha_const=1.):
        start_time = time()
        hvp_LiSSA_dict={}
        for weight_name in self.val_grad_avg_dict:
            # lambda_const computation
            S=torch.zeros(len(self.tr_grad_dict.keys()))
            for tr_id in self.tr_grad_dict:
                tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                S[tr_id]=torch.mean(tmp_grad**2)
            lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

            # hvp computation
            running_hvp=self.val_grad_avg_dict[weight_name]
            for _ in range(n_iteration):
                hvp_tmp=torch.zeros(self.val_grad_avg_dict[weight_name].shape)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    hvp_tmp += (torch.sum(tmp_grad*running_hvp)*tmp_grad - lambda_const*running_hvp) / self.n_train
                running_hvp = self.val_grad_avg_dict[weight_name] + running_hvp - alpha_const*hvp_tmp
            hvp_LiSSA_dict[weight_name] = running_hvp
        self.hvp_dict['LiSSA'] = hvp_LiSSA_dict
        if_tmp_dict = {}
        for tr_id in self.tr_grad_dict:
            if_tmp_value = 0
            for weight_name in self.val_grad_avg_dict:
                if_tmp_value += torch.sum(self.hvp_dict['LiSSA'][weight_name]*self.tr_grad_dict[tr_id][weight_name])
            if_tmp_dict[tr_id]= -if_tmp_value

        self.IF_dict['LiSSA'] = pd.Series(if_tmp_dict, dtype=float).to_numpy()
        self.time_dict['LiSSA'] = time()-start_time


    def compute_IF_RRInf(self, num_iterations=1000, learning_rate=0.01,lambda_const_param=10, layer=True):
        """ Compute influence function using RRInf. If 'layer' is True, a layer is randomly selected each iteration. Otherwise, RRInf randomly samples a neuron. """
        start_time = time()

        # Initialize ω (influence function values) as a zero vector
        omega = torch.zeros(self.n_train, requires_grad=False).to(self.device)

        if layer:
            layer_names = list(self.val_grad_avg_dict.keys()) # Extract layer names
            np.random.shuffle(layer_names)  # Shuffle layers initially
        else:
            lambda_const = torch.mean(self.S) / lambda_const_param
            lam = self.total_param_size/lambda_const
            upsilon = - (self.n_train * self.val_grad_avg)

        for t in tqdm(range(num_iterations), desc="Running RRInf for Influence Function"):
            if layer:    
                selected_layer = torch.randint(0, len(layer_names), (1,)).item() # Randomly select a layer
                layer_name = layer_names[selected_layer]
                upsilon_t = - (self.n_train) * self.val_grad_avg_dict[layer_name].reshape(-1).to(self.device)
                param_size = upsilon_t.size(0) # Get number of parameters in selected layer

                # Construct Phi dynamically (only for selected layer)
                Phi_t = torch.zeros((param_size, self.n_train)).to(self.device)
                S_t=torch.zeros(self.n_train)
                for tr_id, grad_dict in self.tr_grad_dict.items():
                    tmp_grad = grad_dict[layer_name]
                    Phi_t[:, tr_id] = tmp_grad.reshape(-1)
                    S_t[tr_id]=torch.mean(tmp_grad**2)
                lambda_const = torch.mean(S_t) / lambda_const_param # layer-wise lambda
                lam = self.total_param_size/lambda_const

                #normalization step
                normalization=torch.sum(Phi_t**2,1,keepdim=False) 
            else:
                iota_t = torch.randint(0, self.total_param_size, (1,)).item()  # Randomly select a neuron
                
                # Construct Phi dynamically (only for selected neuron)
                Phi_t = self.Phi[iota_t]
                upsilon_t = upsilon[iota_t]
                
                #normalization step
                normalization=torch.sum(Phi_t**2) 


            # Compute residual error for selected model parameters
            e_t = Phi_t @ omega - upsilon_t

            # Compute normalised stochastic gradient
            if layer:
                grad = (2/param_size) * Phi_t.transpose(0,1) @ (e_t/normalization)
            else:
                grad = 2 * e_t * Phi_t/normalization
            grad=grad+2*self.n_train*omega/lam

            # Apply normalized SGD update
            omega -= learning_rate * grad


        # Store the computed influence function
        self.IF_dict["RRInf"] = omega.cpu().numpy()
        self.time_dict["RRInf"] = time() - start_time


    def save_result(self, noise_index, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['noise_index']=noise_index
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)


class IFEngineGeneration(object):
    '''
    This class computes the influence function for every validation data point
    '''
    def __init__(self):
        self.time_dict=defaultdict(list)
        self.hvp_dict=defaultdict(list)
        self.IF_dict=defaultdict(list)
        self.device='cuda'

    def preprocess_gradients(self, tr_grad_dict, val_grad_dict):
        self.tr_grad_dict = tr_grad_dict
        self.val_grad_dict = val_grad_dict

        self.n_train = len(self.tr_grad_dict.keys())
        self.n_val = len(self.val_grad_dict.keys())
        self.total_param_size=0
        for weight_name in self.val_grad_dict[0]:
            self.total_param_size=self.total_param_size+self.val_grad_dict[0][weight_name].numel()
        self.compute_design_matrix()

    def compute_design_matrix(self):
        # Construct the gradient matrix for sampling a neuron in RRInf
        self.S=torch.zeros(self.n_train)
        self.Phi = torch.zeros((self.total_param_size, self.n_train)).to(self.device)
        self.val_grad=torch.zeros((self.total_param_size, self.n_val)).to(self.device)
        for val_id, grad_dict in self.val_grad_dict.items():
            self.val_grad[:, val_id] = torch.cat([v.reshape(-1) for _, v in grad_dict.items()], 0)
        for tr_id, grad_dict in self.tr_grad_dict.items():
            tmp_grad = torch.cat([grad_dict[k].reshape(-1) for k, _ in self.val_grad_dict[0].items()], 0)
            self.Phi[:, tr_id] = tmp_grad
            self.S[tr_id]=torch.mean(tmp_grad**2)


    def compute_IF_baselines(self, lambda_const_param=10):
        self.compute_IF_identity()
        self.compute_IF_DataInf(lambda_const_param=lambda_const_param)


    def compute_IF_identity(self):
        start_time = time()
        self.hvp_dict["identity"] = self.val_grad_dict.copy()
        if_tmp_dict = defaultdict(dict)
        for tr_id in self.tr_grad_dict:
            for val_id in self.val_grad_dict:
                if_tmp_value = 0
                for weight_name in self.val_grad_dict[0]:
                    if_tmp_value += torch.sum(self.hvp_dict['identity'][val_id][weight_name]*self.tr_grad_dict[tr_id][weight_name])
                if_tmp_dict[tr_id][val_id]=-if_tmp_value

        self.IF_dict['identity'] = pd.DataFrame(if_tmp_dict, dtype=float)
        self.time_dict['identity'] = time()-start_time

    def compute_IF_DataInf(self, lambda_const_param=10):
        start_time = time()
        hvp_proposed_dict=defaultdict(dict)
        for val_id in tqdm(self.val_grad_dict.keys()):
            for weight_name in self.val_grad_dict[val_id]:
                # lambda_const computation
                S=torch.zeros(len(self.tr_grad_dict.keys()))
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    S[tr_id]=torch.mean(tmp_grad**2)
                lambda_const = torch.mean(S) / lambda_const_param # layer-wise lambda

                # hvp computation
                hvp=torch.zeros(self.val_grad_dict[val_id][weight_name].shape)
                for tr_id in self.tr_grad_dict:
                    tmp_grad = self.tr_grad_dict[tr_id][weight_name]
                    C_tmp = torch.sum(self.val_grad_dict[val_id][weight_name] * tmp_grad) / (lambda_const + torch.sum(tmp_grad**2))
                    hvp += (self.val_grad_dict[val_id][weight_name] - C_tmp*tmp_grad) / (self.n_train*lambda_const)
                hvp_proposed_dict[val_id][weight_name] = hvp
        self.hvp_dict['DataInf'] = hvp_proposed_dict
        if_tmp_dict = defaultdict(dict)
        for tr_id in self.tr_grad_dict:
            for val_id in self.val_grad_dict:
                if_tmp_value = 0
                for weight_name in self.val_grad_dict[0]:
                    if_tmp_value += torch.sum(self.hvp_dict['DataInf'][val_id][weight_name]*self.tr_grad_dict[tr_id][weight_name])
                if_tmp_dict[tr_id][val_id]=-if_tmp_value

        self.IF_dict['DataInf'] = pd.DataFrame(if_tmp_dict, dtype=float)
        self.time_dict['DataInf'] = time()-start_time


    def compute_IF_RRInf(self, num_iterations=2000, learning_rate=0.01,lambda_const_param=10,layer=True):
        """ Compute influence function using RRInf. If 'layer' is True, a layer is randomly selected each iteration. Otherwise, RRInf randomly samples a neuron. """
        start_time = time()

        # Initialize ω (influence function values) as a zero matrix
        omega = torch.zeros((self.n_train,self.n_val), requires_grad=False).to(self.device)

        if layer:
            layer_names = list(self.val_grad_dict[0].keys()) # Extract layer names
            np.random.shuffle(layer_names)  # Shuffle layers initially
        else:
            lambda_const = torch.mean(self.S) / lambda_const_param 
            lam = self.total_param_size/lambda_const
            upsilon = - (self.n_train * self.val_grad)

        for t in tqdm(range(num_iterations), desc="Running RRInf for Influence Function"):
            if layer:
                selected_layer = torch.randint(0, len(layer_names), (1,)).item()  # Randomly select a layer
                layer_name = layer_names[selected_layer]
                param_size = self.val_grad_dict[0][layer_name].numel() # Get number of parameters in selected layer

                # Construct Phi dynamically (only for selected layer)
                Phi_t = torch.zeros((param_size, self.n_train)).to(self.device)
                upsilon_t = torch.zeros((param_size, self.n_val)).to(self.device)
                S_t=torch.zeros(self.n_train)
                for val_id, grad_dict in self.val_grad_dict.items():
                    upsilon_t[:, val_id] =(-(self.n_train)*grad_dict[layer_name]).reshape(-1)
                for tr_id, grad_dict in self.tr_grad_dict.items():
                    tmp_grad = grad_dict[layer_name]
                    Phi_t[:, tr_id] = tmp_grad.reshape(-1)
                    S_t[tr_id]=torch.mean(tmp_grad**2)
                lambda_const = torch.mean(S_t) / lambda_const_param # layer-wise lambda
                lam = self.total_param_size/lambda_const

                #normalization step
                normalization=torch.sum(Phi_t**2,1,keepdim=True)
            else:
                iota_t = torch.randint(0, self.total_param_size, (1,)).item() # Randomly select a neuron

                # Construct Phi dynamically (only for selected neuron)
                Phi_t = self.Phi[iota_t]
                upsilon_t = upsilon[iota_t]

                #normalization step
                normalization=torch.sum(Phi_t**2)


            # Compute residual error for selected model parameters
            e_t = Phi_t @ omega - upsilon_t

            # Compute normalised stochastic gradient 
            if layer:
                grad = (2/param_size) * Phi_t.transpose(0,1) @ (e_t/normalization)
            else:
                grad = 2 * torch.outer(Phi_t, e_t/normalization)
            grad=grad+2*self.n_train*omega/lam

            # Apply normalized SGD update
            omega -= learning_rate * grad

        # Store the computed influence function
        self.IF_dict["RRInf"] = pd.DataFrame(omega.cpu().transpose(0,1), dtype=float)
        self.time_dict["RRInf"] = time() - start_time


    def save_result(self, run_id=0):
        results={}
        results['runtime']=self.time_dict
        results['influence']=self.IF_dict

        with open(f"./results_{run_id}.pkl",'wb') as file:
            pickle.dump(results, file)