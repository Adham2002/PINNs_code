import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import OrderedDict
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torch.linalg import norm

from data_extracter import AirfoilData
import importlib
import sys
sys.path.append("stored_params")





# implement Stan activation function as a pytorch module
class Stan(nn.Module):
    def __init__(self, layer_size):
        super(Stan, self).__init__()
        # initialise beta for each neuron in a layer
        # neuron is an nn.Parameter so it will be trained through when the PINN makes a backwards pass
        self.beta = nn.Parameter(torch.zeros(layer_size))
 
    # the forward method of an pytorch neuron implements the non linear activation function
    def forward(self, x):
        output = torch.tanh(x) + self.beta * x * torch.tanh(x)  # Stan activation function equation Stan(x) = tanh⁡(x) + β_k^i tanh⁡(x)
        return output





class PINN(nn.Module):
    def __init__(self, NN_architecture, activation_function, loss_term_weights, batch_size, lr, layers, restore_file=None):
        super().__init__()
        
        # initialise equation constants
        # fixed 1/Re
        self._1_div_Re =1/3000000 #* (1/2.75)
        # desnoty is a parameter as it is an unkown constant that cabe optimised for
        self._1_div_density = nn.Parameter(torch.tensor(0.81632, requires_grad=True))
        #self._1_div_density = 0.81632 * (1/1.4)
        
        # initialise PINN architecture
        if NN_architecture == "FNN":
            # FNN is a basic feed forward NN and it's structure is initialised by the initialise_NN method
            # this method outputs a list that is then converted to an OrderedDict and then to a NN
            self.NN = nn.Sequential(OrderedDict(self.initialise_NN(layers, activation_function)))
        
        elif NN_architecture == "SPINN" or NN_architecture == "MSPINN":
            # for both SPINN and MSPINN we need the top and bottom NNs which are each half the size of the FNN
            half_layers = (np.array(layers) / 2).astype(int)
            half_layers[0] = 2
            half_layers[-1] = 3
            
            if NN_architecture == "SPINN":
                # structure of both the top and the bottom NN initialised by the initialise_NN method
                self.top_NN = nn.Sequential(OrderedDict(self.initialise_NN(layers, activation_function)))
                self.bottom_NN = nn.Sequential(OrderedDict(self.initialise_NN(layers, activation_function)))
            
            if NN_architecture == "MSPINN":
                # we remove the last output layer of the top and bottom NNs
                # as we need to define the mixing layer between the secondlast and last layer
                self.top_NN = nn.Sequential(OrderedDict(self.initialise_NN(layers.pop(), activation_function)))
                self.bottom_NN = nn.Sequential(OrderedDict(self.initialise_NN(layers.pop(), activation_function)))
                # nn.Bilinear implements the mixing synapse
                self.mixing_synapse = nn.Bilinear(half_layers[-2], half_layers[-2], 6)
            
        else:
            print("unrecognises neural network type provided please try again")
        
        
        # restore model and optimiser params from restore_file
        if restore_file != None:
            path = os.path.join("stored_params", restore_file+".pth")
            checkpoint = torch.load(path)
            # load in model
            self.load_state_dict(checkpoint["model_state_dict"])
            
            # load optimiser and batch size
            self.optimiser = Adam(self.parameters(), lr=lr)
            # if Stan is used this will restore the Beta parameters of the neurons
            self.optimiser.load_state_dict(checkpoint["optimizer_state_dict"])
            model.optimiser.param_groups[0]['lr'] = lr
            
            # load the NN architecure name and loss term weights
            self.NN_architecture = checkpoint["NN_architecture"]
            self.w = loss_term_weights
            
            # get previous evaluation data for graphing
            self.epochs = checkpoint["epochs"]
            self.data_losses = checkpoint["data_losses"]
            self.eqn_losses = checkpoint["eqn_losses"]
            self.boundary_condition_losses = checkpoint["boundary_condition_losses"]
            self.total_losses = checkpoint["total_losses"]
            self.p_errors = checkpoint["p_errors"]
            self.u_errors = checkpoint["u_errors"]
            self.v_errors = checkpoint["v_errors"]
            self.activation_function = activation_function
            self.batch_size = batch_size
            self.lr = lr
        
        # else initialise NN using auguments given
        else:
            # initilise optimiser with all the parameters and the learning rate
            self.optimiser = Adam(self.parameters(), lr)
            
            # initialise the NN architecure and loss term weights
            self.NN_architecture = NN_architecture
            self.w = loss_term_weights
            
            # initialise variables that will be used for graphing
            self.epochs = []
            self.data_losses = []
            self.eqn_losses = []
            self.boundary_condition_losses = []
            self.total_losses = []
            self.p_errors = []
            self.u_errors = []
            self.v_errors = []
            # these variables are stored here so they can be displayed on the title of graphs
            # they are not used in the creation or training if the PINN
            self.batch_size = batch_size
            self.lr = lr
            if activation_function == nn.Tanh:
                self.activation_function = "tanh"
            else:
                self.activation_function = "Stan"
    
    
    
    # initialises a simple FNN where layers is a list that defines how many nerons are in each layer
    def initialise_NN(self, layers, activation_function):
        # stores synapses and neurons
        synapses_and_neurons = list()
        
        # for every hidden layer synapse except the last we alternate between
        for i in range(0,len(layers)-1):
            # adding a synapse
            synapse = nn.Linear(layers[i],layers[i+1])
            synapses_and_neurons.append(("lin"+str(i+1), synapse))
            
            # and adding a layer of neurons
            if i < len(layers)-2:
                if activation_function == Stan:
                    synapses_and_neurons.append(("Stan"+str(i+1), activation_function(layers[i+1])))
                else:
                    synapses_and_neurons.append(("Tanh"+str(i+1), activation_function()))
                
        return synapses_and_neurons



    # implments forward pass of the PINN
    def forward(self, z_0):
        # for FNN we can simply pass the inouts through the NN and get the outputs
        if self.NN_architecture == "FNN":
            z_L = self.NN(z_0)
            return z_L

        if self.NN_architecture == "SPINN" or self.NN_architecture == "MSPINN":
            top_z_L = self.top_NN(z_0)
            bottom_z_L = self.bottom_NN(z_0)
            
            if self.NN_architecture == "MSPINN":
                # pases the outputs of the last hidden layer of the top and bottom NNs
                # through mixing synapse to the output layer
                mixed_z_L = self.mixing_synapse(combined_z_L)
                return mixed_z_L
            
            combined_z_L = torch.cat((top_z_L, bottom_z_L), dim=1)
            return combined_z_L
    
    
    
    # implements calulating loss and back propogation as well as storing eval data for graphs
    def train(self, data_loader):
        # initialise start time and calculate loss before training so it can be plotted as the datapoint at epoch 0
        start_time = time.time()
        if len(self.epochs) == 0:
            self.epochs.append(0)
            self.evaluate(data_loader.dataset, training=True)
        
        # gradient decent
        for epoch in range(1,101):
            # for batch in dataset
            for coords_batch, D_batch in data_loader:
                # 
                self.optimiser.zero_grad()
                # set requires_grad to True for the imputs so we can partially differentiate with then in thr equation loss
                coords_batch.requires_grad_()
                # forward pass takes a batch of coords and ouputs a batch of flow quatities (a batch_size by 6 matrix)
                # This is much more efficient than a for loop as it uses matrix multiplication and operations
                O_hat_batch = self.forward(coords_batch)
                # slice batch of outputs to get the batch of flow quantities used to calculate data loss
                D_hat_batch = O_hat_batch[:, 0:3]
                
                # apply equations for data, equation and boundary condition loss
                data_loss = torch.mean(torch.norm(D_batch - D_hat_batch, dim=1) **2)
                equation_loss = self.eqn_loss(coords_batch, O_hat_batch)
                boundary_loss = self.boundary_condition_loss(data_loader.dataset.boundary_points, 0.1)
                
                # multiply each by their corresponding loss term weight and add them all together
                loss = self.w[0]*data_loss + self.w[1]*equation_loss + self.w[2]*boundary_loss
                # uses gradient descent to calculate the derivative of the loss with respects to each parameter (dl/dp)
                # then adds this value to p.grad (the stored sum of the gradients)
                loss.backward()
                # the adam optimiser then takes a step for each parameter using its equations
                # it also updates some of its stores values as the sum dl/dp and the sum of (dl/dp)^2
                self.optimiser.step()
            
            # after every epoch we store the eval data so it can be graphed
            if epoch % 1 == 0:
                print(epoch)
            self.epochs.append(len(self.epochs))
            self.evaluate(data_loader.dataset, training=True)
            
            # stopping criteria: if the loss is less than 0.001 or training takes longer than 10 mintes
            if loss < 0.0001:
                print("Success")
                break
            elif (time.time() - start_time) > (10*60):
                print("Training took too long")
                break
        
        # display graphs of the evaluation data against epoch so we can track the conbergence
        self.display_graphs()
        time_taken = time.time() - start_time
        return time_taken
    
    
    
    def eqn_loss(self, coords, O_hat_batch):
        # slice the ouput tensor to get a batch of each flow quantity. E.g. p is 1 batch_size x 1 vector
        p, u, v, uu, uv, vv = O_hat_batch[:, 0], O_hat_batch[:, 1], O_hat_batch[:, 2], O_hat_batch[:, 3], O_hat_batch[:, 4], O_hat_batch[:, 5]
        
        # cakculate the derivatives of the flow quatities with respects to the input coords
        u_derivs = torch.autograd.grad(u, coords, grad_outputs=torch.ones_like(u), create_graph=True, retain_graph=True)[0]
        u_x, u_y = u_derivs[:, 0], u_derivs[:, 1]
        
        v_derivs = torch.autograd.grad(v, coords, grad_outputs=torch.ones_like(v), create_graph=True, retain_graph=True)[0]
        v_x, v_y = v_derivs[:, 0], v_derivs[:, 1]
        
        p_derivs = torch.autograd.grad(p, coords, grad_outputs=torch.ones_like(p), create_graph=True, retain_graph=True)[0]
        p_x, p_y = p_derivs[:, 0], p_derivs[:, 1]
        
        uu_derivs = torch.autograd.grad(uu, coords, grad_outputs=torch.ones_like(uu), create_graph=True, retain_graph=True)[0]
        uu_x, uu_y = uu_derivs[:, 0], uu_derivs[:, 1]
        
        uv_derivs = torch.autograd.grad(uv, coords, grad_outputs=torch.ones_like(uv), create_graph=True, retain_graph=True)[0]
        uv_x, uv_y = uv_derivs[:, 0], uv_derivs[:, 1]
        
        vv_derivs = torch.autograd.grad(vv, coords, grad_outputs=torch.ones_like(vv), create_graph=True, retain_graph=True)[0]
        vv_x, vv_y = vv_derivs[:, 0], vv_derivs[:, 1]
        
        # calculate the doube derivatives
        u_doublex_derivs = torch.autograd.grad(u_x, coords, grad_outputs=torch.ones_like(u_x), create_graph=True, retain_graph=True)[0]
        u_xx, u_xy = u_doublex_derivs[:, 0], u_doublex_derivs[:, 1]
        
        u_doubley_derivs = torch.autograd.grad(u_y, coords, grad_outputs=torch.ones_like(u_y), create_graph=True, retain_graph=True)[0]
        u_yx, u_yy = u_doubley_derivs[:, 0], u_doubley_derivs[:, 1]
        
        v_doublex_derivs = torch.autograd.grad(v_x, coords, grad_outputs=torch.ones_like(v_x), create_graph=True, retain_graph=True)[0]
        v_xx, v_xy = v_doublex_derivs[:, 0], v_doublex_derivs[:, 1]
        
        v_doubley_derivs = torch.autograd.grad(v_y, coords, grad_outputs=torch.ones_like(v_y), create_graph=True, retain_graph=True)[0]
        v_yx, v_yy = v_doubley_derivs[:, 0], v_doubley_derivs[:, 1]
        
        # calculate the equation residuals 
        f1 = u_x + v_y
        f2 = (u*u_x) + (v*u_y) + uu_x + uv_y + (self._1_div_density*p_x) - (self._1_div_Re*(u_xx + u_yy))
        f3 = (u*v_x) + (v*v_y) + uv_x + vv_y + (self._1_div_density*p_y) - (self._1_div_Re*(v_xx + v_yy))
        # sum the residuals to get the equation loss
        equation_loss = torch.mean(f1**2 + f2**2 + f3**2)
        
        return equation_loss
    
    
    
    # calculates the boundary condition loss for Nb the boundary points
    # where Nb = batch_size * sample_frac
    def boundary_condition_loss(self, boundary_points, sample_frac):
        # randomly sample boundary points
        Nb = min(int(np.ceil(self.batch_size * sample_frac)), len(boundary_points))
        indicies = torch.randperm(len(boundary_points))[:Nb]
        sampled_boundary_points = boundary_points[indicies]
        
        # calculate boundary loss
        boundary_conditons_loss = 0
        O_hat = self.forward(sampled_boundary_points)
        u, v = O_hat[:, 1], O_hat[:, 2]
        return torch.mean(u**2 + v**2)



    # calculates the PINNs evaluation metrics on a given dataset
    def evaluate(self, dataset, training=False):
        # This code is very similar to the train method code where the loss is calculated
        coords = dataset.coords
        D = dataset.D
        
        O_hat = self.forward(coords)
        D_hat = O_hat[:, 0:3]
        
        # except we convert the tensors to floats so we can plot the values
        data_loss = float(torch.mean(torch.norm(D - D_hat, dim=1)**2))
        equation_loss = float(self.eqn_loss(coords, O_hat))
        boundary_loss = float(self.boundary_condition_loss(dataset.boundary_points, 1))
        total_loss = self.w[0]*data_loss + self.w[1]*equation_loss + self.w[2]*boundary_loss
        
        # and we calculate the error in each flow quantity
        p = D[:, 0] + (dataset.D_mins[0] / 1000)
        p_error = float(torch.mean(abs((p - D_hat[:, 0]) / p) * 100))
        u = D[:, 1] + (dataset.D_mins[0] / 1000)
        u_error = float(torch.mean(abs((u - D_hat[:, 1]) / u) * 100))
        v = D[:, 2] + (dataset.D_mins[0] / 1000)
        v_error = float(torch.mean(abs((v - D_hat[:, 2]) / v) * 100))
        
        # if this method is called while training the NN it will add the evaluation metric to their respective lists
        # this is called every epoch so the lists will track the evaluation data per epoch so it can be graphed
        if training:
            self.data_losses.append(data_loss)
            self.eqn_losses.append(equation_loss)
            self.boundary_condition_losses.append(boundary_loss)
            self.total_losses.append(total_loss)
            self.p_errors.append(p_error)
            self.u_errors.append(u_error)
            self.v_errors.append(v_error)
        # otherwise the method was called after the PINN has been trained so we just print the evaluation data
        else:
            print(f"% error in p {p_error}")
            print(f"% error in u {u_error}")
            print(f"% error in v {v_error}")
            print(f"MSE of data is {data_loss}")
            print(f"MSE of equation is {equation_loss}")
            print(f"MSE of boundary points is {boundary_loss}")



    # plots graphs of evaluation data thoughout the training process
    def display_graphs(self):
        # create a plot made of 3 subplots
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        # the first plot graphs of equation and boundary condition loss against epoch
        sns.set(style="whitegrid")
        sns.lineplot(x=self.epochs,y=self.eqn_losses, color="red", linewidth=2, label="Equation Loss", ax=axes[0])
        sns.lineplot(x=self.epochs,y=self.boundary_condition_losses, color="blue", linewidth=2, label="Boundary Loss", ax=axes[0])
        axes[0].set_ylabel("Loss")
        axes[0].set_xlabel("epoch")
        axes[0].set_title('Physics Informed')
        
        # the second plot graphs of data loss against epoch
        sns.set(style="whitegrid")
        sns.lineplot(x=self.epochs, y=self.data_losses, color="green", linewidth=2, label="Data Loss", ax=axes[1])
        axes[1].set_ylabel("Loss")
        axes[1].set_xlabel("epoch")
        axes[1].set_title('Data Driven')
        
        # the third plot the flow quantity error against epoch
        sns.set(style="whitegrid")
        sns.lineplot(x=self.epochs, y=self.p_errors, color="green", linewidth=2, label="p error", ax=axes[2])
        sns.lineplot(x=self.epochs,y=self.u_errors, color="red", linewidth=2, label="u error", ax=axes[2])
        sns.lineplot(x=self.epochs,y=self.v_errors, color="blue", linewidth=2, label="v error", ax=axes[2])
        axes[2].set_ylabel("% error")
        axes[2].set_xlabel("epoch")
        axes[2].set_title('Errors')
        
        plt.suptitle(f"lr: {self.lr}, activation: {self.activation_function}, batch_size: {self.batch_size}, loss term weights: {self.w}")
        plt.tight_layout()
        plt.show()
        
        
        
    # stores the PINNs parameters, architecture and graphing information to a .pth file
    # the architecture has to be stores as 
    def store_params(self, filename):
        path = os.path.join("stored_params", filename + ".pth")
        torch.save({
            "model_state_dict": self.state_dict(),
            "optimizer_state_dict": self.optimiser.state_dict(),
            
            "NN_architecture": self.NN_architecture,
            
            "epochs": self.epochs,
            "data_losses": self.data_losses,
            "eqn_losses": self.eqn_losses,
            "boundary_condition_losses": self.boundary_condition_losses,
            "total_losses": self.total_losses,
            "p_errors": self.p_errors,
            "u_errors": self.u_errors,
            "v_errors": self.v_errors,
            }, 
            path)





class Create_train_dataset(Dataset):
    def __init__(self, coords, D, boundary_points):
        coords = torch.tensor(coords)
        self.coords_mins = torch.min(coords, dim=0).values
        self.coords_maxs = torch.max(coords, dim=0).values
        self.coords_ranges = self.coords_maxs - self.coords_mins
        self.coords = (coords - self.coords_mins) / self.coords_ranges
        self.coords.requires_grad_()
        
        boundary_points = torch.tensor(boundary_points)
        self.boundary_points = (boundary_points - self.coords_mins) / self.coords_ranges
        
        D = torch.tensor(D)
        self.D_mins = torch.min(D, dim=0).values
        self.D_maxs = torch.max(D, dim=0).values
        self.D_ranges = self.D_maxs - self.D_mins
        self.D = (D - self.D_mins) / self.D_ranges
    
    def __len__(self):
        return len(self.coords)
  
    def __getitem__(self, index):
        return self.coords[index], self.D[index]





# creates a pytorch dataset for the test testing datasets
# needs mean and std of training dataset so it can normalise the data the same way the training dataset was normalised
class Create_test_dataset(Dataset):
    def __init__(self, coords, D, boundary_points, coords_mins, coords_ranges, D_mins, D_ranges):
        coords = torch.tensor(coords)
        self.coords = (coords - coords_mins) / coords_ranges
        self.coords.requires_grad_()
        
        boundary_points = torch.tensor(boundary_points)
        self.boundary_points = (boundary_points - coords_mins) / coords_ranges
        
        D = torch.tensor(D)
        self.D_mins = D_mins
        self.D = (D - D_mins) / D_ranges
    
    def __len__(self):
        return len(self.coords)
  
    def __getitem__(self, index):
        return self.coords[index], self.D[index]





if __name__ == "__main__":
    TS_size = len(AF.train)
    activation_function = [nn.Tanh, Stan]
    FFN_configs = [[10,          0.0001],    [10,            0.00001], 
                   [100,         0.00001],   [100,           0.00001], 
                   [1000,        0.0001],    [1000,          0.00001], 
                   [.5*TS_size,  0.01],      [.5*TS_size,    0.0001], 
                   [TS_size,     0.01],      [TS_size,       0.0001]]
    for i in range(0, len(FFN_configs)):
        FFN_configs[i].insert(0,nn.Tanh)
        FFN_configs.append(FFN_configs[i])
        FFN_configs[-1][0] = Stan
        
        
    #for config in search space:
    ''' Create PINN with Hyperparameters '''
    # Hyperparameters                     Search space
    # NN_architecture                     {FNN, SPINN, MSPINN}
    # activation_function                 {nn.Tanh, Stan}
    # loss_term_weights                   {[1,1,1], [1,2,1], [2,1,1]}
    # batch_size                          {10, 100, 1000, 5000}
    # lr                                  {0.01, 0.001, 0.0001, 0.00001}
    NN_architecture = "FNN"
    activation_function = nn.Tanh
    loss_term_weights = [1,1,1]
    batch_size = 3000
    #batch_size = 20000
    lr = 0.001
    # layers doesn't change so is not a hyperparameter
    
    layers = [2,20,20,20,20,20,20,20,20,6]
    
    model = PINN(NN_architecture, activation_function, loss_term_weights, batch_size, lr, layers)
    #model = PINN(NN_architecture, activation_function, loss_term_weights, batch_size, lr, layers, restore_file="starting_params_FNN")



    ''' Get Training, Testing Sets'''
    # get flow field data
    AF = AirfoilData("8646")
    
    # create training dataset and get normalising factors for NN inputs and outputs
    training_set = Create_train_dataset(np.array(AF.train[["x", "y"]]), np.array(AF.train[["p", "u", "v"]]), np.array(AF.boundary_points))
    coords_mins, coords_ranges = training_set.coords_mins, training_set.coords_ranges
    D_mins, D_ranges = training_set.D_mins, training_set.D_ranges
   
    # create testing datasets
    testing_set = Create_test_dataset(np.array(AF.test[["x", "y"]]), np.array(AF.test[["p", "u", "v"]]), np.array(AF.boundary_points), coords_mins, coords_ranges, D_mins, D_ranges)
    extrap_testing_set = Create_test_dataset(np.array(AF.extrap_test[["x", "y"]]), np.array(AF.extrap_test[["p", "u", "v"]]), np.array(AF.boundary_points), coords_mins, coords_ranges, D_mins, D_ranges)
    
    
    
    ''' Train PINN and Display Results'''
    # print the initial evaluation data of the PINN on the training set
    print("Initial training set reults")
    model.evaluate(training_set)
    
    
    # put the dataset in a dataloader so it can be batched
    training_set_loader = DataLoader(training_set, batch_size=batch_size)
    
    # train the PINN and output the time taken
    train_time = model.train(training_set_loader)
    print("\n")
    print(f"Training time: {train_time}")
    print("\n")
    
    # print the evaluation data for the training and testing sets
    print("Training set results")
    model.evaluate(training_set)
    print("\n")
    print("Testing set results")
    model.evaluate(testing_set)
    print("\n")
    print("Extrapolation testing set results")
    model.evaluate(extrap_testing_set)
    
    '''
    # once we have found the optimal hyperparams we can test the PINN on 3 different airfoils
    # we test the trained PINN on entire reduced domain
    # then on the entire extrap domain
    
    # we display the optimal solution in paraview
    AF.generate_vtk_file(model, coords_means, coords_stds, D_mean, D_std)
    '''
