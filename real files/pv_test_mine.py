"""This file tests the neural network on the test dataset"""


import random
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import torchdiffeq
import torch.nn as n
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import json

#loading the train_data.json and test_data.json
cwd = os.getcwd()
test_data_filepath = os.path.join(cwd, "..", "data", "test_data.json")
with open(test_data_filepath) as test_file:
   test_data_dictionary = json.load(test_file)

test_times = test_data_dictionary["test_times"]
test_force = test_data_dictionary["test_force"]
test_voltage = test_data_dictionary["test_voltage"]
test_acceleration1 = test_data_dictionary["test_acceleration1"]
test_acceleration2 = test_data_dictionary["test_acceleration2"]
test_acceleration3 = test_data_dictionary["test_acceleration3"]
test_size = len(test_force)

#creating a compiled train and test dataset
test_data = [test_force, test_voltage, test_acceleration1, test_acceleration2, test_acceleration3]

#formatting the data into torch tensors
test_data = torch.tensor(test_data, dtype=torch.float32)

test_data = torch.transpose(test_data,0,1)

test_augment = torch.zeros(test_size, 3)
test_data = torch.cat((test_data, test_augment), 1)

#modifying the method to work with the dataset. Consider saving the data at some point

device = 'cpu'
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device_str='cpu'

t_d= torch.tensor(test_times, dtype=torch.float32) #times
tt_tors= test_data[0, :] #initial conditions
tt_tors = tt_tors.reshape((1,-1)) # shaping into a 2D tensor
xy_d = test_data

#defines the loss function
def loss_er(x_pred,x_gt):
    mse_l=torch.norm(x_pred-x_gt,dim=1)
    sum_c=0.0
    for i in range(len(mse_l)):
        sum_c+=(mse_l[i])#*(1/float(i+1))
    return(sum_c/len(mse_l))




#turning the train function into an evaluate function

def evaluate(hidden_size, num_layers):
        print('='*20)
        #print(f'hidden_size={hidden_size}, num_layers={num_layers}')
        global tt_tors, t_d
        #seed = 123
        torch.manual_seed(seed)
        random.seed(seed)
        np.random.seed(seed)
        #f_x the architecture of the base ODE. Input dimension is now 8
        f_x=n.Sequential(
        n.Linear(8, 30),
        n.Tanh(),
        n.Linear(30, 30),
        n.Tanh(),
        n.Linear(30, 30),
        n.Tanh(),
        n.Linear(30, 8)
        )
        #does the forward pass through the base ODE
        def fx(t,x):
            return(f_x(x))

        #forward pass through the INN, inn is defined lower down
        def for_inn(x):
            return(inn(x)[0])
        #reverse pass through INN, inn is defined lower down
        def rev_inn(x):
            return(inn(x,rev=True)[0])
        
        #does an evaluation from initial value to terminal value for all times in t_d i.e a full forward pass of the computation graph
        def linear_val_ode2(init_v,t_d):
            init_v_in=rev_inn(init_v)
            eval_lin=torchdiffeq.odeint(fx,init_v_in,t_d,method='dopri5',atol=1e-5,rtol=1e-5)[:,0,:]#options={'step_size':0.01}
            eval_out=for_inn(eval_lin)
            return(eval_out)
        
        N_DIM = 8
        #defining the subnetwork within the INN
        def subnet_fc(dims_in, dims_out):
            return n.Sequential(n.Linear(dims_in, hidden_size), n.ReLU(),
                                 n.Linear(hidden_size, dims_out))

        #making the inn itself
        inn = Ff.SequenceINN(N_DIM)
        for k in range(num_layers):
            inn.append(Fm.AllInOneBlock, subnet_constructor=subnet_fc,permute_soft=True)

        tt_tors = tt_tors.to(device)
        #t_d = t_d.to(device)
        inn.to(device)
        t_d = t_d.to(device)
        #xy_d_list = torch.stack(xy_d_list).to(device)
        
        #loading the states
        f_x.load_state_dict(torch.load("f_x_base_save_pv.tar"))
        inn.load_state_dict(torch.load("inn2_save_pv.tar"))

        eval_nl=linear_val_ode2(tt_tors,t_d).detach() # computes the estimated times for all t_d
        loss_cur = torch.mean(torch.norm(eval_nl[:,:5]-xy_d[:,:5],dim=1))

        return eval_nl, loss_cur.item()



predictions, test_loss = evaluate(1500, 5) #INN with 5 layers, 1500 neurons in the hidden layer of the subnet
#casting to an np array

predictions = np.array(predictions)
print(f"Test loss: {test_loss}")
predicted_force = predictions[:,0]
predicted_voltage = predictions[:,1]
predicted_acceleration1 = predictions[:,2]
predicted_acceleration2 = predictions[:,3]
predicted_acceleration3 = predictions[:,4]

#create plots with pre-defined labels
fig, ax = plt.subplots()
ax.plot(test_times, test_force, "b", label="Actual")
ax.plot(test_times, predicted_force, "r", label="Predicted")
plt.xlabel("Time")
plt.ylabel("Force")
legend = ax.legend()
plt.grid()
plt.show()
