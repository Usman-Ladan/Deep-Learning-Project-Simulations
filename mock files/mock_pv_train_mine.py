"""
setting up the data. We want to take out 10% of datapoints at random, have the first 75% for training"""

import csv
import random
import torch
import numpy as np
#import matplotlib.pyplot as pl
import torchdiffeq
import torch.nn as n
import FrEIA.framework as Ff
import FrEIA.modules as Fm
import json
"""
use this if you want to generate data uniquely every time

with open("../data/pv.csv") as pv_data:
    reader = csv.DictReader(pv_data)
    force = []
    voltage = []
    acceleration1 = []
    acceleration2 = []
    acceleration3 = []
    
    #cast the strings to floats
    for row in reader:
        force.append(float(row["Force"]))
        voltage.append(float(row["Voltage"]))
        acceleration1.append(float(row["Acceleration1"]))
        acceleration2.append(float(row["Acceleration2"]))
        acceleration3.append(float(row["Acceleration3"]))

total_size = 73728 #number of datapoints we use from pv dataset. Total is 73728
sample_size = int(total_size*0.9)
all_times = list(range(total_size))
sample_times = random.sample(all_times, k=sample_size) #samples 90% of the times
sample_times.sort()

print(len(force))

#making the sample dataset
rand_force = [force[i] for i in sample_times]
rand_voltage = [voltage[i] for i in sample_times]
rand_acceleration1 = [acceleration1[i] for i in sample_times]
rand_acceleration2 = [acceleration2[i] for i in sample_times]
rand_accleration3 = [acceleration3[i] for i in sample_times]

#making the train and test sets
train_size = int(sample_size*0.75)
test_size = sample_size - train_size
train_times = sample_times[:train_size]

train_force = rand_force[:train_size]
train_voltage = rand_voltage[:train_size]
train_acceleration1 = rand_acceleration1[:train_size]
train_acceleration2 = rand_acceleration2[:train_size]
train_acceleration3 = rand_accleration3[:train_size]

test_times = sample_times[train_size:]
test_force = rand_force[train_size:]
test_voltage = rand_voltage[train_size:]
test_acceleration1 = rand_acceleration1[train_size:]
test_acceleration2 = rand_acceleration2[train_size:]
test_acceleration3 = rand_accleration3[train_size:]
"""

#loading the train_data.json files
with open("../data/mock_train_data.json") as train_file:
   train_data_dictionary = json.load(train_file) 

train_times = train_data_dictionary["train_times"]
train_force = train_data_dictionary["train_force"]
train_voltage = train_data_dictionary["train_voltage"]
train_acceleration1 = train_data_dictionary["train_acceleration1"]
train_acceleration2 = train_data_dictionary["train_acceleration2"]
train_acceleration3 = train_data_dictionary["train_acceleration3"]
train_size = len(train_force)

#creating a compiled dataset
train_data = [train_force, train_voltage, train_acceleration1, train_acceleration2, train_acceleration3]

#formatting the data into torch tensors
train_data = torch.tensor(train_data, dtype=torch.float32)

train_data = torch.transpose(train_data,0,1)

train_augment = torch.zeros(train_size, 3)
train_data = torch.cat((train_data, train_augment), 1)

#modifying the method to work with the dataset. Consider saving the data at some point

#this may be supurfluous
device = 'cpu'
seed = 0
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)
device_str='cpu'

t_d= torch.tensor(train_times, dtype=torch.float32) #times
tt_tors= train_data[0, :] #initial conditions
tt_tors = tt_tors.reshape((1,-1)) # shaping into a 2D tensor
xy_d = train_data #expected output

print(f'The shapes of the below are as follows,\ntimes: {t_d.shape}\ninitial_conditions: {tt_tors.shape}\nexpected_outputs: {xy_d.shape}')

#defines the loss function
def loss_er(x_pred,x_gt):
    mse_l=torch.norm(x_pred-x_gt,dim=1)
    sum_c=0.0
    for i in range(len(mse_l)):
        sum_c+=(mse_l[i])#*(1/float(i+1))
    return(sum_c/len(mse_l))




#the training process

def train(hidden_size, num_layers):
        print('='*20)
        #print(f'hidden_size={hidden_size}, num_layers={num_layers}')
        global tt_tors, xy_d_list, t_d
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
        #does the forward pass gthrough the base ODE
        def fx(t,x):
            return(f_x(x))

        #forward pass through the INN, inn is defined lower down
        def for_inn(x):
            return(inn(x)[0])
        #reverse pass through INN, inn is defined lower down
        def rev_inn(x):
            return(inn(x,rev=True)[0])
        
        def rev_mse_inn_eig(rf,x_gt):
            return(torch.mean(torch.norm(rf-x_gt,dim=1)))
        
        def linear_val_ode(w_vec,init_v,t_d):
            init_v_in=rev_inn(init_v)
            eval_lin=eigen_ode__(w_vec,init_v_in,t_d)
            ori_shape = eval_lin.shape
            eval_out=for_inn(eval_lin.reshape(-1, eval_lin.shape[-1]))
            return(eval_out.reshape(ori_shape))
        
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

        #the optimizer optimises the parameters of the INN and Base ODE
        optimizer_comb = torch.optim.Adam(
        [{'params': f_x.parameters(),'lr': 0.0001},{'params': inn.parameters(),
                            'lr': 0.0001}])
        print(sum(p.numel() for p in inn.parameters())) #prints all paremeters in the INN

        startings=tt_tors.clone().detach()

        #Training loop
        import timeit
        epoch_time=[]
        loss_at_epoch=[]
        import tqdm
        from tqdm import trange

        tt_tors = tt_tors.to(device)
        #t_d = t_d.to(device)
        inn.to(device)
        t_d = t_d.to(device)
        #xy_d_list = torch.stack(xy_d_list).to(device)

        print('training INN')
        #for i in trange(0, 5000):
        total_epochs = 500
        for i in trange(0, total_epochs):
            optimizer_comb.zero_grad()
            loss=0.0
            start = timeit.default_timer()
            eval_nl=linear_val_ode2(tt_tors,t_d) # computes the estimated times for all t_d
            
            loss_cur = torch.mean(torch.norm(eval_nl[:,:5]-xy_d[:,:5],dim=1))
            loss+=loss_cur

            loss.backward()
            optimizer_comb.step()
            end = timeit.default_timer()
            epoch_time.append(end-start)
            loss_at_epoch.append(loss.item())
            if(i%10==0):
                print('Combined loss:'+str(i)+': '+str(loss) + f'. epoch time = {epoch_time[-1]}')
        ep_time=np.array(epoch_time)
        print(f'mean train time:{ep_time.mean():.3f} {ep_time.std():.3f}')
        print(f'total train time: {ep_time.sum():.2f}')
        #saving the epoch_times and loss_times
        mock_epoch_loss_data = {"total_epochs": total_epochs, "epoch_time": epoch_time, "loss_at_epoch": loss_at_epoch}
        with open("../data/mock_epoch_loss_data.json", "w") as json_file_loader:
            json.dump(mock_epoch_loss_data, json_file_loader) 
        torch.save(f_x.state_dict(),'mock_f_x_base_save_pv.tar')
        torch.save(inn.state_dict(),'mock_inn2_save_pv.tar')




train(1500, 5) #INN with 5 layers, 1500 neurons in the hidden layer of the subnet
