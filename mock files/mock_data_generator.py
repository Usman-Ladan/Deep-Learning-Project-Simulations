import csv
import os
import random
import json

cwd = os.getcwd() # the path to the current directory
pv_dataset_filepath = os.path.join(cwd, "..", "data", "pv.csv") # the path to the pv dataset

with open(pv_dataset_filepath) as pv_data:
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

total_size = 200 #number of datapoints we use from pv dataset. Total is 73728
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

#we will write a train set and test tes as json files. First, we make the dictionaries

train_json = {"train_times": train_times, "train_force": train_force, "train_voltage": train_voltage,
"train_acceleration1": train_acceleration1, "train_acceleration2": train_acceleration2,
"train_acceleration3": train_acceleration3}

test_json = {"test_times": test_times, "test_force": test_force, "test_voltage": test_voltage,
"test_acceleration1": test_acceleration1, "test_acceleration2": test_acceleration2,
"test_acceleration3": test_acceleration3}

#writing the json files
mock_train_data_filepath = os.path.join(cwd, "..", "data", "mock_train_data.json")
mock_test_data_filepath = os.path.join(cwd, "..", "data", "mock_test_data.json")

with open(mock_train_data_filepath, "w") as train_file:
    json.dump(train_json, train_file)

with open(mock_test_data_filepath, "w") as test_file:
    json.dump(test_json, test_file)