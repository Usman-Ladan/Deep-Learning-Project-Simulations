import json
import os
import numpy as np
from matplotlib import pyplot as plt

cwd = os.getcwd()
epoch_loss_data_filepath = os.path.join(cwd, "..", "data", "epoch_loss_data.json")
with open(epoch_loss_data_filepath) as data_loader:
    epoch_loss_data = json.load(data_loader)

total_epochs = epoch_loss_data["total_epochs"]
epoch_time = epoch_loss_data["epoch_time"]
loss_at_epoch = epoch_loss_data["loss_at_epoch"]
epochs = list(range(total_epochs))

#fixing the epoch_time list to remove the outlier

fixed_epochs = []
for epoch in epoch_time:
    if epoch < 100:
        fixed_epochs.append(epoch)
    else:
        pass

fixed_epochs = np.array(fixed_epochs)
mean_epoch_time = np.mean(fixed_epochs)
std_epoch_time = np.std(fixed_epochs)
print(f"no. of epochs: {len(fixed_epochs)}, mean epoch time: {mean_epoch_time}. standard deviation: {std_epoch_time}")
plt.plot(epochs[:], loss_at_epoch[:], "r-")
plt.grid()
#plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Training loss")
plt.show()