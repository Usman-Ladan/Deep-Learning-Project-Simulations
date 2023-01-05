import json
import os
import numpy as np
from matplotlib import pyplot as plt

cwd = os.getcwd()
mock_epoch_loss_data_filepath = os.path.join(cwd, "..", "data", "mock_epoch_loss_data.json")
with open(mock_epoch_loss_data_filepath) as data_loader:
    epoch_loss_data = json.load(data_loader)

total_epochs = epoch_loss_data["total_epochs"]
epoch_time = epoch_loss_data["epoch_time"]
loss_at_epoch = epoch_loss_data["loss_at_epoch"]
epochs = list(range(total_epochs))


mean_epoch_time = np.mean(epoch_time)
std_epoch_time = np.std(epoch_time)
print(f"no. of epochs: {len(epoch_time)}, mean epoch time: {mean_epoch_time}. standard deviation: {std_epoch_time}")
plt.plot(epochs[:], loss_at_epoch[:], "r-")
plt.grid()
# plt.yscale("log")
plt.xlabel("Epochs")
plt.ylabel("Training loss")
plt.show()