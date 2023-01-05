import json
import numpy as np
from matplotlib import pyplot as plt

with open("./data/mock_epoch_loss_data.json") as data_loader:
    epoch_loss_data = json.load(data_loader)

total_epochs = epoch_loss_data["total_epochs"]
epoch_time = epoch_loss_data["epoch_time"]
loss_at_epoch = epoch_loss_data["loss_at_epoch"]

epochs = list(range(total_epochs))
plt.plot(epochs, loss_at_epoch)
plt.show()