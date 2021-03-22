import csv
import numpy as np
import matplotlib.pyplot as plt

SEQUENCE = 1440

# [ time, low, high, open, close, volume ]
data = np.genfromtxt('eggs.csv', delimiter=',')
data = np.flip(data, axis=0)

# normalize 
data = (data - np.mean(data, axis=0)) / np.std(data, axis=0)
# data = data / data.max(axis=0)

# split into equal sequences
total_rows = data.shape[0]
PARTS = int(total_rows / SEQUENCE) 
missing = (data.shape[0]) % PARTS
data = data[missing:]
data = np.array(np.array_split(data, PARTS))
print(data.shape)

# plot
SAMPLE = 333
fig, axs = plt.subplots(2)
axs[0].plot(data[SAMPLE, :, 1])
axs[1].plot(data[SAMPLE, :, 5])
# axs[2].plot(data[SAMPLE,:,3])
# axs[3].plot(data[SAMPLE,:,4])
# axs[4].plot(data[SAMPLE,:,5])
plt.show()

