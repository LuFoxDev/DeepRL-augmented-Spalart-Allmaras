import numpy as np
from tqdm import tqdm
import sys
import os
import pickle
  
# Getting all memory using os.popen()
total_memory, used_memory, free_memory = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])

io_dict = {}

rnd_size = 100000
# create
for i in tqdm(range(10000)):
    io_dict[i] = np.random.rand(rnd_size)

print(f"saving dict")
with open('filename.pkl', 'wb') as handle:
    pickle.dump(io_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

io_dict = None

print(f"opening dict")
with open('filename.pkl', 'rb') as handle:
    io_dict = pickle.load(handle)

# access
for i in tqdm(range(10000)):
    temp_var = io_dict[i]

# Getting all memory using os.popen()
total_memory_after, used_memory_after, free_memory_after = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
used_memory_by_dict = used_memory_after-used_memory 
print(f"dict used memory: {used_memory_by_dict/1024:.2f} GB")

