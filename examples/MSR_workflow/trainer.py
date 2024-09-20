import os
import numpy as np
from time import sleep

from smartredis import Client

LIST_NAME = 'training_list'

# Initialize SR Client
SSDB = os.getenv('SSDB')
client = Client(address=SSDB,cluster=False)
print('Initialized client\n',flush=True)

# Every so often tead the training data list and print it
max_iter = 0
count = 0
past_list_length = 0
while count<=10 and max_iter<50:
    sleep(60)
    max_iter+=1
    list_length = client.get_list_length(LIST_NAME)
    print(f'Read list with length {list_length}',flush=True)
    if list_length == 0:
        continue
    else:
        datasets = client.get_datasets_from_list(LIST_NAME)
        data = np.zeros((len(datasets),2))
        for i in range(len(datasets)):
            data[i] = datasets[i].get_tensor('train')
        print('Current training data: \n',data,'\n',flush=True)
        if list_length == past_list_length:
            count+=1
        else:
            past_list_length = list_length
            count = 0

# Exit
print('\nExiting ...',flush=True)



