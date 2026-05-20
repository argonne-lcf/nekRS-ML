from time import sleep
from smartredis import Client
client = Client(cluster=False)
while True:
    if client.key_exists('stop-coDB'):
        print('Found stop-coDB',flush=True)
        break
    else:
        sleep(5)
