from smartredis import Client
import os
import numpy as np
SSDB = os.getenv('SSDB')
client = Client(address=SSDB,cluster=False)
client.put_tensor('stop-coDB',np.array([1]))
