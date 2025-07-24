import matplotlib
from matplotlib import pyplot as plt

# Load statistics printed by nekRS
def load_statistics(filename):
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
    
    # Find the index of the last line starting with '#'
    last_hash_idx = -1
    for idx, line in enumerate(lines):
        if line.lstrip().startswith('#'):
            last_hash_idx = idx

    # Collect data lines after the last hash line
    for line in lines[last_hash_idx+1:]:
        line = line.strip()
        if line and not line.startswith('#'):
            data.append([float(x) for x in line.replace(',',' ').split()])
    return data

fname = './NekAvgData_1D.csv'
data = load_statistics(fname)
numpts = len(data) // 2
data = data[:numpts]

# Plot average streamwise velocity
y_crd = [y[1] for y in data]
yod = [y/y_crd[-1] for y in y_crd]
u = [u[3] for u in data]
plt.figure(figsize=(6,4))
plt.plot(yod, u, marker='o', linestyle='-')
plt.xlabel('y')
plt.ylabel('u')
plt.title('Streamwise Velocity')
plt.grid()
#plt.xscale('log')
plt.savefig('velocity.png')

