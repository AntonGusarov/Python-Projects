import numpy as np
import matplotlib.pyplot as plt
data=list()
with open('ca1_step1_input_data.txt', mode='r') as file:
    for x in range(0,2500):
        line = file.readline()
        if '# time_step ' in line:
            line = file.readline()
            l = list()
            for z in range(0, 400):
                line = file.readline()
                line = line.replace(' ', '')
                line = line.replace('\t', '')
                line = line.replace('\n', '')
                line = line.replace(';', ',')
                n = line.split(',')
                n[0] = float(n[0])
                n[1] = float(n[1])
                n[2] = float(n[2])
                n[3] = float(n[3])
                l.append(n)
            data.append(l)
            del l

data = np.array(data)
R = data[0:,0:,0:2]
V = data[0:,0:,2:4]

R = np.swapaxes(R, 1, 2)
V = np.swapaxes(V, 1, 2)


plt.xlim(-1.1)
plt.ylim(-1.1)
plt.axis('square')
plt.scatter(R[0], V[0])
plt.show()
