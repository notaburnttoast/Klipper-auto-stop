import matplotlib.pyplot as plt
import ast
import numpy as np
from scipy.signal import savgol_filter

layer = 2
number_of_conv2d = 3
number_of_iterations = [3,3,3]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


x,y,z = [],[],[]
for i in range(layer,number_of_conv2d+layer):
    cx, cz = [], []
    for j in range(number_of_iterations[i-layer]):
        with open(f"Model saves/Science fair/data log/L{layer} N{i} {j+1}.dat", "r") as file:
            nx, nz = ast.literal_eval(file.read())
        cx.append(nx)
        cz.append(nz)
    newx = [0.0 for _ in range(len(cx[0]))]
    newz = [0.0 for _ in range(len(cx[0]))]
    for j in range(len(newx)):
        for k in range(len(cx)):
            newx[j] += cx[k][j]
            newz[j] += cz[k][j]
        newx[j] = newx[j]/len(cx)
        newz[j] = newz[j]/len(cx)
    ny = [float(i) for _ in range(len(newx))]
    x.append(newx)
    y.append(ny)
    for _ in range(0):
        newz = savgol_filter(newz, window_length=75, polyorder=1)
    z.append(newz)

X = np.array(x)
Y = np.array(y)
Z = np.array(z)

ax.set_zlim(zmin=3, zmax=10)
ax.set_ylim(ymin=2,ymax=4)
ax.set_xlim(xmin=0,xmax=20)
surf = ax.plot_surface(X, Y, Z, cmap='magma', edgecolor='none', rstride=1, cstride=1)
cbar = fig.colorbar(surf, pad=0.20)
cbar.set_ticks([4,5,6,7,8,9,10])
ax.set_title("Loss for different number of Conv2d with 2 model layers")
ax.set_xlabel("Epoch")
ax.set_ylabel("Conv2d")
ax.set_zlabel("Loss")

plt.savefig(r"Model saves\Science fair\graphs\current.png", dpi=600)
#plt.show()
