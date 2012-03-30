import matplotlib
import matplotlib.pyplot as plt
cm = matplotlib.cm.get_cmap('RdYlBu')
xy = range(20)
z = xy
sc = plt.scatter(xy, xy, c=z, cmap='cool')
plt.colorbar(sc)
plt.show()