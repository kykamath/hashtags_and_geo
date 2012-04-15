#import matplotlib
#import matplotlib.pyplot as plt
#cm = matplotlib.cm.get_cmap('RdYlBu')
#xy = range(20)
#z = xy
#sc = plt.scatter(xy, xy, c=z, cmap='cool')
#plt.colorbar(sc)
#plt.show()

from subprocess import Popen, PIPE

pipe = Popen("ls -alth", shell=True, stdout=PIPE).stdout
output = pipe.read()
f = open('temp_f', 'w')
f.write(output)