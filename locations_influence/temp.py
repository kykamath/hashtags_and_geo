import matplotlib.pyplot as plt

def plotPointsOnWorldMap(points, blueMarble=False, bkcolor='#85A6D9', returnBaseMapObject = False, pointLabels=[], resolution='l', *args, **kwargs):
    from mpl_toolkits.basemap import Basemap
    m = Basemap(projection='mill', llcrnrlon=-180. ,llcrnrlat=-60, urcrnrlon=180. ,urcrnrlat=80, resolution=resolution)
    if blueMarble: m.bluemarble()
    else:
        m.drawmapboundary(fill_color=bkcolor)
        m.fillcontinents(color='white',lake_color=bkcolor)
        m.drawcoastlines(color='#6D5F47', linewidth=.4)
        m.drawcountries(color='#6D5F47', linewidth=.4)
    
    lats, lngs = zip(*points)
    
    x,y = m(lngs,lats)
    scatterPlot = m.scatter(x, y, zorder = 2, *args, **kwargs)
    for population, xpt, ypt in zip(pointLabels, x, y):
        label_txt = str(population)
        plt.text( xpt, ypt, label_txt, color = 'black', size='small', horizontalalignment='center', verticalalignment='center', zorder = 3)
    if not returnBaseMapObject: return scatterPlot
    else: return (scatterPlot, m)


input_location = [43.644026,-99.755859] # Source point []
locations = [[40.780541,-78.134766], [27.683528,-106.171875], [46.498392,-121.640625]] # List of locations [[lat1, lng1], [lat2, lng2], ... ]
_, m = plotPointsOnWorldMap(locations, resolution= 'l', blueMarble=True, bkcolor='#ffffff', c='#FF00FF', returnBaseMapObject=True, lw = 0)
for location in locations: 
    m.drawgreatcircle(location[1], location[0], input_location[1], input_location[0], color='#FAA31B', lw=1., alpha=0.5)
plotPointsOnWorldMap([input_location], resolution= 'l', blueMarble=True, bkcolor='#ffffff', c='#003CFF', s=40, lw = 0)
plt.show()
#    plt.savefig('output_file.png')
#    plt.clf()