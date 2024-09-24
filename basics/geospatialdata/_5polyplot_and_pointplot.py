import geoplot as gplt
import geopandas as gpd
import os
import matplotlib.pyplot as plt

# got data from 'https://github.com/ResidentMario/geoplot-data/tree/master'

# cwd = os.getcwd()
# print('working directory: ', cwd)


# Reading the world shapefile
#path = gplt.datasets.get_path(os.getcwd() + '\geoplot-data\world')

os.chdir(os.getcwd() + '\geoplot-data')

''' not working
path = gplt.datasets.get_path('world')
world = gpd.read_file(path)
gplt.polyplot(world)
plt.show()
'''

path = gplt.datasets.get_path("contiguous_usa")
contiguous_usa = gpd.read_file(path)
gplt.polyplot(contiguous_usa)
plt.show()