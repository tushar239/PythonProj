import geopandas as gpd
import matplotlib.pyplot as plt

world_data = gpd.read_file('world.shp')

'''
We can color each country in the world using a head column and cmap. 
To find out head column type “world_data.head()” in console. 
We can choose different color maps(cmap) available in matplotlib. 
In the following code, we have colored countries using plot() arguments column and cmap. 
'''
# for different country names, give different color
world_data.plot(column='NAME', cmap='hsv')
plt.show()

# different colors as per different AREA values
world_data.plot(column='AREA', cmap='hsv')
plt.show()