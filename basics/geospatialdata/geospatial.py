import geopandas as gpd
import matplotlib.pyplot as plt

# Download data from 'https://drive.google.com/drive/folders/1c-TF7X45tAR-gVHmkT4Wbn_1Ol32nKQg'
# Reading the world shapefile
world_data = gpd.read_file('world.shp')
# exporting data to excel
#world_data.to_excel('world_export.xlsx')

print(world_data)
world_data.plot()
plt.show()

world_data = world_data[['NAME', 'geometry']]
print(world_data)
# exporting data to excel
#world_data.to_excel('world_name_geometry_export.xlsx')