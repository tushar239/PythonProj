import geopandas as gpd
import matplotlib.pyplot as plt

# Download data from 'https://drive.google.com/drive/folders/1c-TF7X45tAR-gVHmkT4Wbn_1Ol32nKQg'
# Reading the world shapefile
world_data = gpd.read_file('world.shp')
# exporting data to excel
#world_data.to_excel('world_export.xlsx')
print(type(world_data)) # GeoDataFrame
print(world_data)
world_data.plot()
plt.show()

# you can choose specific Geoseries by data[[‘attribute 1’, ‘attribute 2’]]
world_data = world_data[['NAME', 'geometry']]
print(world_data)

# Calculating the area of each country
world_data['area'] = world_data.area

# exporting data to excel
world_data.to_excel('world_name_geometry_export.xlsx')

# We can remove a specific element from the Geoseries.
# Here we are removing the continent named “Antarctica” from the “Name” Geoseries.
# Removing Antarctica from GeoPandas GeoDataframe
world_data = world_data[world_data['NAME'] != 'Antarctica']
# OR
# world_data = world_data.loc[(world_data.NAME != 'Antarctica')]
world_data.plot()
plt.show()

#Visualizing a specific Country
#We can visualize/plot a specific country by selecting it. In the below example, we are selecting “India” from the “NAME” column.

world_data[world_data.NAME=="India"].plot()
plt.show()