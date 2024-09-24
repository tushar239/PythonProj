import geopandas as gpd
import matplotlib.pyplot as plt

world_data = gpd.read_file('world.shp')

# Removing Antarctica from GeoPandas GeoDataframe
world_data = world_data[world_data['NAME'] != 'Antarctica']

world_data.plot()
plt.show()

# to_crs() method transform geometries to a new coordinate reference system.
# Transform all geometries in an active geometry column to a different coordinate reference system.
# The CRS attribute on the current GeoSeries must be set.
# Either CRS or epsg may be specified for output.
# This method will transform all points in all objects.

# EPSG:3857, also known as Web Mercator projection, is a projected CRS commonly used for online maps and web mapping applications.
# It is based on the WGS84 geographic CRS, but uses the Mercator projection to map the Earth's spherical surface onto a flat, two-dimensional plane.
current_crs = world_data.crs
print('current crs: ', current_crs) # EPSG:4326
world_data.to_crs(epsg=3857, inplace=True)
new_crs = world_data.crs
print('new crs: ', new_crs) # EPSG:3857
world_data.to_excel('world_export_crs_3857.xlsx')
world_data.plot()
plt.show()

#summary=world_data.describe()
#print(summary)