import geopandas as gpd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

world_data = gpd.read_file('world.shp')

# Re-calculate the areas in Sq. Km.
world_data['area'] = world_data.area / 1000000

# Adding a legend
world_data.plot(column='area', cmap='hsv', legend=True,
                legend_kwds={'label': "Area of the country (Sq. Km.)"},
                figsize=(7, 7))
plt.show()

# Resizing the legend
fig, ax = plt.subplots(figsize=(10, 10))
divider = make_axes_locatable(ax)

cax = divider.append_axes("right", size="7%", pad=0.1)
world_data.plot(column='area', cmap='hsv', legend=True,
                legend_kwds={'label': "Area of the country (Sq. Km.)"},
                ax=ax, cax=cax)
plt.show()