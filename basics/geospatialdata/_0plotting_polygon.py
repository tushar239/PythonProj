from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import geopandas as gpd

# https://stackoverflow.com/questions/55522395/how-do-i-plot-shapely-polygons-and-objects-using-matplotlib

polygon1 = Polygon([(0,5),
                    (1,1),
                    (3,0),
                    ])
p=gpd.GeoSeries(polygon1)
p.plot()
plt.show()