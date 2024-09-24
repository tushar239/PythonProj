from shapely.geometry import Polygon
import matplotlib.pyplot as plt
import geopandas as gpd

polygon1 = Polygon([(0,5),
                    (1,1),
                    (3,0),
                    ])
p=gpd.GeoSeries(polygon1)
p.plot()
plt.show()