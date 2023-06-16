from typing import Union, Tuple

from phi import math
from phi.math import Tensor, tensor

from ..geom._geom import Geometry, _keep_vector
from ..math import wrap, Tensor, expand, Shape, tensor
from ..math.magic import slicing_dict

from ..math._shape import concat_shapes, channel, spatial
from ..geom._transform import embed
from ..field._grid import Grid

import shapely.geometry as gs
import shapely.affinity as  aff
import numpy as np

from ..geom._polygon import AdapterPolygon
from scipy.interpolate import RegularGridInterpolator,  LinearNDInterpolator




def interpolate_on_perimeter(field_grid: Grid, polygon: AdapterPolygon):
    mid_points = polygon.mid_points

    if field_grid.shape.channel.is_empty == True:
        grid_points_x = field_grid.points['x'].numpy('x,y')[:,0]
        grid_points_y = field_grid.points['y'].numpy('x,y')[0,:]
        grid_values = field_grid.values.numpy('x,y')
        interp = RegularGridInterpolator((grid_points_x, grid_points_y), grid_values)
        interp_values_d = [interp(mid_points)]

    else:
        vec_dim = field_grid.shape.channel.size
        interp_values_d = []
        for d in range(vec_dim):
            grid_points_x = field_grid[d].points['x'].numpy('x,y')[:,0]
            grid_points_y = field_grid[d].points['y'].numpy('x,y')[0,:]
            grid_values = field_grid[d].values.numpy('x,y')
            interp = RegularGridInterpolator((grid_points_x, grid_points_y), grid_values)
            interp_values_d.append(interp(mid_points))

    interp_values = np.stack(interp_values_d, axis=1)

    return interp_values



def integrate_on_perimeter(field_grid: Grid, polygon: AdapterPolygon):
    field_on_perimeter = interpolate_on_perimeter(field_grid, polygon)
    edge_normals = polygon.edge_normals

    if field_on_perimeter.shape[1]==1:
        sum_over = field_on_perimeter * edge_normals
    elif field_on_perimeter.shape[1]==2:
        sum_over = np.dot(field_on_perimeter, edge_normals)
    else:
        raise ValueError("Only supports perimeter integration with scalar or 2d vector fields") 
    
    integral = np.sum(sum_over, axis=0)
    return integral
