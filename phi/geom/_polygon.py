from typing import Union, Tuple

from phi import math
from phi.math import Tensor, tensor

from ._geom import Geometry, _keep_vector
from ..math import wrap, Tensor, expand, Shape, tensor
from ..math.magic import slicing_dict

from ..math._shape import concat_shapes, channel, spatial

import shapely.geometry as gs
import shapely.affinity as  aff
import numpy as np

# import matplotlib.pyplot as plt

class AdapterPolygon(Geometry):
    """
    Adapter class to define an 2D polygon using Shapely.
    Polygon is defined through vertices

    """

    def __init__(self, vertices: Tuple[Tensor]):
        """
        Args:
            vertices: list of tensors containing the coordinates of the polygon vertices of format (x,y). 
            The vertices should provided in counterclockwise order from the "ideal" center of the polygon.
            This is to allow a correct numerical integration across the perimeter.

        """
        # assert 
        assert len(vertices)>2, f"A Polygon must have at least 3 vertices, but len(vertices)={len(vertices>2)}"
        # for vi, v in enumerate(vertices):
        #     assert isinstance(v, Tensor), f"vertex {vi} must be a Tensor but is of type {type(v)}"
        self.shapely = gs.Polygon(shell=vertices) # Adaptee: shapely class that is to adapt


    @property
    def center(self) -> Tensor:
        return wrap(tensor(list(self.shapely.centroid.coords)[0]), channel(vector='x,y'))
    
    @property
    def vertices(self) -> Tensor:
        return wrap(list(self.shapely.exterior.coords[:-1]), channel(vector='x,y'))


    @property
    def shape(self):
        vrt = self.vertices
        # if any(self.vertices) is None:
        #     return None
        return vrt.shape
    

    @property
    def volume(self) -> math.Tensor:
        """
        Returns volume at different dimension. Only perimeter and area are available for now.
        """
        if self.spatial_rank == 1:
            return self.shapely.length
        
        elif self.spatial_rank == 2:
            return self.shapely.area
        
        elif self.spatial_rank == 3:
            return NotImplementedError()
        
        else:
            raise NotImplementedError()
            # n = self.spatial_rank
            # return math.pi ** (n // 2) / math.faculty(math.ceil(n / 2)) * self._radius ** n

    @property
    def shape_type(self) -> Tensor:
        return math.tensor('S')
    

    
    def lies_inside(self, location):
        location = location.numpy('vector,x,y')
        lies_in = np.zeros(location.shape[-2:], dtype=np.bool_)
        for yi in range(location.shape[-1]):
            for xi in range(location.shape[-2]):
                loc = gs.Point(location[:, xi, yi])
                lies_in[xi, yi] = self.shapely.contains(loc)
        lies_in = tensor(lies_in, spatial(x=lies_in.shape[0], y=lies_in.shape[1]))
        # lies_in = math.any(lies_in, self.shape.instance)
        return lies_in
    


    def approximate_signed_distance(self, location: Union[Tensor, tuple]):
        """
        Computes the exact distance from location to the closest point on the sphere.
        Very close to the sphere center, the distance takes a constant value.

        Args:
          location: float tensor of shape (batch_size, ..., rank)

        Returns:
          float tensor of shape (*location.shape[:-1], 1).

        """
        location = location.numpy('vector,x,y')
        dist = np.zeros(location.shape[-2:], dtype=np.bool_)
        for yi in range(location.shape[-1]):
            for xi in range(location.shape[-2]):
                loc = gs.Point(location[:, xi, yi])
                dist[xi, yi] = math.round(self.shapely.exterior.distance(loc))
        dist = tensor(dist, spatial(x=dist.shape[0], y=dist.shape[1]))
        return dist

    def sample_uniform(self, *shape: math.Shape):
        raise NotImplementedError('Not yet implemented')  # ToDo

    def bounding_radius(self):
        """
        Returns the radius of a Sphere object that fully encloses this geometry.
        The sphere is centered at the center of this geometry.

        :return: radius of type float

        Args:

        Returns:

        """
        distances = []
        for vertex in list(self.shapely.exterior.coords):
            distance = self.shapely.centroid.distance(gs.Point(vertex))
            distances.append(distance)
        return wrap(max(distances))

    def bounding_half_extent(self):
        return expand(self.radius, self._center.shape.only('vector'))

    def at(self, center: Tensor) -> 'Geometry':
        translation = center - self.center
        self.shapely = aff.translate(self.shapely, xoff=translation[0] , yoff=translation[1])
        return self

    def rotated(self, angle, origin='centroid'):
        self.shapely = aff.rotate(self.shapely, angle=angle, origin=origin)
        return self

    def scaled(self, factor: Union[float, Tensor]) -> 'Geometry':
        self.shapely = aff.scale(self.shapely, xfact=factor[0], yfact=factor[1])
        return self

    def __variable_attrs__(self):
        return ['vertices']

    def __getitem__(self, item):
        return NotImplementedError()

    def push(self, positions: Tensor, outward: bool = True, shift_amount: float = 0) -> Tensor:
        raise NotImplementedError()

    def __hash__(self):
        return sum([hash(v) for v in self.vertices])
    
    @property
    def mid_points(self):
        vrt = self.vertices.numpy()
        vrt_n1 = vrt
        vrt_n2 = np.roll(vrt, 1, axis=0)
        mid_points_x = ((vrt_n1[:,0] + vrt_n2[:,0])/2).reshape(-1,1)
        mid_points_y = ((vrt_n1[:,1] + vrt_n2[:,1])/2).reshape(-1,1)
        mid_points = np.hstack((mid_points_x, mid_points_y))
        return mid_points
    
    @property
    def edge_normals(self):
        """
        Computes normal versors to the edges of the polygon.
        Warning: polygon vertices must be defined in clockwise order 
        to point the versors towards the outside
        """
        vrt = self.vertices.numpy()
        vrt_n1 = vrt
        vrt_n2 = np.roll(vrt, 1, axis=0)
        diff_vec = vrt_n1 - vrt_n2
        diff_vec = np.where(diff_vec==0, 1e-7, diff_vec)
        angles = np.arctan(diff_vec[:,1]/diff_vec[:,0])
        norm_angles = angles - (np.pi / 2)
        normal_x = np.cos(norm_angles).reshape(-1,1)
        normal_y = np.sin(norm_angles).reshape(-1,1)
        return np.hstack((normal_x, normal_y))


    @property
    def perimeter(self):
        return self.shapely.length



class ParametricPolygon(AdapterPolygon):

    def __init__(self, parameters_dict):
        self.parameters_dict = parameters_dict
        vrt = self.compute_and_check_vertices()
        super().__init__(vertices=vrt)
        
    def compute_vertices(self):
        return NotImplementedError()
    
    def compute_and_check_vertices(self):
        vert = self.compute_vertices()
        assert isinstance(vert, list), f"Expected vertices to be a list of lists, but vert={vert}"
        return vert






class airfoil_naca_4digit(ParametricPolygon):

    def __init__(self, c=1.0, m=0.05, p=0.4, t=0.12, num_points=100):
        """
        Naca 4gidit airfoil shape
        Args:
            c = 1.0  # Chord length
            m = 0.05  # Maximum camber
            p = 0.4  # Location of maximum camber
            t = 0.12  # Thickness
            num_points = 100

        """
        self.parametric_dict = {
            'c' : c,
            'm' : m,
            'p' : p,
            't' : t,
            'num_points' :num_points
        }
        super().__init__(self.parametric_dict)


    def compute_vertices(self):
        c = self.parametric_dict['c']
        m = self.parametric_dict['m']
        p = self.parametric_dict['p']
        t = self.parametric_dict['t']
        num_points = self.parametric_dict['num_points']

        x = np.linspace(0, c, num_points)
        
        y_c = (m / p**2) * (2 * p * (x / c) - (x / c)**2)
        thickness = (t / 0.2) * c * (0.2969 * np.sqrt(x / c) - 0.126 * (x / c) - 0.3516 * (x / c)**2 + 0.2843 * (x / c)**3 - 0.1015 * (x / c)**4)
        upper_surface = y_c + thickness
        lower_surface = y_c - thickness
        x_all = np.concatenate((x, np.flip(x)))
        y_all = np.concatenate((upper_surface, np.flip(lower_surface)))
        vertices = np.stack((x_all, y_all)).T
        return vertices.tolist()