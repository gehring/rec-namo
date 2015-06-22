import numpy as np
from shapely.geometry import Polygon, LinearRing, MultiPoint
from shapely.ops import cascaded_union, unary_union
from shapely.affinity import translate
from shapely.geometry.collection import GeometryCollection

from itertools import chain

def aa_rect_contain_point_test(rectangles, points):
    min_rect = np.min(rectangles, axis=1)
    max_rect = np.max(rectangles, axis=1)
    points = points.reshape((-1,1))
    
    within = min_rect[:,None,:] < points[None,:,:] & max_rect[:,None,:] > points[None,:,:]
    within = np.all(within, axis=2)
    return within.squeeze()
    
def aa_get_sweep(rectangle, displacement):
    displacement = np.array(displacement)
    if np.all(displacement == 0.0):
        return rectangle
    
    if displacement[0] >= 0.0 and displacement[1] >= 0.0:
        i = 2
    elif displacement[0] >= 0.0 and displacement[1] < 0.0:
        i = 1
    elif displacement[0] < 0.0 and displacement[1] >= 0.0:
        i = 3
    elif displacement[0] < 0.0 and displacement[1] < 0.0:
        i = 0
        
    
    
    points = np.array(rectangle.exterior.coords) + np.array(displacement)[None,:]
    return MultiPoint(list(chain(rectangle.exterior.coords, points))).convex_hull
    
class RectNamo(object):
    
    DEFAULT_AGENT_SIZE = (1.,1.)
    DEFAULT_AGENT_START = (0., 0.)
    
    def __init__(self, 
                 rectangles, 
                 state_range, 
                 agent_start = None,
                 agent_size = None):
        poly = []
        for rect in rectangles:
            poly.append(Polygon(rect))
            
        self.config_range = state_range
        limits = LinearRing([state_range[0], 
                            (state_range[1][0], state_range[0][1]),
                            state_range[1],
                            (state_range[0][0], state_range[1][1])])
        
        if agent_start is None:
            agent_start = np.array(self.DEFAULT_AGENT_START)
        else:
            agent_start = np.array(agent_start)
            
        if agent_size is None:
            self.agent_size = np.array(self.DEFAULT_AGENT_SIZE)   
        else:
            self.agent_size = np.array(agent_size)
        
        end_point = agent_start + self.agent_size
        self.agent = Polygon([agent_start,
                         (end_point[0], agent_start[1]),
                         end_point,
                         (agent_start[0], end_point[1])])
        self.objects = unary_union(poly)
        self.last_sweep = None
        self.intersect = None
        self.candidates = None
    def move(self, displacement):
        sweep = aa_get_sweep(self.agent, displacement)
        collision = self.objects.intersects(sweep)
        if not collision:
            self.agent = translate(self.agent, 
                                   xoff = displacement[0], 
                                   yoff = displacement[1])
            self.intersect = None
            self.candidates = None
            
        else:
            self.intersect = self.objects.intersection(sweep)
            if self.intersect.geom_type == 'Polygon':
                intersect = [self.intersect]
            else:
                intersect = self.intersect
                

            inner_poly = []
            for p in intersect:
                for v in p.exterior.coords:
                    agent_start = np.array(v)
                    end_point = agent_start + self.agent_size
                    b1 = Polygon([agent_start,
                                 (end_point[0], agent_start[1]),
                                 end_point,
                                 (agent_start[0], end_point[1])])
                    b2 = translate(b1, xoff= -self.agent_size[0], yoff=0.0)
                    b3 = translate(b1, xoff= 0.0, yoff= -self.agent_size[1])
                    b4 = translate(b1, xoff= -self.agent_size[0], yoff= -self.agent_size[1])
                    inner_poly.extend(filter(lambda p: sweep.contains(p), [b1,b2,b3,b4]))
#                     inner_poly.extend( [b1,b2,b3,b4])

            self.candidates = None if len(inner_poly) == 0 else inner_poly
        self.last_sweep = sweep
        
        

