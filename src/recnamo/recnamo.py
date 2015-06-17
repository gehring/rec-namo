import numpy as np
from shapely.geometry import Polygon, LinearRing, MultiPoint
from shapely.ops import cascaded_union, unary_union
from shapely.affinity import translate

from itertools import chain

def aa_rect_contain_point_test(rectangles, points):
    min_rect = np.min(rectangles, axis=1)
    max_rect = np.max(rectangles, axis=1)
    points = points.reshape((-1,1))
    
    within = min_rect[:,None,:] < points[None,:,:] & max_rect[:,None,:] > points[None,:,:]
    within = np.all(within, axis=2)
    return within.squeeze()
    
def aa_get_sweep(rectangle, displacement):
    points = np.array(rectangle.exterior.coord) + displacement[None,:]
    return MultiPoint(chain(rectangle.exterior.coord, points)).convex_hull
    
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
            agent_size = np.array(self.DEFAULT_AGENT_SIZE)   
        else:
            agent_size = np.array(agent_size)
        
        end_point = agent_start + agent_size
        self.agent = Polygon([agent_start,
                         (end_point[0], agent_start[1]),
                         end_point,
                         (agent_start[0], end_point[1])])
        self.objects = unary_union(poly)
        self.last_sweep = None
        
        
    def move(self, displacement):
        sweep = aa_get_sweep(self.agent, displacement)
        collision = self.objects.intersects(sweep)
        if not collision:
            self.agent = translate(self.agent, 
                                   xoff = displacement[0], 
                                   yoff = displacement[1])
        self.last_sweep = sweep
        
        

