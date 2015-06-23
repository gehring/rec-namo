import numpy as np
from shapely.geometry import Polygon, LinearRing, MultiPoint, MultiLineString
from shapely.ops import cascaded_union, unary_union
from shapely.affinity import translate
from shapely.geometry.collection import GeometryCollection

from itertools import chain
from shapely.geometry.linestring import LineString

def aa_rect_contain_point_test(rectangles, points):
    min_rect = np.min(rectangles, axis=1)
    max_rect = np.max(rectangles, axis=1)
    points = points.reshape((-1,1))
    
    within = min_rect[:,None,:] < points[None,:,:] & max_rect[:,None,:] > points[None,:,:]
    within = np.all(within, axis=2)
    return within.squeeze()
    
def aa_get_sweep(rectangle, displacement):
    points = np.array(rectangle.exterior.coords) + np.array(displacement)[None,:]
    return MultiPoint(list(chain(rectangle.exterior.coords, points))).convex_hull

def compute_config_obst(agent, polys, signs= (-1,1)):
    if len(polys) > 1:
        return [ compute_config_obst(agent, [p]) for p in polys]
    
    Averts = np.array(agent.exterior.coords)
    Bverts = np.array(polys[0].exterior.coords)
    verts = []
    for i in range(Bverts.shape[0]):
        verts.append(signs[1]*Bverts[i:i+1,:] + signs[0]*Averts)
#     for v in verts: 
#         v[3,:]=1.0
        
    return MultiPoint(np.vstack(verts)).convex_hull
    
class RectNamo(object):
    
    DEFAULT_AGENT_SIZE = (1.,1.)
    DEFAULT_AGENT_START = (0., 0.)
    EPS = 1e-10
    
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
        
        end_point = np.zeros(2) + self.agent_size
        self.agent = Polygon([agent_start,
                         (end_point[0], agent_start[1]),
                         end_point,
                         (agent_start[0], end_point[1])])
        
        
        self.config_poly = compute_config_obst(self.agent, poly)
        self.config_objects = unary_union(self.config_poly)
        self.objects = unary_union(poly)
        self.last_sweep = None
        self.intersect = None
        self.config_intersect = None
        self.agent = translate(self.agent, 
                                   xoff = agent_start[0], 
                                   yoff = agent_start[1])
    def move(self, displacement):
        sweep = aa_get_sweep(self.agent, displacement)
        
        self.intersect = self.objects.intersection(sweep)
        self.config_intersect = self.config_objects.intersection(LineString([self.agent.exterior.coords[0],
                                                                               np.array(self.agent.exterior.coords[0]) + np.array(displacement)]))
        
        if self.config_intersect.is_empty or self.config_intersect.geom_type == 'Point':
            self.intersect = None
            self.config_intersect = None
        else:
            if self.config_intersect.geom_type in ('GeometryCollection', 'MultiLineString'):
                points = np.vstack([ np.array(p.coords) for p in self.config_intersect if p.geom_type == 'LineString'])
            elif self.config_intersect.geom_type == 'LineString':
                points = np.array(self.config_intersect.coords)
            i = np.argmin(np.linalg.norm(points - np.array(self.agent.exterior.coords[0])[None,:], axis=1))
            point = points[i]
            displacement = point - np.array(self.agent.exterior.coords[0])
            
        self.agent = translate(self.agent, 
                       xoff = displacement[0], 
                       yoff = displacement[1])
        self.last_sweep = sweep
        
        
