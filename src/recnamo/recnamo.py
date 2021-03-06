import numpy as np
from shapely.geometry import Polygon, LinearRing, MultiPoint, MultiLineString, Point
from shapely.ops import cascaded_union, unary_union
from shapely.affinity import translate
from shapely.geometry.collection import GeometryCollection

import xml.etree.ElementTree as ET

from itertools import chain
from shapely.geometry.linestring import LineString

def aa_rect_contain_point_test(rectangles, points):
    min_rect = np.min(rectangles, axis=1)
    max_rect = np.max(rectangles, axis=1)
    points = points.reshape((-1,1))
    
    within = min_rect[:,None,:] < points[None,:,:] & max_rect[:,None,:] > points[None,:,:]
    within = np.all(within, axis=2)
    return within.squeeze()
    

def parse_world(filename):
    e = ET.parse(filename).getroot()
    polygons = []
    for ce in e.findall('obstacle'):
        vertex = [(float(ve.attrib['x']), float(ve.attrib['y']))
                         for ve in ce.findall('vertex')]
        polygons.append(vertex)  
    re= e.find('rangex')
    rangex = ( float(re.attrib['min']), float(re.attrib['max']) )

    re= e.find('rangey')
    rangey = ( float(re.attrib['min']), float(re.attrib['max']) )

    return ((np.array([rangex[0], rangey[0]]),
             np.array([rangex[1], rangey[1]])),
            polygons)
    
    
def aa_get_sweep(rectangle, displacement):
    points = np.array(rectangle.exterior.coords) + np.array(displacement)[None,:]
    return MultiPoint(list(chain(rectangle.exterior.coords, points))).convex_hull

def compute_config_obst(agent, polys, signs= (-1,1)):
    if len(polys) > 1:
        return [ compute_config_obst(agent, [p]) for p in polys]
    
    Averts = np.array(agent.exterior.coords)
    Averts -= Averts[0][None,:]
    Bverts = np.array(polys[0].exterior.coords)
    verts = []
    for i in range(Bverts.shape[0]):
        verts.append(signs[1]*Bverts[i:i+1,:] + signs[0]*Averts)
        
    return MultiPoint(np.vstack(verts)).convex_hull

def compute_config_all_obst(agent, polys, signs= (-1,1)):
    return [ compute_config_obst(agent, [p]) for p in polys]
    
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
        
        self.poly = poly
        self.config_poly = compute_config_all_obst(self.agent, poly)
        self.config_objects = unary_union(self.config_poly)
        
        self.obs_config = { i : compute_config_all_obst(self.poly[i], self.poly[:i] + self.poly[i+1:]) for i in range(len(self.poly))}
#         self.union_obs_config = { k : unary_union(v) for k, v in self.obs_config.iteritems()}
        
        
        self.objects = unary_union(poly)
        self.last_sweep = None
        self.intersect = None
        self.config_intersect = None
        self.agent = translate(self.agent, 
                                   xoff = agent_start[0], 
                                   yoff = agent_start[1])
        self.attached = None
    def move(self, displacement):
        sweep = aa_get_sweep(self.agent, displacement)
        
        if self.attached is not None:
            poly = self.poly[:self.attached] + self.poly[self.attached+1:]
            config_obstacle = unary_union(self.config_poly[:self.attached] + self.config_poly[self.attached+1:])
        else:
            poly = self.poly
            config_obstacle = self.config_objects
        
        self.intersect = unary_union(poly).intersection(sweep)
        self.config_intersect = config_obstacle.intersection(LineString([self.agent.exterior.coords[0],
                                                                               np.array(self.agent.exterior.coords[0]) + np.array(displacement)]))
        
#         print self.config_intersect
        
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
            
        if self.attached is not None:
            index = self.attached
            poly_intersect = unary_union(self.obs_config[index]).intersection(LineString([self.poly[index].exterior.coords[0],
                                                                               np.array(self.poly[index].exterior.coords[0]) + np.array(displacement)]))
        
#             print 'obs', poly_intersect
            if poly_intersect.is_empty or poly_intersect.geom_type == 'Point':
                pass
            else:
                if poly_intersect.geom_type in ('GeometryCollection', 'MultiLineString'):
                    points = np.vstack([ np.array(p.coords) for p in poly_intersect if p.geom_type == 'LineString'])
                elif poly_intersect.geom_type == 'LineString':
                    points = np.array(poly_intersect.coords)
                i = np.argmin(np.linalg.norm(points - np.array(self.poly[index].exterior.coords[0])[None,:], axis=1))
                point = points[i]
                obs_pos = point
                displacement = point - np.array(self.poly[index].exterior.coords[0])
            self.move_obs(index, displacement)
            
        self.agent = translate(self.agent, 
                       xoff = displacement[0], 
                       yoff = displacement[1])
        self.last_sweep = sweep
        
    def move_obs(self, index, displacement):
        for i in range(0,index):
            self.obs_config[i][index-1] = translate(self.obs_config[i][index-1], 
                                                       xoff = displacement[0], 
                                                       yoff = displacement[1])
        for i in range(index+1, len(self.poly)):
            self.obs_config[i][index] = translate(self.obs_config[i][index], 
                                                   xoff = displacement[0], 
                                                   yoff = displacement[1])
        self.config_poly[index] = translate(self.config_poly[index], 
                                           xoff = displacement[0], 
                                           yoff = displacement[1])
        self.config_objects = unary_union(self.config_poly)
        self.poly[index] = translate(self.poly[index], 
                                       xoff = displacement[0], 
                                       yoff = displacement[1])
        self.objects = unary_union(self.poly)

    def move_agent_to(self, pos):
        end_point = np.zeros(2) + self.agent_size
        self.agent = Polygon([pos,
                         (end_point[0], pos[1]),
                         end_point,
                         (pos[0], end_point[1])])

    def toggle_grab(self):
        if self.attached is None:
            distance_to_poly = np.empty(len(self.poly))
            for i in xrange(len(self.poly)):
                distance_to_poly[i] = self.poly[i].distance(self.agent)
                
                                                   
            i = np.argmin(distance_to_poly)
            if distance_to_poly[i]<= self.EPS:
                self.attached = i
        else:
            self.attached = None
            
    def is_valid_state(self, state):
        s = self.state
        self.state = state
        
        if self.attached is not None:
            config_obstacle = unary_union(self.config_poly[:self.attached] + self.config_poly[self.attached+1:])
        else:
            config_obstacle = self.config_objects
        collide = config_obstacle.intersects(Point(self.agent.exterior.coords[0]))
        
        for i in xrange(len(self.poly)):
            if collide:
                break
            collide |= unary_union(self.obs_config[i]).intersects(Point(self.poly[i].exterior.coords[0]))
            
        self.state = s
            
        return not collide
    
    @property
    def state(self):
        state = np.hstack(chain([p.exterior.coords[0] for p in self.poly],
                                [self.agent.exterior.coords[0]],
                                [0 if self.attached is None else 1]))
        return state
    
    @state.setter
    def state(self, value):
        d_state = value - self.state
        
        self.agent = translate(self.agent, d_state[-3], d_state[-2])
        
        for i in xrange(len(self.poly)):
            self.move_obs(i, d_state[i*2:(i+1)*2])
            
        
        
        # if attach bit is active, set the agent attached to the closest polygon
        self.attached = None
        if value[-1] > 0.0:
            self.toggle_grab()
        
            