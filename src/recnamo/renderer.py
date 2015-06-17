import pyglet
import numpy as np

from itertools import chain

from shapely.geometry import Polygon, MultiPolygon
from shapely.geometry.collection import GeometryCollection

class line_point_group(pyglet.graphics.Group):
    def __init__(self, line_width, point_width,  parent = None):
        super(line_point_group, self).__init__(parent=parent)
        self.line_width = line_width
        self.point_width = point_width

    def set_state(self):
        pyglet.gl.glLineWidth(self.line_width)
        pyglet.gl.glPointSize(self.point_width)


class Renderer(object):
    def __init__(self, **kargs):
        render_call = kargs['render_call']
        del kargs['render_call']
        self.batch = render_call(**kargs)

    def draw(self):
        self.batch.draw()



def Enviornment_draw(environment,
                     obs_color = (200, 200, 200, 255),
                     sweep_color = (260, 200, 200, 255),
                     edge_width = 2.0,
                     point_width = 3.0):

    batch = pyglet.graphics.Batch()
    obs =  environment.objects
    sweep_group = line_point_group(line_width = edge_width,
                                  point_width = point_width)
    group = line_point_group(line_width = edge_width,
                                  point_width = point_width,
                                  parent=sweep_group)
    
    if environment.last_sweep is not None:
        add_polygon_render(environment.last_sweep, sweep_group, batch, sweep_color)
    add_polygon_render(obs, group, batch, obs_color)
    return batch

def add_polygon_render(poly, group, batch, color):
    if isinstance(poly, MultiPolygon) or isinstance(poly, GeometryCollection):
        for g in poly.geoms:
            add_polygon_render(g, group, batch, color)
    elif isinstance(poly, Polygon):
        vertices = poly.exterior.coords
        index = [ (i,i) for i in xrange(1, len(vertices))] + [[0]]
        edges = [x for i in chain([0], *index) for x in vertices[i]]
        batch.add(len(edges)/2, pyglet.gl.GL_LINES, group,
                                     ('v2f', edges),
                                     ('c4B', color*(len(edges)/2)))
    else:
        batch.add(1, pyglet.gl.GL_POINTS, group,
                    ('v2f', poly.coord),
                    ('c4B', color))

def set_projection(environment, width, height):
        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()

        rangex = (environment.config_range[0][0], environment.config_range[1][0])
        rangey = (environment.config_range[0][1], environment.config_range[1][1])

        ratio = float(height)/width
        lx = rangex[1] - rangex[0]
        ly = rangey[1] - rangey[0]

        if lx*ratio >= ly:
            dy = lx*ratio - ly
            pyglet.gl.glOrtho(rangex[0], rangex[1], rangey[0]- dy/2, rangey[1]+dy/2, -1, 1)
        else:
            dx = ly/ratio - lx
            pyglet.gl.glOrtho(rangex[0]-dx/2, rangex[1] + dx/2, rangey[0], rangey[1], -1, 1)


        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)

def get_mouse_coord(x, y):
        vp = (pyglet.gl.GLint * 4)()
        mvm = (pyglet.gl.GLdouble * 16)()
        pm = (pyglet.gl.GLdouble * 16)()

        pyglet.gl.glGetIntegerv(pyglet.gl.GL_VIEWPORT, vp)
        pyglet.gl.glGetDoublev(pyglet.gl.GL_MODELVIEW_MATRIX, mvm)
        pyglet.gl.glGetDoublev(pyglet.gl.GL_PROJECTION_MATRIX, pm)

        wx = pyglet.gl.GLdouble()
        wy = pyglet.gl.GLdouble()
        wz = pyglet.gl.GLdouble()

        pyglet.gl.gluUnProject(x, y, 0, mvm, pm, vp, wx, wy, wz)
        mcoord = (wx.value, wy.value)

        return mcoord