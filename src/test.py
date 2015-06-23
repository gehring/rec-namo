import pyglet
import numpy as np

from recnamo.renderer import Renderer, Enviornment_draw, get_mouse_coord, set_projection
from recnamo.recnamo import RectNamo


# CREATE ENVIRONMENT

state_range = [np.ones(2)*-3, np.ones(2) *3]

rect = [[np.array([1,2]),
         np.array([3,2]),
         np.array([3,3]),
         np.array([1,3])]]

rect += [[ v + 2 for v in rect[0]]]
envi = RectNamo(rect, state_range)



configTemp = pyglet.gl.Config(sample_buffers=1,
    samples=4,
    double_buffer=True,
    alpha_size=0)

platform = pyglet.window.get_platform()
display = platform.get_default_display()
screen = display.get_default_screen()

try:
    config= screen.get_best_config(configTemp)
except:
    config=pyglet.gl.Config(double_buffer=True)

window = pyglet.window.Window(config=config, resizable=True)


env_renderer = Renderer(environment=envi, render_call=Enviornment_draw)


@window.event
def on_draw():
    window.clear()
    env_renderer.draw()

@window.event
def on_resize(width, height):
    pyglet.gl.glViewport(0, 0, width, height)
    if envi != None:
        set_projection(envi, width, height)
    else:
        pyglet.gl.glMatrixMode(pyglet.gl.GL_PROJECTION)
        pyglet.gl.glLoadIdentity()
        pyglet.gl.glOrtho(0, 10, 0, 10, -1, 1)
        pyglet.gl.glMatrixMode(pyglet.gl.GL_MODELVIEW)
    return True

@window.event
def on_mouse_scroll(x, y, scroll_x, scroll_y):
    (mx, my)= get_mouse_coord(x, y)
    pyglet.gl.glTranslatef(mx, my, 0)
    pyglet.gl.glScalef(1.05**scroll_y, 1.05**scroll_y, 1)
    pyglet.gl.glTranslatef(-mx, -my, 0)

@window.event
def on_mouse_drag(x, y, dx, dy, buttons, modifiers):
    if pyglet.window.mouse.RIGHT & buttons:
        mcoord1 = get_mouse_coord(x, y)
        mcoord2 = get_mouse_coord(x + dx, y+ dy)
        pyglet.gl.glTranslatef(mcoord2[0] - mcoord1[0], mcoord2[1] - mcoord1[1], 0)
        
@window.event
def on_mouse_release(x, y, button, modifiers):
    if pyglet.window.mouse.LEFT & button:
        mcoord2 = np.array(get_mouse_coord(x, y))# - envi.agent_size/2
        mcoord1 = np.min(envi.agent.exterior.coords, axis=0) 
        envi.move((mcoord2[0] - mcoord1[0], mcoord2[1] - mcoord1[1]))
        global env_renderer
        env_renderer = Renderer(environment=envi, render_call=Enviornment_draw)

# def on_key_press(symbol, modifiers):
#     global index, index_range, rrt_renderer
#     if symbol == key.RIGHT:
#         index = np.clip(index + 1, *index_range)
#         rrt_renderer = Renderer(rrt_data=rrt_data,
#                                 render_call=RRT_draw,
#                                 index = index)
#     if symbol == key.LEFT:
#         index = np.clip(index -1, *index_range)
#         rrt_renderer = Renderer(rrt_data=rrt_data,
#                                 render_call=RRT_draw,
#                                 index = index)
# 
# 
# 
# window.push_handlers(on_key_press)


if __name__ == '__main__':
    pyglet.gl.glEnable(pyglet.gl.GL_BLEND)
    pyglet.gl.glBlendFunc(pyglet.gl.GL_SRC_ALPHA, pyglet.gl.GL_ONE_MINUS_SRC_ALPHA)
    pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH )
    pyglet.gl.glEnable(pyglet.gl.GL_POLYGON_SMOOTH )
    pyglet.gl.glEnable(pyglet.gl.GL_POINT_SMOOTH )
    pyglet.gl.glClearColor(0, 0, 0, 1.0)
    pyglet.app.run()