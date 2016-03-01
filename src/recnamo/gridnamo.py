import numpy as np
from scipy import sparse as sp


class gridnamo(object):
    
    actions = [np.array((1, 0), dtype = 'int'),
               np.array((-1, 0), dtype = 'int'),
               np.array((0, 1), dtype = 'int'),
               np.array((0, -1), dtype = 'int')]
    
    def __init__(self, objects, limits, walls = None):
        i = 0
        world = {}
        object_pos = []
        for o in objects:
            world[o] = i
            object_pos.append(np.array(o), dtype = 'int')
            i += 1
            
        world[(0,0)] = i
        self.agent_id = i
        self.agent = np.array((0,0), dtype = 'int')
        
        self.attached = None
        if walls is not None:
            for w in walls:
                world[tuple(w)] = -1
        self.walls = walls
        self.limits = np.tile(limits, reps=len(objects) + 1)
        self.world = world
        self.obj_pos = object_pos
    
    def step(self, a):
        if a<4:
            collision = False
            next_a = self.agent + self.actions[a]
            
            collision |= not(np.all(0 <= next_a) 
                                 and np.all(self.limits > next_a))
            
            if self.attached is None:
                
                collision |= tuple(next_a) in self.world
                if not collision:
                    self.move_agent(next_a)
                    
            else:
                next_o = self.obj_pos[self.attached] + self.actions[a]
                collision |= not(np.all(self.limits[0] <= next_o) 
                                 and np.all(self.limits[1] > next_o))
                collision |= (tuple(next_a) in self.world 
                              and self.world[tuple(next_a)] != self.attached)
                collision |= (tuple(next_o) in self.world 
                              and self.world[tuple(next_o)] != self.agent_id)
                if not collision:
                    self.move_agent(next_a)
                    self.move_obj(next_o, self.attached)
                    
        else:
            if self.attached is not None:
                self.attached = None
            else:
                # check surrounding squares, attach to the first occupied one
                # If all are empty, don't attach
                for a in self.actions:
                    pos = tuple(self.agent + a)
                    if pos in self.world:
                        self.attached = self.world[pos]
                        break
                    
    
    def move_agent(self, next_a):
        del self.world[tuple(self.agent)]
        self.agent = next_a
        self.world[tuple(next_a)] = self.agent_id
        
    def move_obj(self, next_o, obj):
        del self.world[tuple(self.obj_pos[obj])]
        self.obj_pos[obj] = next_o
        self.world[tuple(self.obj_pos[obj])] = obj
        
        
    def get_binary_vector(self, states):
        if states.ndim == 1:
            states = states.reshape((1,-1))
        limits = self.limits
        index = np.ravel_multi_index(states.T, limits)
        s = sp.coo_matrix((np.ones(index.shape[0]), (np.arange(index.shape[0]), index)), shape =(index.shape[0], np.prod(limits) ))
        return s
    
    def get_state_vector(self, binary_states):
        limits = self.limits
        binary_states = binary_states.tocoo()
        states = np.unravel_index(binary_states.col, limits)
        return np.array(states, dtype='int').T
        
    def get_state(self):
        state = np.hstack((self.obj_pos + [self.agent])).astype('int')
        return state
    
    def set_state_from_binary(self, binary):
        state = self.get_state_vector(binary).squeeze()
        self.set_state(state)
        
    def set_state(self, state):
        world = {}
        object_pos = []
        for i in xrange(0, state.shape[0]-2, 2):
            o = tuple(state[i:i+1])
            world[o] = int(i/2)
            object_pos.append(np.array(o), dtype = 'int')
            
        agent_pos = state[-2:]
        world[tuple(agent_pos)] = self.agent_id
        self.agent = np.array(agent_pos, dtype = 'int')
            
            
        if self.walls is not None:
            for w in self.walls:
                world[tuple(w)] = -1
                
        self.world = world
        self.obj_pos = object_pos
        
    def is_state_valid(self, state):
        pass
    
    def get_state_img(self, state):
        img = np.zeros(self.limits[:2])
        img[self.agent[0], self.agent[1]] = 1
        for o in self.obj_pos:
            img[o[0], o[1]] = 2
        if self.walls is not None:
            for w in self.walls:
                img[w[0], w[1]] = 3
        return img.T
    
        
        