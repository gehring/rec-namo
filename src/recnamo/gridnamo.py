import numpy as np

class gridnamo(object):
    
    actions = [np.array((1, 0), dtype = 'int'),
               np.array((-1, 0), dtype = 'int'),
               np.array((0, 1), dtype = 'int'),
               np.array((0, -1), dtype = 'int')]
    
    def __init__(self, size, objects, limits, walls = None):
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
        self.limits = limits
        self.world = world
        self.obj_pos = object_pos
    
    def step(self, a):
        if a<4:
            collision = False
            next_a = self.agent + self.actions[a]
            
            collision |= not(np.all(self.limits[0] <= next_a) 
                                 and np.all(self.limits[1] > next_a))
            
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
                    
        
        pass
    
    def move_agent(self, next_a):
        del self.world[tuple(self.agent)]
        self.agent = next_a
        self.world[tuple(next_a)] = self.agent_id
        
    def move_obj(self, next_o, obj):
        del self.world[tuple(self.obj_pos[obj])]
        self.obj_pos[obj] = next_o
        self.world[tuple(self.obj_pos[obj])] = obj
        
        
    def is_state_valid(self, state):
        pass