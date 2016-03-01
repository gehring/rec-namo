import numpy as np
import copy

"""
authors: Clement Gehring
contact: gehring@csail.mit.edu
date: May 2015
"""

################## VARIOUS INTERFACES ########################################
class Environment(object):
    def __init__(self):
        pass
    
    """ Method to reinitialize the domain. This should always be called before
        a new episode is started.
    """
    def reset(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    """ Step the RL environment forward. This should step the internal step
        forward after applying the given action. It returns the reward received
        and the next state (which will be None if the episode terminates).
        
        action: action to be applied during this time step.
    """
    def step(self, action):
        s_t = self.state
        
        s_tp1 = self.sample_next_state(s_t, action)
        r_t = self.sample_reward(s_t, action, s_tp1)
        
        self.state = s_tp1
        
        return r_t, s_tp1
    
    """ Sample a next state given an action sampled at a given state.
    """
    def sample_next_state(self, state, action):
        raise NotImplementedError("Subclasses should implement this!")
    
    """ Sample a reward for observing a given transition.
    """
    def sample_reward(self, s_t, a_t, s_tp1):
        raise NotImplementedError("Subclasses should implement this!")
    
    """ Generate a deep copy of the domain. Useful to evaluate an agent
        on a domain without side effects.
    """
    def copy(self):
        new_domain = copy.deepcopy(self)
        return new_domain
    
    """ Range of the state space.
    """
    @property
    def state_range(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    """ Range of the action space.
    """
    @property
    def action_range(self):
        raise NotImplementedError("Subclasses should implement this!")
    
    
class DiscreteActionEnvironment(Environment):
    
    """ Set of discrete actions that can be executed.
    """
    @property
    def discrete_actions(self):
        raise NotImplementedError("Subclasses should implement this!")
    
################## MOUTAIN CAR IMPLEMENTATION #################################
class MountainCar(DiscreteActionEnvironment):
    min_pos = -1.2
    max_pos = 0.6

    max_speed = 0.07

    goal_pos = 0.5

    pos_start = -0.5
    vel_start = 0.0

    s_range =[ np.array([min_pos, -max_speed]),
                   np.array([max_pos, max_speed])]
    a_range = [np.array([-0.001]), 
                    np.array([0.001])]

    __discrete_actions = [ np.array([-0.001]),
                          np.array([0]),
                          np.array([0.001])]

    def __init__(self, random_start = False, **argk):
        self.state = np.zeros(2)
        self.random_start = random_start
        
        # prevent the ranges from being accidentally changed
        for i in range(2):
            self.s_range[i].flags.writeable = False
            self.a_range[i].flags.writeable = False
        
        self.reset()
        
    def reset(self):
        if self.random_start:
            self.state = np.array([np.random.uniform(self.state_range[0][0], 
                                                     self.state_range[1][0]),
                                   np.random.uniform(self.state_range[0][1], 
                                                     self.state_range[1][1])])
        else:
            self.state = np.array([self.pos_start, self.vel_start])

        return self.state.copy()
    
    def inGoal(self, state):
        return state[0] >= self.goal_pos
    
    def sample_next_state(self, state, action):
        state = state.copy()
        
        # update rules according to the RL book (Sutton and Barto, 98)
        state[1] += (np.clip(action[0], *self.action_range)
                            + np.cos(3*state[0])*-0.0025)

        state[:] = np.clip(state, *self.state_range)
        state[0] += state[1]

        state[:] = np.clip(state, *self.state_range)
        if state[0] <= self.min_pos and state[1] < 0:
            state[1] = (np.random.rand(1)-0.5)*0.01
            
        # check if next state is in goal, if it is, make it terminal
        if self.inGoal(state):
            state = None
            
        return state
    
    def sample_reward(self, s_t, a_t, s_tp1):
        # constant cost until termination
        return -1
    
    def get_pumping_policy(self):
        return lambda state: 2 if state[1] >= 0 else 0
    
    @property
    def state_range(self):
        return self.s_range
    
    @property
    def action_range(self):
        return self.a_range
    
    @property
    def discrete_actions(self):
        return self.__discrete_actions
 
################## SWING UP IMPLEMENTATION ####################################   
class SwingUpPendulum(DiscreteActionEnvironment):
    
    min_pos = -np.pi
    max_pos = np.pi

    umax = 2.0
    mass = 1.0
    length = 1.0
    G = 9.8
    integ_rate = 0.05
    
    required_up_time = 1.0
    up_range = np.pi/8.0
    max_speed = np.pi*3

    pos_start = np.pi/2.0
    vel_start = 0.0

    damping = 0.2
    
    s_range =np.array([ np.array([min_pos, -max_speed]),
                   np.array([max_pos, max_speed])])


    a_range = np.array([[-umax], [umax]])

    __discrete_actions = [np.array([-umax]),
                          np.array([0]),
                          np.array([umax])]
    
    
    def __init__(self, random_start = False, **argk):
        self.state = np.zeros(2)
        self.random_start = random_start
        
        # prevent the ranges from being accidentally changed
        for i in range(2):
            self.s_range[i].flags.writeable = False
            self.a_range[i].flags.writeable = False
        
        self.reset()
    
    
    def reset(self):
        if self.random_start:
            self.state = np.array([np.random.uniform(self.state_range[0][0], 
                                                     self.state_range[1][0]),
                                   np.random.uniform(self.state_range[0][1], 
                                                     self.state_range[1][1])])
        else:
            self.state = np.array([self.pos_start, self.vel_start])

        return self.state.copy()
    
    def update(self, action, policy=None):
        torque = np.clip(action, *self.action_range)
        moment = self.mass*self.length**2
                
        theta_acc = (torque - self.damping * self.state[1] \
                    - self.mass*self.G *self.length*np.sin(self.state[0]))/moment
        
        theta_delta_acc = self.integ_rate * theta_acc
        self.state[1] = np.clip(self.state[1] + theta_delta_acc, self.state_range[0][1], self.state_range[1][1])
        self.state[0] += self.state[1] * self.integ_rate
        self.adjustTheta()
        
    def adjustTheta(self):
        if self.state[0] >= np.pi:
            self.state[0] -= 2*np.pi
        if self.state[0] < -np.pi:
            self.state[0] += 2*np.pi
            
    def sample_next_state(self, state, action):
        state = state.copy()
        
        torque = np.clip(action, *self.action_range)
        moment = self.mass*self.length**2
                
        theta_acc = (torque - self.damping * state[1] \
                    - self.mass*self.G *self.length*np.sin(state[0]))/moment
        
        theta_delta_acc = self.integ_rate * theta_acc
        state[1] = np.clip(state[1] + theta_delta_acc, self.state_range[0][1], self.state_range[1][1])
        state[0] += state[1] * self.integ_rate
        self.adjustTheta()
            
        # check if next state is in goal, if it is, make it terminal
        if self.inGoal(state):
            state = None
            
        return state
    
    def sample_reward(self, s_t, a_t, s_tp1):
        # constant cost until termination
        return -1
            
    def inGoal(self, state):
        return angle_range_check(np.pi -self.up_range, np.pi + self.up_range, state[0])
        
    def get_pumping_policy(self):
        return lambda state: np.array([self.umax]) if state[1] >= 0 else np.array([-self.umax])
        
    @property
    def state_range(self):
        return self.s_range
    
    @property
    def action_range(self):
        return self.a_range
    
    @property
    def discrete_actions(self):
        return self.__discrete_actions
    
    
        
############### HELPER FUNCTIONS ######################################
""" check if every angle in vector x is between the angle limits a and b, which
    are also vectors
"""
def angle_range_check( a, b, x):
    a = np.mod(a, 2*np.pi)
    b = np.mod(b, 2*np.pi)
    theta_bar = np.mod(b-a, 2*np.pi)
    return np.mod(x-a, 2*np.pi)<=theta_bar