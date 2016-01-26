# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <codecell>

# %matplotlib inline
import pygame
import numpy as np
import matplotlib.pyplot as plt

# <codecell>

class PointMass:
    def __init__(self, position0, p0, v0, a0, angle0, angle_v0, angle_a0):
        self.position0 = np.copy(position0)
        self.position = position0
        
        self.p0 = np.copy(p0)
        self.v0 = np.copy(v0)
        self.a0 = np.copy(a0)
        
        self.pos = position0
        self.speed = v0 # m/s
        self.acceleration = a0
        
        self.angle0 = np.copy(angle0) # orientation
        self.angle_v0 = np.copy(angle_v0)
        self.angle_a0 = np.copy(angle_a0)

        self.angle = angle0
        self.angle_speed = angle_v0 # m/s
        self.angle_acceleration = angle_a0
        
        self.friction = .02
        self.angle_friction = 0.01

        # circle properties
        self.size = 6
        self.color = 0, 0, 0
        
        # line properties
        self.line_length = 15
        
    def reset(self):
        """ Reset state to initial state """
        self.position = np.copy(self.position0)
        self.pos = self.p0
        self.speed = self.v0
        self.acceleration = self.a0
        self.angle = self.angle0
        self.angle_speed = self.angle_v0 # m/s
        self.angle_acceleration = self.angle_a0
        
    def motion(self, dt):
        """ Integrate motion equations """
        self.angle_speed = np.append(self.angle_speed, self.angle_speed[-1] * (1 - self.angle_friction) + self.angle_acceleration[-1] * dt)
        self.angle = np.append(self.angle, self.angle[-1] + self.angle_speed[-2] * dt)
        
        self.speed = np.append(self.speed, self.speed[-1] * (1 - self.friction) + self.acceleration[-1] * dt)
#         print self.speed
        
        self.position[0] = self.position[0] + self.speed[-1] * dt * np.cos(self.angle[-1])
        self.position[1] = self.position[1] - self.speed[-1] * dt * np.sin(self.angle[-1])
        
        self.pos = np.vstack((self.pos, self.position))                

    def display(self,screen):
        line_end_point_x = self.line_length * np.cos(self.angle[-1])
        line_end_point_y = self.line_length * np.sin(self.angle[-1])
        pygame.draw.line(screen, self.color, (int(self.position[0]), int(self.position[1])), (int(self.position[0]+line_end_point_x), int(self.position[1]-line_end_point_y)), 4)
        pygame.draw.circle(screen, self.color, (int(self.position[0]), int(self.position[1])), self.size)

# <codecell>

def boundaries(point_mass_object, width, height):
    """ check if point mass is outside the specified borders and reset() """
    if point_mass_object.position[0] < 0 or point_mass_object.position[0] > width or point_mass_object.position[1] < 0 or point_mass_object.position[1] > height:
        point_mass_object.reset()

# <headingcell level=1>

# Record Training Data

# <codecell>

from time import sleep

width = 100;
height = 100;
# screen = pygame.display.set_mode((width,height))
# bgcolor = 255, 255, 255

# running = 1

dt = 0.01 # [seconds]
T = .2 # duration of acceleration

ntrials = 10000
### Store data: theta_t, x_t, y_t, theta_t+1, x_t+1, y_t+1, a_t, w_t, T ###
data = np.zeros((ntrials,9))

for itrial in range(ntrials):
    t = 0. # current time
    
    ### Initialize point mass ###
    position0 = np.array([50., 50.])
    p0 = np.array([0.])
    v0 = np.array([0.])
    a0 = np.array([0.])
    angle0 = np.random.random((1,))*2*np.pi
    angle_v0 = np.array([0.])
    angle_a0 = np.array([0.])
    pm = PointMass(position0, p0, v0, a0, angle0, angle_v0, angle_a0)

    ### Run Simulation ###
    # while running:
    for _ in range(500):
#         event = pygame.event.poll()
#         if event.type == pygame.QUIT:
#             running = 0
#         screen.fill(bgcolor)
        
        if t == 0:
            pm.acceleration = np.random.random((1,)) * 500
            pm.angle_acceleration = (np.random.random((1,))-0.5) * 10 * np.pi
    #         print pm.angle_acceleration
        elif t < T:
            pm.acceleration = np.append(pm.acceleration, pm.acceleration[-1])
            pm.angle_acceleration = np.append(pm.angle_acceleration, pm.angle_acceleration[-1])
    #         print pm.angle_acceleration
        else:
            pm.acceleration = np.append(pm.acceleration, a0.reshape((1,)))
            pm.angle_acceleration = np.append(pm.angle_acceleration, angle_a0.reshape((1,)))
        pm.motion(dt)
        
    #     boundaries(pm, width, height)
#         pm.display(screen)
    #     target.display(screen)
#         pygame.display.flip()
        
        t = t + dt
#         sleep(0.01)
    
#     pygame.quit()
    ### angle_start, x_start, y_start, angle_end, x_end, y_end, acceleration, angle_acceleration, T
    data[itrial,:] = np.asarray((pm.angle[0], pm.pos[0][0], pm.pos[0][1], pm.angle[-1], pm.pos[-1][0], pm.pos[-1][1], pm.acceleration[0], pm.angle_acceleration[0], T))
    
### Save 'data' to file ###
data.dump('data')
print 'DONE!'

# <headingcell level=1>

# Train Model

# <headingcell level=3>

# Load and standardize recorded data

# <codecell>

import numpy as np
from sklearn import preprocessing

# Load dataset
dat = np.load('data')

# Find mean and std of data
data_mean = np.mean( dat , axis = 0 )
data_std = np.std( dat , axis = 0 )
# Standardize data
data = ( dat - data_mean ) / data_std

test_percent = 10 # 10 percent of the data should be used for the testset
n_test = data.shape[0]*test_percent/100

### angle_start, angle_end, x_start, x_end, y_start, y_end, acceleration, angle_acceleration, T
state_one_train = data[:-n_test,0:3]
state_two_train = data[:-n_test,3:6]
motor_commands_train = data[:-n_test,6:8]
state_one_test = data[-n_test:,0:3]
state_two_test = data[-n_test:,3:6]
motor_commands_test = data[-n_test:,6:8]

# Inverse Model Features
X_inverse_train = np.append(state_one_train,state_two_train,axis = 1)
Y_inverse_train = motor_commands_train
X_inverse_test = np.append(state_one_test,state_two_test,axis = 1)
Y_inverse_test = motor_commands_test

# Forward Model Features
X_forward_train = np.append(state_one_train,motor_commands_train,axis = 1)
Y_forward_train = state_two_train
X_forward_test = np.append(state_one_test,motor_commands_test,axis = 1)
Y_forward_test = state_two_test

# <headingcell level=3>

# Train MLP on recorded data

# <codecell>

from sknn.mlp import Regressor, Layer

# Do some MLP black magic
inverse = Regressor(layers=[Layer("Sigmoid", units=4), Layer("Linear",units=2)],learning_rate=0.02)
inverse.fit(X_inverse_train,Y_inverse_train)

# <codecell>

# Save model for later use
from sklearn.externals import joblib
joblib.dump(inverse, 'neural_network_inverse_model.pkl')

# <headingcell level=1>

# Test Model

# <headingcell level=3>

# Load and standardize recorded data

# <codecell>

import numpy as np
from sklearn import preprocessing

# Load dataset
dat = np.load('data')

# Find mean and std of data
data_mean = np.mean( dat , axis = 0 )
data_std = np.std( dat , axis = 0 )
data = ( dat - data_mean ) / data_std

test_percent = 10 # 10 percent of the data should be used for the testset
n_test = data.shape[0]*test_percent/100

### angle_start, x_start, y_start, angle_end, x_end, y_end, acceleration, angle_acceleration, T
state_one_train = data[:-n_test,0:3]
state_two_train = data[:-n_test,3:6]
motor_commands_train = data[:-n_test,6:8]
state_one_test = data[-n_test:,0:3]
state_two_test = data[-n_test:,3:6]
motor_commands_test = data[-n_test:,6:8]

# Inverse Model Features
X_inverse_train = np.append(state_one_train,state_two_train,axis = 1)
Y_inverse_train = motor_commands_train
X_inverse_test = np.append(state_one_test,state_two_test,axis = 1)
Y_inverse_test = motor_commands_test

# Forward Model Features
X_forward_train = np.append(state_one_train,motor_commands_train,axis = 1)
Y_forward_train = state_two_train
X_forward_test = np.append(state_one_test,motor_commands_test,axis = 1)
Y_forward_test = state_two_test

# <headingcell level=3>

# Load Inverse Model

# <codecell>

from sklearn.externals import joblib
nn = joblib.load('neural_network_inverse_model.pkl')

# print np.mean((nn.predict(X_inverse_test)-Y_inverse_test)**2)

# <headingcell level=3>

# Simulate point mass using inverse model

# <codecell>

from time import sleep

width = 100;
height = 100;
screen = pygame.display.set_mode((width,height))
bgcolor = 255, 255, 255

running = 1

dt = 0.01 # [seconds]
T = .2 # duration of acceleration

# Initialize Target Point Mass ###
random_index = np.random.randint(data.shape[0]-n_test,data.shape[0])
t_position0 = dat[random_index,[4, 5]]
t_p0 = np.array([0.])
t_v0 = np.array([0.])
t_a0 = np.array([0.])
t_angle0 = np.array([dat[random_index,3]])
t_angle_v0 = np.array([0.])
t_angle_a0 = np.array([0.])
target = PointMass(t_position0, t_p0, t_v0, t_a0, t_angle0, t_angle_v0, t_angle_a0)
target.color = 255, 0, 0

ntrials = 1

for itrial in range(ntrials):
    t = 0. # current time
    
    ### Initialize Point Mass ###
    position0 = np.array([50., 50.])
    p0 = np.array([0.])
    v0 = np.array([0.])
    a0 = np.array([0.])
    angle0 = np.array([dat[random_index,0]])
    angle_v0 = np.array([0.])
    angle_a0 = np.array([0.])
    pm = PointMass(position0, p0, v0, a0, angle0, angle_v0, angle_a0)

    ### Run Simulation ###
    # while running:
    for _ in range(300):
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = 0 
        screen.fill(bgcolor)
        
        if t == 0:
### angle_start, x_start, y_start, angle_end, x_end, y_end, acceleration, angle_acceleration, T
            feature_pre = np.asarray((pm.angle[0], pm.pos[0], pm.pos[1], target.angle[0], target.pos[0], target.pos[1]))
            # standardize feature
            feature = (feature_pre - data_mean[0:6]) / data_std[0:6]
            # predict motor_commands
            motor_commands = nn.predict(feature.reshape([1,6]))
            # Go back to original scale of motor_commands
            pm.acceleration = np.array([motor_commands[0][0]]) * data_std[6] + data_mean[6]
            pm.angle_acceleration = np.array([motor_commands[0][1]]) * data_std[7] + data_mean[7]
        elif t < T:
            pm.acceleration = np.append(pm.acceleration, pm.acceleration[-1])
            pm.angle_acceleration = np.append(pm.angle_acceleration, pm.angle_acceleration[-1])
    #         print pm.angle_acceleration
        else:
            pm.acceleration = np.append(pm.acceleration, a0.reshape((1,)))
            pm.angle_acceleration = np.append(pm.angle_acceleration, angle_a0.reshape((1,)))
        pm.motion(dt)
        
    #     boundaries(pm, width, height)
        pm.display(screen)
        target.display(screen)
        pygame.display.flip()
        
        t = t + dt
        sleep(0.01)
    
    pygame.quit()

