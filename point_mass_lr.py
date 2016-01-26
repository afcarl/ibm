# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# dominik koller, lab rotation

# <codecell>

# %matplotlib inline
import pygame
import numpy as np
import matplotlib.pyplot as plt
from point_mass_PM import PointMass
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

ntrials = 1000 # 00
### Store data: theta_t, x_t, y_t, theta_t+1, x_t+1, y_t+1, a_t, w_t, T ###
data = np.zeros((ntrials,9))

for itrial in range(ntrials):
    print("numtrial %d" % itrial)
    t = 0. # current time
    
    ### Initialize point mass ###
    position0 = np.random.uniform(40, 60, (2,)) # np.array([50., 50.])
    p0 = np.array([0.])
    v0 = np.array([0.])
    a0 = np.array([0.])
    angle0 = np.random.random((1,))*2*np.pi # np.zeros((1,)) #
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
            pm.angle_acceleration = (np.random.random((1,))-0.5) * 10 * np.pi # np.zeros((1,)) # 
    #         print pm.angle_acceleration
        elif t < T:
            # probly better to alloc zeros right at the start
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
    # print pm.angle.shape, pm.pos.shape
    # plt.subplot(411)
    # plt.plot(pm.angle)
    # plt.subplot(412)
    # plt.plot(pm.pos)
    # plt.subplot(413)
    # plt.plot(pm.acceleration)
    # plt.subplot(414)
    # plt.plot(pm.angle_acceleration)
    # plt.show()
    data[itrial,:] = np.asarray((pm.angle[0], pm.pos[0][0], pm.pos[0][1], pm.angle[-1], pm.pos[-1][0], pm.pos[-1][1], pm.acceleration[0], pm.angle_acceleration[0], T))
    
### Save 'data' to file ###
data.dump('data')
print 'DONE! generating data'


