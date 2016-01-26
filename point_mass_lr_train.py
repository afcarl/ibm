# <headingcell level=1>

# Train Model

# <headingcell level=3>

# Load and standardize recorded data

# <codecell>

import numpy as np
import pylab as plt
from sklearn import preprocessing
import pygame
from point_mass_PM import PointMass

# Load dataset
dat = np.load('data')


# Find mean and std of data
data_mean = np.mean( dat , axis = 0 )
# data_mean = np.zeros_like(data_mean)
data_std = np.std( dat , axis = 0 ) + 1e-8
# data_std = np.zeros_like(data_std)
print data_mean, data_std
# Standardize data
data = ( dat - data_mean ) / data_std

# plt.plot(dat[:,6])
# plt.show()

    
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

print state_one_train.shape
print state_two_train.shape
print motor_commands_train.shape


X_forward_train = np.append(state_one_train,motor_commands_train,axis = 1)
Y_forward_train = state_two_train

print X_forward_train.shape
print Y_forward_train.shape

# plt.subplot(211)
# plt.plot(X_forward_train)
# plt.subplot(212)
# plt.plot(Y_forward_train)
# plt.show()

X_forward_test = np.append(state_one_test,motor_commands_test,axis = 1)
Y_forward_test = state_two_test

# <headingcell level=3>

# Train MLP on recorded data

# <codecell>

import os

from sknn.mlp import Regressor, Layer
from sklearn.externals import joblib

N_ITER = 1000 # None
N_HIDDEN = 100
# HIDDEN_LYR_ACT = "Sigmoid"
HIDDEN_LYR_ACT = "Tanh"
DROPOUT = 0.01
LR = 0.02

################################################################################
# forward model

forward_df = "point_mass_lr_forward_model"

if os.path.exists(forward_df):
    print "loading %s" % forward_df
    forward = joblib.load(forward_df)
else:
    print "training %s" % forward_df
    forward = Regressor(layers=[
        Layer(HIDDEN_LYR_ACT, units=N_HIDDEN),
        Layer("Linear", units=Y_forward_train.shape[1],
              dropout=DROPOUT)
        ],
        #learning_rate=0.02, verbose=True)
    learning_rate=LR, n_iter=N_ITER, verbose=True)
    forward.fit(X_forward_train, Y_forward_train)
    joblib.dump(forward, forward_df)

Y_forward_pred = forward.predict(X_forward_test)

numcomps = Y_forward_pred.shape[1]
for i in range(Y_forward_pred.shape[1]):
    plt.subplot(numcomps, 1, i+1)
    plt.plot(Y_forward_pred[:,i], "k-", label="prediction")
    plt.plot(Y_forward_test[:,i], "k--", label="target")
    plt.legend()
plt.show()

################################################################################
# inverse
inverse_df = "point_mass_lr_inverse_model"

if os.path.exists(inverse_df):
    print "loading %s" % inverse_df
    inverse = joblib.load(inverse_df)
else:
    print "training %s" % inverse_df
    inverse = Regressor(layers=[
        Layer(HIDDEN_LYR_ACT, units=N_HIDDEN),
        Layer("Linear",units=Y_inverse_train.shape[1],
              dropout=DROPOUT)
        ],
        # learning_rate=0.02, verbose=True)
    learning_rate=LR, n_iter=N_ITER, verbose=True)
    inverse.fit(X_inverse_train, Y_inverse_train)
    joblib.dump(inverse, inverse_df)

Y_inverse_pred = inverse.predict(X_inverse_test)


numcomps = Y_inverse_pred.shape[1]
for i in range(Y_inverse_pred.shape[1]):
    plt.subplot(numcomps+1, 1, i+1)
    plt.plot(Y_inverse_pred[:,i], "k-", label="prediction")
    plt.plot(Y_inverse_test[:,i], "k--", label="target")
    plt.legend()
plt.subplot(numcomps+1, 1, i+2)
plt.plot(X_inverse_test, label="blub")
plt.legend()
plt.show()


# # Do some MLP black magic

# # <codecell>

# # Save model for later use
# joblib.dump(inverse, 'neural_network_inverse_model.pkl')

# # <headingcell level=1>

# # Test Model

# # <headingcell level=3>

# # Load and standardize recorded data

# # <codecell>

# import numpy as np
# from sklearn import preprocessing

# # Load dataset
# dat = np.load('data')

# # Find mean and std of data
# data_mean = np.mean( dat , axis = 0 )
# data_std = np.std( dat , axis = 0 )
# data = ( dat - data_mean ) / data_std

# test_percent = 10 # 10 percent of the data should be used for the testset
# n_test = data.shape[0]*test_percent/100

# ### angle_start, x_start, y_start, angle_end, x_end, y_end, acceleration, angle_acceleration, T
# state_one_train = data[:-n_test,0:3]
# state_two_train = data[:-n_test,3:6]
# motor_commands_train = data[:-n_test,6:8]
# state_one_test = data[-n_test:,0:3]
# state_two_test = data[-n_test:,3:6]
# motor_commands_test = data[-n_test:,6:8]

# # Inverse Model Features
# X_inverse_train = np.append(state_one_train,state_two_train,axis = 1)
# Y_inverse_train = motor_commands_train
# X_inverse_test = np.append(state_one_test,state_two_test,axis = 1)
# Y_inverse_test = motor_commands_test

# # Forward Model Features
# X_forward_train = np.append(state_one_train,motor_commands_train,axis = 1)
# Y_forward_train = state_two_train
# X_forward_test = np.append(state_one_test,motor_commands_test,axis = 1)
# Y_forward_test = state_two_test

# # <headingcell level=3>

# # Load Inverse Model

# # <codecell>

# from sklearn.externals import joblib
# nn = joblib.load('neural_network_inverse_model.pkl')

# # print np.mean((nn.predict(X_inverse_test)-Y_inverse_test)**2)

# # <headingcell level=3>

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
    for _ in range(500):
        event = pygame.event.poll()
        if event.type == pygame.QUIT:
            running = 0 
        screen.fill(bgcolor)
        
        if t == 0:
### angle_start, x_start, y_start, angle_end, x_end, y_end, acceleration, angle_acceleration, T
            feature_pre = np.asarray((pm.angle[0], pm.pos[0], pm.pos[1], target.angle[0], target.pos[0], target.pos[1]))
            # standardize feature
            feature = (feature_pre - data_mean[0:6]) / data_std[0:6]
            print feature_pre, feature
            # predict motor_commands
            motor_commands = inverse.predict(feature.reshape([1,6]))
            print "motor_commands", motor_commands, data_std[6], data_mean[6]
            # Go back to original scale of motor_commands
            pm.acceleration = (np.array([motor_commands[0][0]]) * data_std[6]) + data_mean[6]
            print "pm.acceleration", pm.acceleration
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

    # plt.clf()
    # plt.plot(pm.acceleration)
    # plt.show()
        
