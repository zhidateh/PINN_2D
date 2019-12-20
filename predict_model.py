
import utils as utils
import os
import dnn as dnn
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import time
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('plot', help='Plot or not(key in 1 to plot)')
args = parser.parse_args()

np.random.seed(1234)
tf.set_random_seed(1234)

setting_name    = "DNN00061"
case            = 0
node_filter     = 1
batch_size      = 500
num_iter        = 5001
num_epochs      = 10
learning_rate   = 1e-3
layers = [10,10,10]


model_path = os.getcwd() + '/2d_inviscid_model/%s/'%setting_name
test_path = os.getcwd() + '/test/'
train_path = os.getcwd() + '/train/'

##-----------------------------------Loading----------------------------------------------------
        
#data is matrix of m x n, where m is number of nodes, n is number of datasets
P_back,x,y,P,rho,u,v,T,Et,E = utils.loadData(train_path,node_filter)
P_back_test,x_test,y_test,P_test,rho_test,u_test,v_test,T_test,Et_test,E_test = utils.loadData(test_path,node_filter)
case = P_back_test.flatten()[0]
case_number = P_back_test.shape[0]
node_number = P_back_test.shape[1]
#normalize data with stagnation condition------------------------------------------------
cLength = 1
cVelocity = np.amax(np.sqrt(np.array(u)**2 + np.array(v)**2))  

#normalizer  
x_norm      = cLength
y_norm      = cLength
P_norm      = 1.225*cVelocity**2
rho_norm    = 1.225
u_norm      = cVelocity
v_norm      = cVelocity
T_norm      = 1
Et_norm     = 1.225*cVelocity**2
E_norm      = 1.225*cVelocity**2

#-------------------------Training Data------------------------------------------------
x       /= x_norm
y       /= y_norm
P_back  /= P_norm
P       /= P_norm#(P/P)*101325/P_norm
rho     /= rho_norm#(rho/rho)*1.225/rho_norm
u       /= u_norm#(u/u)*0*u_norm 
v       /= v_norm#(v/v)*0*v_norm
T       /= T_norm#(T/T)*300/T_norm
Et      /= Et_norm#(E/E)*(300*287/0.4)/Et_norm
E       /= E_norm#(E/E)*0*E_norm

P_back = P_back.flatten()[:,None]
x = x.flatten()[:,None]
y = y.flatten()[:,None]
P = P.flatten()[:,None]
rho = rho.flatten()[:,None]
u = u.flatten()[:,None]
v = v.flatten()[:,None]
T = T.flatten()[:,None]
Et = Et.flatten()[:,None]
E = E.flatten()[:,None]

#-------------------------Wall Data------------------------------------------------

wall_x, wall_y =  utils.getWallIndex(x_test,y_test, 0.0,0.14,281,"horizontal")
outlet_x, outlet_y =  utils.getWallIndex(x_test,y_test, 0.84,0.0022,51,"vertical")

wall_x = wall_x.flatten()[:,None]
wall_y = wall_y.flatten()[:,None]
outlet_x = outlet_x.flatten()[:,None]
outlet_y = outlet_y.flatten()[:,None]

#-------------------------Testing Data------------------------------------------------
x_test       /= x_norm
y_test       /= y_norm
P_back_test  /= P_norm
P_test       /= P_norm
rho_test     /= rho_norm
u_test       /= u_norm
v_test       /= v_norm
T_test       /= T_norm
Et_test      /= Et_norm
E_test       /= E_norm


P_back_test = P_back_test.flatten()[:,None]
x_test = x_test.flatten()[:,None]
y_test = y_test.flatten()[:,None]
P_test = P_test.flatten()[:,None] 
rho_test = rho_test.flatten()[:,None]
u_test = u_test.flatten()[:,None]
v_test = v_test.flatten()[:,None]
T_test = T_test.flatten()[:,None]
Et_test = Et_test.flatten()[:,None]
E_test = E_test.flatten()[:,None]


# plt.plot(wall_x, wall_y + 0.0025, 'xr')
# plt.plot(x_test,y_test,'xb')
# plt.show()

#prepare wall data -----------------------------------------------
x_w = []
y_w = []
P_back_w    = []
P_w         = []
rho_w       = []
u_w         = []
v_w         = []
Et_w        = []

z = np.zeros(x.shape,dtype=bool)
for ix,iy in zip(wall_x,wall_y):
    idx =  np.logical_and( x == ix, y == iy).flatten()

    x_w         += x[idx].flatten().tolist()
    y_w         += (y[idx]+0.00269).flatten().tolist()
    P_back_w    += P_back[idx].flatten().tolist()
    P_w         += P[idx].flatten().tolist()
    rho_w       += rho[idx].flatten().tolist()
    u_w         += (u[idx]*0.0).flatten().tolist()
    v_w         += (v[idx]*0.0).flatten().tolist()
    Et_w        += Et[idx].flatten().tolist()

x_w         =np.array(x_w      ).flatten()[:,None]
y_w         =np.array(y_w      ).flatten()[:,None]
P_back_w    =np.array(P_back_w ).flatten()[:,None]
P_w         =np.array(P_w      ).flatten()[:,None]
rho_w       =np.array(rho_w    ).flatten()[:,None]
u_w         =np.array(u_w      ).flatten()[:,None]
v_w         =np.array(v_w      ).flatten()[:,None]
Et_w        =np.array(Et_w     ).flatten()[:,None]

#prepare outlet data -----------------------------------------------
x_outlet = []
y_outlet = []

z = np.zeros(x.shape,dtype=bool)
for ix,iy in zip(outlet_x,outlet_y):
    idx =  np.logical_and( x == ix, y == iy).flatten()

    x_outlet         += (x[idx]+0.002968).flatten().tolist()
    y_outlet         += y[idx].flatten().tolist()

x_outlet         =np.array(x_outlet).flatten()[:,None]
y_outlet         =np.array(y_outlet).flatten()[:,None]



#only use data near inlet or outlet-----------------------------------------
# filter_idx =np.logical_or(x<0.2, x>0.7)
# #filter_idx = [x>0.5]
# P_back  = P_back[filter_idx].flatten()[:,None]
# x       = x[filter_idx].flatten()[:,None]
# y       = y[filter_idx].flatten()[:,None]
# P       = P[filter_idx].flatten()[:,None]
# rho     = rho[filter_idx].flatten()[:,None]
# u       = u[filter_idx].flatten()[:,None]
# v       = v[filter_idx].flatten()[:,None]
# T       = T[filter_idx].flatten()[:,None]
# Et      = Et[filter_idx].flatten()[:,None]
# E       = E[filter_idx].flatten()[:,None]

# plt.scatter(x,y, c=u)
# plt.plot(x_w,y_w)
# plt.show()



#--------------------------------------------------------------------------------------
#check if batch size > max of nodes in each category
# if batch_size < min(x.shape[0],outlet_x.shape[0]):
#     batch_size = min(x.shape[0],outlet_x.shape[0])

#initiaise PINN class
model = dnn.DenseNN_2D(P_back,\
                        x, \
                        y, \
                        P, \
                        rho, \
                        u, \
                        v, \
                        Et, \
                        P_back_w, \
                        x_w, \
                        y_w, \
                        P_w, \
                        rho_w, \
                        u_w, \
                        v_w, \
                        Et_w, \
                        x_outlet, \
                        y_outlet, \
                        layers, \
                        learning_rate)


model.ckpt_name = 'tmp/' + setting_name 

#Load trained model-------------------------------------------------------------------------------------------------
saver = tf.compat.v1.train.import_meta_graph('/home/zhida/Documents/PINN_CFD_2D/2d_inviscid_model/DNN00061/DNN00061-9.meta')
saver.restore(model.sess,("/home/zhida/Documents/PINN_CFD_2D/2d_inviscid_model/DNN00061/DNN00061-9"))
#--------------------------------------------------------------------------------------------------------------------

#Training-----------------------------------------------------------------------------------------------------------
#model.train(num_epochs, num_iter, batch_size)
#model.save(os.getcwd() + '/save/')
#-------------------------------------------------------------------------------------------------------------------

##Testing---------------------------------------------------------------------------------------------------------------

#Prediction
P_pred, rho_pred, u_pred, v_pred, Et_pred = model.predict(P_back_test, x_test,y_test)

#Error
error_P = np.linalg.norm(P_test-P_pred,2)/np.linalg.norm(P_test,2)
print("Test Error in P: "+str(error_P))
error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
print("Test Error in rho: "+str(error_rho))
error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
print("Test Error in u: "+str(error_u))
error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
print("Test Error in v: "+str(error_v))
error_Et = np.linalg.norm(Et_test-Et_pred,2)/np.linalg.norm(Et_test,2)
print("Test Error in E: "+str(error_Et))


print(case_number)
print(node_number)


# P_pred       *= P_norm
# rho_pred     *= rho_norm
# u_pred       *= u_norm
# v_pred       *= v_norm
# Et_pred       *= Et_norm


if(args.plot == "1"):
    plt.figure(1)
    plt.title('Pressure')
    for i in range(case_number):
        ##get centreline data
        y_ = y_test.flatten()[i:i+node_number]
        filter_idx = y_ < 0.0035
        plot_x = x_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_P_true = P_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_P_pred = P_pred.flatten()[i*node_number:(i+1)*node_number][filter_idx]

        sort_ = np.argsort(plot_x)

        plt.plot(plot_x[sort_],plot_P_true[sort_],'-r')
        plt.plot(plot_x[sort_],plot_P_pred[sort_],'-b')
        plt.legend(['True value','Predicted value'],loc="best")


    plt.figure(2)
    plt.title('Density')
    for i in range(case_number):
        ##get centreline data
        y_ = y_test.flatten()[i:i+node_number]
        filter_idx = y_ < 0.0035
        plot_x = x_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_rho_true = rho_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_rho_pred = rho_pred.flatten()[i*node_number:(i+1)*node_number][filter_idx]

        sort_ = np.argsort(plot_x)

        plt.plot(plot_x[sort_],plot_rho_true[sort_],'-r')
        plt.plot(plot_x[sort_],plot_rho_pred[sort_],'-b')
        plt.legend(['True value','Predicted value'],loc="best")
    
    plt.figure(3)
    plt.title('Velocity u')
    for i in range(case_number):
        ##get centreline data
        y_ = y_test.flatten()[i:i+node_number]
        filter_idx = y_ < 0.0035
        plot_x = x_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_u_true = u_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_u_pred = u_pred.flatten()[i*node_number:(i+1)*node_number][filter_idx]

        sort_ = np.argsort(plot_x)

        plt.plot(plot_x[sort_],plot_u_true[sort_],'-r')
        plt.plot(plot_x[sort_],plot_u_pred[sort_],'-b')
        plt.legend(['True value','Predicted value'],loc="best")

    
    plt.figure(4)
    plt.title('Velocity v')
    for i in range(case_number):
        ##get centreline data
        y_ = y_test.flatten()[i:i+node_number]
        filter_idx = y_ < 0.0035
        plot_x = x_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_v_true = v_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_v_pred = v_pred.flatten()[i*node_number:(i+1)*node_number][filter_idx]

        sort_ = np.argsort(plot_x)

        plt.plot(plot_x[sort_],plot_v_true[sort_],'-r')
        plt.plot(plot_x[sort_],plot_v_pred[sort_],'-b')
        plt.legend(['True value','Predicted value'],loc="best")

    
    plt.figure(5)
    plt.title('Energy')
    for i in range(case_number):
        ##get centreline data
        y_ = y_test.flatten()[i:i+node_number]
        filter_idx = y_ < 0.0035
        plot_x = x_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_Et_true = Et_test.flatten()[i*node_number:(i+1)*node_number][filter_idx]
        plot_Et_pred = Et_pred.flatten()[i*node_number:(i+1)*node_number][filter_idx]

        sort_ = np.argsort(plot_x)

        plt.plot(plot_x[sort_],plot_Et_true[sort_],'-r')
        plt.plot(plot_x[sort_],plot_Et_pred[sort_],'-b')
        plt.legend(['True value','Predicted value'],loc="best")
    
    
    
    plt.show()
 



    