import os
import sys
import csv
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate
import matplotlib.cm as cm
from matplotlib.colors import LogNorm

name = "DNN00059"
save = False

#Load all data from a directory
#Specify the path of the directory
def loadData(path,filter):
    
    #path = "%s/%s/"%(os.getcwd(),purpose)
    #path = "/home/svu/e0072438/PINN/%s"%purpose


    P_back, P, x, y, rho, u, v, T, Et, E = [],[],[],[],[],[],[],[],[],[]
    cnt = 0
    print("\nLoading mother dataset in %s..."%path)

    #check all files in the directory
    for f in os.listdir(path):
        #get correct file extension
        if f.startswith("bp") and f.endswith(".csv"):
            
            try:
                P_back_ , P_ , x_ , y_ , rho_, u_, v_, T_, Et_, E_ = [],[],[],[],[],[],[],[],[],[]

                #Obtain magnitude of back pressure in the file name
                P_b = float(f[f.index('=')+1:-4])
                cnt += 1
                print("\t Progress: %d/%d"%(cnt, len(os.listdir(path))))

                f = path + f

                with open(f, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter=',')
                    isFirst = True
                    nrow = 0
                    ncol = 0

                    for row in reader:
                        #ignore header row
                        if isFirst:
                            isFirst = False
                            continue

                        # if abs(float(row[1])) < 0.001:
                        #     nrow += 1

                        if (nrow%filter ==0):
                            #column 1: node number
                            #column 2: x-coordinate
                            #column 3: y-coordinate
                            #column 4: pressure
                            #column 5: density
                            #column 6: x-velocity, u
                            #column 7: y-velocity, v
                            #column 8: temperature
                            #column 9: total temperature
                            #column 10: total energy
                            #column 11: internal energy

                            if (ncol % filter ==0 and abs(float(row[2]))>0.00001):

                                P_back_.append(float(P_b))
                                x_.append(float(row[1]))
                                y_.append(float(row[2]))
                                P_.append(float(row[3]))
                                rho_.append(float(row[4]))
                                u_.append(float(row[5]))
                                v_.append(float(row[6]))
                                T_.append(float(row[7]))
                                Et_.append(float(row[9]))
                                E_.append(float(row[10]))
                                
                                # P_back.append(float(P_b))
                                # x.append(float(row[1]))
                                # y.append(float(row[2]))
                                # P.append(float(row[3]))
                                # rho.append(float(row[4]))
                                # u.append(float(row[5]))
                                # v.append(float(row[6]))
                                # T.append(float(row[7]))
                                # Et.append(float(row[9]))
                                # E.append(float(row[10]))



                        ncol +=1    
                        nrow +=1

                print("\t\t Number of nodes: %d"%(len(P_back_)))
                P_back.append(P_back_)
                x.append(x_)
                y.append(y_)
                P.append(P_)
                rho.append(rho_)
                u.append(u_)
                v.append(v_)
                T.append(T_)
                Et.append(Et_)
                E.append(E_)


            except:
                print("\t !! Warning %s contains invalid filename"%f)
                pass

    P_back  = np.asarray(P_back)
    x       = np.asarray(x)
    y       = np.asarray(y)
    P       = np.asarray(P)
    rho     = np.asarray(rho)
    u       = np.asarray(u)
    v       = np.asarray(v)
    T       = np.asarray(T)
    Et      = np.asarray(Et)
    E       = np.asarray(E)

    print("Successfully loaded %d dataset(s) in %s\n" %(cnt,path))
    return P_back,x,y,P,rho,u,v,T,Et,E

def writeData(path,x,y,P,rho,u,v,Et):
    
    #column 1: node number
    #column 2: x-coordinate
    #column 3: y-coordinate
    #column 4: pressure
    #column 5: density
    #column 6: x-velocity, u
    #column 7: y-velocity, v
    #column 8: temperature
    #column 9: total temperature
    #column 10: total energy
    #column 11: internal energy
    e = np.zeros(x.shape[0]).flatten()[:,None]
    data = np.concatenate((e,x,y,P,rho,u,v,e,e,Et,e),axis = 1)

    with open(path, 'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')

        #write header
        writer.writerow(['Node number',
                        'x-coordinate',
                        'y-coordinate',
                        'pressure',
                        'density',
                        'x-velocity',
                        'y-velocity',
                        'temperature',
                        'total temperature',
                        'total energy',
                        'internal energy',])


        #write data
        writer.writerows(data)

    print("Successfully saved predicted data in %s\n" %(path))

def loadCSV(path):
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        isFirst = True
        nrow = 0
        ncol = 0
        P_back, P, x, y, rho, u, v, T, Et, E = [],[],[],[],[],[],[],[],[],[]
        P_b = float(path[path.index('=')+1:-4])

        for row in reader:
            #ignore header row
            if isFirst:
                isFirst = False
                continue

            if (abs(float(row[2]))>=0.0001):
                P_back.append(float(P_b))
                x.append(float(row[1]))
                y.append(float(row[2]))
                P.append(float(row[3]))
                rho.append(float(row[4]))
                u.append(float(row[5]))
                v.append(float(row[6]))
                T.append(float(row[7]))
                Et.append(float(row[9]))
                E.append(float(row[10]))

            ncol +=1    
            nrow +=1


    P_back  = np.asarray(P_back)
    x       = np.asarray(x)
    y       = np.asarray(y)
    P       = np.asarray(P)
    rho     = np.asarray(rho)
    u       = np.asarray(u)
    v       = np.asarray(v)
    T       = np.asarray(T)
    Et      = np.asarray(Et)
    E       = np.asarray(E)


    return P_back.T,x.T,y.T,P.T,rho.T,u.T,v.T,T.T,Et.T,E.T

def getWallIndex( x,y, first_x, first_y, n, direction):
    print("Getting wall index... this may take awhile...")
    x = x.flatten()
    y = y.flatten()

    x_wall = []
    y_wall = []
    
    target_x = first_x
    target_y = first_y

    x_ = x
    y_ = y

    while len(x_wall) < n:    
        min_dist = 9999
        index = 0
        for i in range(len(x_)):
            dist = ((x_[i] - target_x)**2 + (y_[i] - target_y)**2)**0.5
            if direction == "horizontal":
                dx = (x_[i] - target_x)
                if dist < min_dist and dx > 0.002:
                    min_dist = dist
                    index = i

            elif direction == "vertical":
                dy = (y_[i] - target_y)
                if dist < min_dist and dy>0.001 :
                    min_dist = dist
                    index = i
            
        target_x = x_[index]
        target_y = y_[index]

        x_ = np.delete(x_,index)
        y_ = np.delete(y_,index)

        x_wall.append(target_x)
        y_wall.append(target_y)


    print("Obtained wall index... ")
    return np.array(x_wall), np.array(y_wall)


def writeLoss(path,loss_vector,ts_vector):

    loss_vector = np.array(loss_vector).flatten()[:,None]
    ts_vector   = np.array(ts_vector).cumsum().flatten()[:,None]
    
    data = np.concatenate((ts_vector,loss_vector),axis = 1)

    with open(path,'w') as csvfile:
        writer = csv.writer(csvfile, delimiter=',')
        writer.writerow(['timestamp','loss'])
        writer.writerows(data)


def plotLoss(path,x_lower_lim,x_upper_lim,y_lower_lim,y_upper_lim):
    x,y = [],[]
    with open(path,'r') as csvfile:
        reader = csv.reader(csvfile,delimiter=',')
        isFirst = True
        for row in reader:
            #ignore header row
            if isFirst:
                isFirst = False
                continue

            # if float(row[1]) > 10**4:
            #     continue 

            x.append(float(row[0]))
            y.append(float(row[1]))
        
    plt.plot(x,y)
    plt.title('Loss Curve')
    plt.xlim((x_lower_lim,x_upper_lim))
    plt.ylim((y_lower_lim,y_upper_lim))
    plt.xlabel('Iteration')
    plt.ylabel('Loss value')

    if save:
        plt.savefig('%s_loss.png'%name)
    else:
        plt.show()

def plotAll(path,x,y,P,rho,u,v,Et,P_pred,rho_pred,u_pred,v_pred,Et_pred):
    p_max   = max(P.max(),P_pred.max())
    p_min   = min(P.min(),P_pred.min())
    rho_max = max(rho.max(),rho_pred.max())
    rho_min = min(rho.min(),rho_pred.min())
    u_max   = max(u.max(),u_pred.max())
    u_min   = min(u.min(),u_pred.min())
    v_max   = max(v.max(),v_pred.max())
    v_min   = min(v.min(),v_pred.min())
    Et_max  = max(Et.max(),Et_pred.max())
    Et_min  = min(Et.min(),Et_pred.min())


    plt.figure(1)
    plt.subplot(3, 1, 1)
    plt.title('Pressure plot (true value)')
    plt.scatter(x,y, c=P,vmin=p_min,vmax=p_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.title('Pressure plot (predicted value)')
    plt.scatter(x,y, c=P_pred,vmin=p_min,vmax=p_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.title('Error plot   ')
    plt.scatter(x,y, c=(P_pred-P)/P,vmin=-1,vmax=1,cmap='seismic')
    plt.colorbar()
    plt.axis('off')
    if save:
        plt.savefig('%s_1.png'%name)

    plt.figure(2)
    plt.subplot(3, 1, 1)
    plt.title('Density plot (true value)')
    plt.scatter(x,y, c=rho,vmin=rho_min,vmax=rho_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.title('Density plot (predicted value)')
    plt.scatter(x,y, c=rho_pred,vmin=rho_min,vmax=rho_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.title('Error plot   ')
    plt.scatter(x,y, c=(rho_pred-rho)/rho,vmin=-1,vmax=1,cmap='seismic')
    plt.colorbar()
    plt.axis('off')
    if save:
        plt.savefig('%s_2.png'%name)

    plt.figure(3)
    plt.subplot(3, 1, 1)
    plt.title('Velocity u plot (true value)')
    plt.scatter(x,y, c=u,vmin=u_min,vmax=u_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.title('Velocity u plot (predicted value)')
    plt.scatter(x,y, c=u_pred,vmin=u_min,vmax=u_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.title('Error plot   ')
    plt.scatter(x,y, c=(u_pred-u)/u,vmin=-1,vmax=1,cmap='seismic')
    plt.colorbar()
    plt.axis('off')
    if save:
        plt.savefig('%s_3.png'%name)

    plt.figure(4)
    plt.subplot(3, 1, 1)
    plt.title('Velocity v plot (true value)')
    plt.scatter(x,y, c=v,vmin=v_min,vmax=v_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.title('Velocity v plot (predicted value)')
    plt.scatter(x,y, c=v_pred,vmin=v_min,vmax=v_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.title('Error plot   ')
    plt.scatter(x,y, c=(v_pred-v)/v,vmin=-1,vmax=1,cmap='seismic')
    plt.colorbar()
    plt.axis('off')
    if save:
        plt.savefig('%s_4.png'%name)

    plt.figure(5)
    plt.subplot(3, 1, 1)
    plt.title('Total Energy plot (true value)')
    plt.scatter(x,y, c=Et,vmin=Et_min,vmax=Et_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 2)
    plt.title('Energy plot (predicted value)')
    plt.scatter(x,y, c=Et_pred,vmin=Et_min,vmax=Et_max)
    plt.axis('off')
    plt.colorbar()
    plt.subplot(3, 1, 3)
    plt.title('Error plot   ')
    plt.scatter(x,y, c=(Et_pred-Et)/Et,vmin=-1,vmax=1,cmap='seismic')
    plt.colorbar()
    plt.axis('off')
    if save:
        plt.savefig('%s_5.png'%name)

    plot_x = []
    center_p_true = []
    center_p_pred = []
    center_rho_true = []
    center_rho_pred = []
    center_u_true = []
    center_u_pred = []
    center_v_true = []
    center_v_pred = []
    center_Et_true = []
    center_Et_pred = []

    ##Centerline plot

    for i in range(len(y)):
        if abs(y[i] - min(y)) < 0.005:
            plot_x.append(x[i])
            center_p_true.append(P[i]) 
            center_p_pred.append(P_pred[i]) 
            center_rho_true.append(rho[i]) 
            center_rho_pred.append(rho_pred[i]) 
            center_u_true.append(u[i]) 
            center_u_pred.append(u_pred[i]) 
            center_v_true.append(v[i]) 
            center_v_pred.append(v_pred[i]) 
            center_Et_true.append(Et[i]) 
            center_Et_pred.append(Et_pred[i]) 

        else:
            break

    #100 sample for plotting only
    A = np.random.choice(range(len(plot_x)), size=(100,), replace=False)
    plot_x              = np.array(plot_x)[A]
    center_p_true       = np.array(center_p_true)[A] 
    center_p_pred       = np.array(center_p_pred)[A]
    center_rho_true     = np.array(center_rho_true)[A]
    center_rho_pred     = np.array(center_rho_pred)[A]
    center_u_true       = np.array(center_u_true)[A]
    center_u_pred       = np.array(center_u_pred)[A]
    center_v_true       = np.array(center_v_true)[A]
    center_v_pred       = np.array(center_v_pred)[A]
    center_Et_true       = np.array(center_Et_true)[A]
    center_Et_pred       = np.array(center_Et_pred)[A]
    
    p = plot_x.argsort()

    plt.figure(6)
    plt.title('Pressure')
    plt.plot(plot_x[p],center_p_true[p], '-r')
    plt.plot(plot_x[p],center_p_pred[p], 'xb')
    plt.legend(['True value','Predicted value'],loc="best")
    if save:
        plt.savefig('%s_6.png'%name)

    plt.figure(7)
    plt.title('Density')
    plt.plot(plot_x[p],center_rho_true[p], '-r')
    plt.plot(plot_x[p],center_rho_pred[p], 'xb')
    plt.legend(['True value','Predicted value'],loc="best")
    if save:
        plt.savefig('%s_7.png'%name)

    plt.figure(8)
    plt.title('Velocity u')
    plt.plot(plot_x[p],center_u_true[p], '-r')
    plt.plot(plot_x[p],center_u_pred[p], 'xb')
    plt.legend(['True value','Predicted value'],loc="best")
    if save:
        plt.savefig('%s_8.png'%name)

    plt.figure(9)
    plt.title('Velocity v')
    plt.plot(plot_x[p],center_v_true[p], '-r')
    plt.plot(plot_x[p],center_v_pred[p], 'xb')
    plt.legend(['True value','Predicted value'],loc="best")
    if save:
        plt.savefig('%s_9.png'%name)

    plt.figure(10)
    plt.title('Energy')
    plt.plot(plot_x[p],center_Et_true[p], '-r')
    plt.plot(plot_x[p],center_Et_pred[p], 'xb')
    plt.legend(['True value','Predicted value'],loc="best")
    if save:
        plt.savefig('%s_10.png'%name)
    else:
        plt.show()


def plotAll_1D(path,x,P,rho,u,Et,P_pred,rho_pred,u_pred,Et_pred):

    plt.figure(1)
    plt.title('Pressure')
    plt.plot(x,P, '-r')
    plt.plot(x,P_pred, 'xb')
    plt.legend(['True value','Predicted value'],loc="best")

    plt.figure(2)
    plt.title('Density')
    plt.plot(x,rho, '-r')
    plt.plot(x,rho_pred, 'xb')
    plt.legend(['True value','Predicted value'],loc="best")

    plt.figure(3)
    plt.title('Velocity u')
    plt.plot(x,u, '-r')
    plt.plot(x,u_pred, 'xb')
    plt.legend(['True value','Predicted value'],loc="best")

    plt.figure(4)
    plt.title('Energy')
    plt.plot(x,Et, '-r')
    plt.plot(x,Et_pred, 'xb')
    plt.legend(['True value','Predicted value'],loc="best")


    plt.show()


if __name__ == '__main__':
    loadData('train')



