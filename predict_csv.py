import utils as utils
import numpy as np
import os
import csv
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('case', help='Case ID')
parser.add_argument('plot', help='Plot or not(key in 1 to plot)')
args = parser.parse_args()

project_name = args.case 
model_path = os.getcwd() + '/2d_inviscid_model/%s/'%project_name

for f in os.listdir(model_path):
    if (f.endswith('.csv') and f[-5] != 's'):
        pb = (f[f.index('=')+1:-4])
        
test_path       = os.getcwd() + '/data/bp=%s.csv'%pb
predict_path    = model_path + '%s_bp=%s.csv' %(project_name,pb)

P_back_test,x_test,y_test,P_test,rho_test,u_test,v_test,T_test,Et_test,E_test = utils.loadCSV(test_path)
P_back_pred,x_pred,y_pred,P_pred,rho_pred,u_pred,v_pred,T_pred,Et_pred,E_pred = utils.loadCSV(predict_path)

#Error
error_P = np.linalg.norm(P_test-P_pred,2)/np.linalg.norm(P_test,2)
print("Test Error in P: "+str(error_P))
error_rho = np.linalg.norm(rho_test-rho_pred,2)/np.linalg.norm(rho_test,2)
print("Test Error in rho: "+str(error_rho))
error_u = np.linalg.norm(u_test-u_pred,2)/np.linalg.norm(u_test,2)
print("Test Error in u: "+str(error_u))
error_v = np.linalg.norm(v_test-v_pred,2)/np.linalg.norm(v_test,2)
print("Test Error in v: "+str(error_v))
error_Et = np.linalg.norm(Et_test- Et_pred,2)/np.linalg.norm(Et_test,2)
print("Test Error in E: "+str(error_Et))
error_T = np.linalg.norm(T_test- T_pred,2)/np.linalg.norm(T_test,2)
print("Test Error in T: "+str(error_T))

if(args.plot == "1"):
    utils.plotLoss(model_path + '%s_bp=%s_loss.csv'%(project_name,pb), 0,3000,0,20)
    utils.plotAll(model_path,
                    x_test,
                    y_test,
                    P_test,
                    rho_test,
                    u_test,
                    v_test,
                    Et_test,
                    P_pred,
                    rho_pred,
                    u_pred,
                    v_pred,
                    Et_pred)
