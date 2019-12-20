import tensorflow as tf
import numpy as np
import time
import utils as utils 
from tensorflow.contrib.layers import fully_connected

class PINN_2D:
    # Initialize the class
    def __init__(self, P_back, x, y, P, rho, u, v, Et,\
                    P_back_w, x_w, y_w, P_w, rho_w, u_w, v_w, Et_w, \
                    x_outlet, y_outlet, \
                    layers, learning_rate):
        self.sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(allow_soft_placement=True,
                                                                        log_device_placement=True))
        
        self.loss_vector, self.step_vector = [], []
       
        self.learning_rate = learning_rate
        
        self.P_back     = P_back
        self.x          = x
        self.y          = y
        self.P          = P
        self.rho        = rho
        self.u          = u
        self.v          = v
        self.Et         = Et

        self.P_back_w     = P_back_w
        self.x_w          = x_w
        self.y_w          = y_w
        self.P_w          = P_w
        self.rho_w        = rho_w
        self.u_w          = u_w
        self.v_w          = v_w
        self.Et_w         = Et_w

        self.x_outlet     = x_outlet
        self.y_outlet     = y_outlet

        self.P_back_tf      = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_tf           = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y_tf           = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.P_tf           = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.rho_tf         = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.u_tf           = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.v_tf           = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.Et_tf          = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.x_w_tf         = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y_w_tf         = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) 
        self.P_w_tf         = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) 
        self.u_w_tf         = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) 
        self.v_w_tf         = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) 
        self.Et_w_tf         = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) 
        self.x_outlet_tf    = tf.compat.v1.placeholder(tf.float32, shape=[None, 1])
        self.y_outlet_tf    = tf.compat.v1.placeholder(tf.float32, shape=[None, 1]) 
        
        #Neural Network
        self.actf              = tf.tanh
        self.input_layer       = tf.concat([self.P_back_tf, self.x_tf, self.y_tf], 1) 
        self.hidden_layer_1    = fully_connected(self.input_layer   ,layers[0],activation_fn= self.actf,scope = 'l1',reuse=False)
        self.hidden_layer_2    = fully_connected(self.hidden_layer_1,layers[1],activation_fn= self.actf,scope = 'l2',reuse=False)
        self.hidden_layer_3    = fully_connected(self.hidden_layer_2,layers[2],activation_fn= self.actf,scope = 'l3',reuse=False)
        self.output_pred       = fully_connected(self.hidden_layer_3,5,activation_fn= self.actf,scope = 'l4',reuse=False)
        self.output_true       = tf.concat([self.P_tf,self.rho_tf,self.u_tf,self.v_tf,self.Et_tf],1)

        self.e_P        = tf.reduce_sum(tf.square((self.output_true[:,0:1] - self.output_pred[:,0:1])))
        self.e_rho      = tf.reduce_sum(tf.square((self.output_true[:,1:2] - self.output_pred[:,1:2])))
        self.e_u        = tf.reduce_sum(tf.square((self.output_true[:,2:3] - self.output_pred[:,2:3])))
        self.e_v        = tf.reduce_sum(tf.square((self.output_true[:,3:4] - self.output_pred[:,3:4])))
        self.e_Et       = tf.reduce_sum(tf.square((self.output_true[:,4:5] - self.output_pred[:,4:5])))

        #PINN
        self.e_1, self.e_2, self.e_3, self.e_4, self.e_5 = self.NavierStokeEquation()
        self.e_1    = tf.reduce_mean(tf.square(self.e_1))
        self.e_2    = tf.reduce_mean(tf.square(self.e_2))
        self.e_3    = tf.reduce_mean(tf.square(self.e_3))
        self.e_4    = tf.reduce_mean(tf.square(self.e_4))
        self.e_5    = tf.reduce_mean(tf.square(self.e_5))


        #BC at wall
        self.input_layer_2       = tf.concat([self.P_back_tf, self.x_w_tf, self.y_w_tf], 1) 
        self.hidden_layer_2_1    = fully_connected(self.input_layer_2   ,layers[0],activation_fn= self.actf,scope='l1',reuse=True)
        self.hidden_layer_2_2    = fully_connected(self.hidden_layer_2_1,layers[1],activation_fn= self.actf,scope='l2',reuse=True)
        self.hidden_layer_2_3    = fully_connected(self.hidden_layer_2_2,layers[2],activation_fn= self.actf,scope='l3',reuse=True)
        self.output_pred_2       = fully_connected(self.hidden_layer_2_3,5,activation_fn= self.actf ,scope='l4',reuse=True)

        self.e_wall_1 = tf.reduce_mean(tf.square(self.u_w_tf - self.output_pred_2[:,2:3]))    #u 
        self.e_wall_2 = tf.reduce_mean(tf.square(self.v_w_tf - self.output_pred_2[:,3:4]))    #v 
        self.e_wall_3 = tf.reduce_mean(tf.square(self.P_w_tf - self.output_pred_2[:,0:1])) # dP/dn = 0

        #BC at outlet
        self.input_layer_3       = tf.concat([self.P_back_tf, self.x_outlet_tf, self.y_outlet_tf], 1) 
        self.hidden_layer_3_1    = fully_connected(self.input_layer_3   ,layers[0],activation_fn= self.actf,scope='l1',reuse=True)
        self.hidden_layer_3_2    = fully_connected(self.hidden_layer_3_1,layers[1],activation_fn= self.actf,scope='l2',reuse=True)
        self.hidden_layer_3_3    = fully_connected(self.hidden_layer_3_2,layers[2],activation_fn= self.actf,scope='l3',reuse=True)
        self.output_pred_3       = fully_connected(self.hidden_layer_3_3,5,activation_fn= self.actf ,scope='l4',reuse=True)

        self.e_outlet_1 = tf.reduce_mean(tf.square(self.output_pred_3[:,0:1] -self.P_back_tf))

        self.loss   =   1*self.e_P      +\
                        1*self.e_rho  +\
                        1*self.e_u    +\
                        1*self.e_v    +\
                        1*self.e_Et +\
                        1*self.e_1    +\
                        1*self.e_2    +\
                        1*self.e_3    +\
                        1*self.e_4    +\
                        1*self.e_5    +\
                        0*self.e_wall_1 +\
                        0*self.e_wall_2 +\
                        0*self.e_wall_3 +\
                        1*self.e_outlet_1

        self.optimizer_Adam = tf.compat.v1.train.AdamOptimizer(learning_rate=self.learning_rate)
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)   

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss, 
                                                                method = 'L-BFGS-B', 
                                                                options = {'maxiter': 15000,
                                                                           'maxfun': 15000,
                                                                           'maxcor': 50,
                                                                           'maxls': 50,
                                                                           'ftol' : 1.0 * np.finfo(float).eps})        
        
                 
        
        self.saver = tf.train.Saver(save_relative_paths=True)

    def NavierStokeEquation(self):
        x   = self.x_tf
        y   = self.y_tf
        P_b = self.P_back_tf
        P   = self.output_pred[:,0:1]
        rho = self.output_pred[:,1:2]
        u   = self.output_pred[:,2:3]
        v   = self.output_pred[:,3:4]
        Et  = self.output_pred[:,4:5]

        # autodiff residual 1
        e_1 = tf.gradients(rho*u, x)[0] + tf.gradients(rho*v, y)[0]
        # autodiff residual 2
        e_2 = tf.gradients(rho*u*u + P, x)[0] + tf.gradients(rho*u*v, y)[0]
        # autodiff residual 3
        e_3 = tf.gradients(rho*v*u, x)[0] + tf.gradients(rho*v*v + P, y)[0]
        # autodiff residual 4
        e_4 = tf.gradients(1.225*rho*Et*u + P*u, x)[0] + tf.gradients(1.225*rho*Et*v + P*v, y)[0]
     
        #ideal gas equation
        gamma = 1.4
        e_5 = P - (gamma -1)*rho*(1.225*Et - 0.5*u**2 - 0.5*v**2) 

        return e_1, e_2, e_3, e_4, e_5




    def callback(self, loss):
        self.loss_vector.append(loss)
        self.step_vector.append(1)
        print('Loss: %.3e' % (loss))

    def train(self, num_epochs, num_iter, batch_size):
        self.batch_size = batch_size

        init = tf.global_variables_initializer()
        self.sess.run(init)
        switch = True

        # for epoch in range(num_epochs):
        for epoch in range(num_epochs):

            start_time = time.time()

            if batch_size <= self.P.shape[0]:
                A = np.random.choice(range(self.P.shape[0]), size=(batch_size,), replace=False)
            else:
                A = np.random.choice(range(self.P.shape[0]), size=(batch_size,), replace=True)
            
            if batch_size <= self.x_w.shape[0]:
                B = np.random.choice(range(self.x_w.shape[0]), size=(batch_size,), replace=False)
            else:
                B = np.random.choice(range(self.x_w.shape[0]), size=(batch_size,), replace=True)
            
            if batch_size <= self.x_outlet.shape[0]:
                C = np.random.choice(range(self.x_outlet.shape[0]), size=(batch_size,), replace=False)
            else:
                C = np.random.choice(range(self.x_outlet.shape[0]), size=(batch_size,), replace=True)
            

            for it in range(num_iter):
                # if it == 0 and epoch == 5 :
                #     switch = False
                #     self.loss   =   1*self.e_P      +\
                #                     1*self.e_rho  +\
                #                     1*self.e_u    +\
                #                     1*self.e_v    +\
                #                     1*self.e_Et +\
                #                     1*self.e_1    +\
                #                     1*self.e_2    +\
                #                     1*self.e_3    +\
                #                     1*self.e_4    +\
                #                     1*self.e_5    +\
                #                     0*self.e_wall_1 +\
                #                     0*self.e_wall_2 +\
                #                     0*self.e_wall_3 +\
                #                     1*self.e_outlet_1

                #     self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)   

                #for it in range(0,N_nodes,batch_size):
                #node_idx = nodes_perm[np.arange(it,it+batch_size)]

                #slice data & add boundary data
                P_back_batch    = self.P_back[A].flatten()[:,None]
                x_batch         = self.x[A].flatten()[:,None]
                y_batch         = self.y[A].flatten()[:,None]
                P_batch         = self.P[A].flatten()[:,None]
                rho_batch       = self.rho[A].flatten()[:,None]
                u_batch         = self.u[A].flatten()[:,None]
                v_batch         = self.v[A].flatten()[:,None]
                Et_batch        = self.Et[A].flatten()[:,None]
                
                x_w_batch       = self.x_w[B].flatten()[:,None]
                y_w_batch       = self.y_w[B].flatten()[:,None]
                P_w_batch       = self.P_w[B].flatten()[:,None]
                u_w_batch       = self.u_w[B].flatten()[:,None]
                v_w_batch       = self.v_w[B].flatten()[:,None]
                Et_w_batch      = self.Et_w[B].flatten()[:,None]

                x_outlet_batch  = self.x_outlet[C].flatten()[:,None]
                y_outlet_batch  = self.y_outlet[C].flatten()[:,None]

                tf_dict = {self.P_back_tf: P_back_batch, self.x_tf: x_batch, self.y_tf: y_batch,
                            self.x_w_tf: x_w_batch, self.y_w_tf: y_w_batch, self.P_w_tf: P_w_batch, self.u_w_tf: u_w_batch, self.v_w_tf: v_w_batch, self.Et_w_tf: Et_w_batch,
                            self.x_outlet_tf: x_outlet_batch, self.y_outlet_tf: y_outlet_batch,
                            self.P_tf: P_batch, self.rho_tf: rho_batch, self.u_tf: u_batch, self.v_tf: v_batch,self.Et_tf: Et_batch}
        
                self.sess.run(self.train_op_Adam, tf_dict)

                # Print
                if it % 100 == 0:

                    elapsed = time.time() - start_time

                    loss_value = self.sess.run([self.loss],tf_dict)[0]

                    e_1, e_2, e_3,e_4,e_5 = self.sess.run([self.e_1,self.e_2,self.e_3,self.e_4, self.e_5],tf_dict)
                    e_P, e_rho, e_u,e_v,e_Et = self.sess.run([self.e_P,self.e_rho,self.e_u,self.e_v,self.e_Et],tf_dict)
                    e_wall_1, e_wall_2, e_wall_3, e_outlet_1  = self.sess.run([self.e_wall_1,self.e_wall_2, self.e_wall_3, self.e_outlet_1],tf_dict)
                    
                        #self.train_op_Adam = self.optimizer_Adam.minimize(self.loss) 

                    # res1 = self.sess.run(self.e1, tf_dict)
                    # res2 = self.sess.run(self.e2, tf_dict)
                    #res3 = self.sess.run(self.total_res, tf_dict)
                    #print(res3)
                    print('Epoch: %d, It: %d, Loss: %.3e, Time: %.2f' % 
                        (epoch, it, loss_value, elapsed))

                    print("\tE_1: %.3f, E_2: %.3f, E_3: %.3f, E_4: %.3f, E_5: %.3f"%
                        (e_1,e_2,e_3,e_4,e_5))

                    print("\tE_P: %.3f, E_rho: %.3f, E_u: %.3f, E_v: %.3f, E_Et: %.3f"%
                        (e_P,e_rho,e_u,e_v,e_Et))

                    print("\tE_wall_1: %.3f, E_wall_2: %.3f, E_wall_3: %.3f, E_outlet_1: %.3f"%
                        (e_wall_1,e_wall_2,e_wall_3, e_outlet_1))

                    # print('Mass Residual: %f\t\tMomentum Residual: %f\tEnergy Residual: %f'
                    #     %(sum(map(lambda a:a*a,res1))/len(res1), sum(map(lambda a:a*a,res2))/len(res2), sum(map(lambda a:a*a,res3))/len(res3)))
                    
                    start_time = time.time()
                    self.saver.save(self.sess,self.ckpt_name,global_step = epoch)

                    self.loss_vector.append(loss_value)
                    self.step_vector.append(1)

                if (epoch % 5 == 0 and it == 0):
                    path2 = self.ckpt_name + '_temp_loss.csv'
                    utils.writeLoss(path2,self.loss_vector,self.step_vector)
    
        self.optimizer.minimize(self.sess,
                            feed_dict = tf_dict,
                            fetches = [self.loss],
                            loss_callback = self.callback)


            
    def predict(self, P_back_test, x_test, y_test):
        writer = tf.summary.FileWriter("./output",self.sess.graph)
        tf_dict     = {self.P_back_tf: P_back_test, self.x_tf: x_test, self.y_tf: y_test}
        output_test = self.sess.run(self.output_pred,tf_dict)
        
        P_test      = output_test[:,0:1]
        rho_test    = output_test[:,1:2]
        u_test      = output_test[:,2:3]
        v_test      = output_test[:,3:4]
        Et_test     = output_test[:,4:5]
        writer.close()
        return P_test, rho_test, u_test, v_test, Et_test
