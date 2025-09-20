import os
pid = os.getpid()
print(f"Current process ID: {pid}")
current_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = os.path.join(current_dir, f"0g_1tp_{pid}")
print(output_dir)
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
import numpy as np
from sklearn.linear_model import OrthogonalMatchingPursuit
from numpy import linalg as LA
import scipy.sparse as sparse
from scipy.sparse import csc_matrix
from scipy.sparse import dia_matrix
import itertools
import operator
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
import scipy.io as sio
import itertools
import random
#rhs_des = ['','u', 'u_xx.', 'u_t.']
rhs_des = ['','u', 'u**2', 'u**3', 'u_x', 'u*u_x', 'u**2*u_x',
                    'u3*u_x', 'u_xx.','u*u_xx', 'u**2*u_xx', 'u3*u_xx', 'u_t.', 'u*u_t', 'u**2*u_t', 'u3*u_t']
def print_pde(w, rhs_description, ut = 'u_tt'):
        """
    Output the identified PDE in symbolic form.

    Parameters
    ----------
    w : np.ndarray or torch.Tensor
        Identified coefficients of candidate terms.
    rhs_description : list of str
        List of candidate term expressions (right-hand side descriptors).
        For example: ['','u','u**2','u_x','u*u_x', ...].
    ut : str, optional (default = 'u_tt')
        The left-hand side term of the PDE (e.g. 'u_t', 'u_tt').

    Returns
    -------
    None
        Prints the PDE in the form:
            ut = (coeff1)*term1 + (coeff2)*term2 + ...
    """
    if isinstance(w, np.ndarray):
        lambda_1_value = w
    else:
        lambda_1_value = w.detach().cpu().numpy() 
    pde = ut + ' = '
    first = True
    for i in range(len(w)):
        if w[i] != 0:
            if not first:
                pde = pde + ' + '
            pde = pde + "(%05f )" % (w[i].real) + rhs_description[i] + "\n   "
            first = False
    print(pde)



import torch
from torch import nn
from torch.autograd import Variable

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.interpolate import griddata
import scipy.io

from typing import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

print(f"Using device: {device}")

# Store parameters
lambda_list = []
lambda_list2 = []
loss_f_list=[]
loss_m_list=[]
loss_ut_list=[]
loss_l_list=[]
term_list = []
loss_record_list = []
diff_list = []


seed = 3428   # random seed 
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

class Net(nn.Module):
    def makedense(self,myinput,myoutput):
        """
        Create a simple dense (fully connected) layer.

        Args:
            myinput (int): number of input features
            myoutput (int): number of output features

        Returns:
            torch.nn.Sequential: a sequential container with a single linear layer
        """        
        dense=torch.nn.Sequential(torch.nn.Linear(myinput,myoutput),)
        return dense
    
    def __init__(self, layers):
        """
        Initialize the neural network with given layer sizes.

        Args:
            layers (list[int]): a list defining the number of neurons in each layer.
                                Example: [input_dim, hidden1, hidden2, ..., output_dim]
        """
        super(Net, self).__init__()
        layers_seq = list()
        self.lin1=nn.Linear(layers[0], layers[1])
        self.lin1.weight.data.normal_(0,1)
        self.lin1.bias.data.fill_(0.)
        self.lin1.weight.requires_grad = True
        self.lin1.bias.requires_grad = True
        self.layers_cnt = len(layers) - 1
        for i in range(1,len(layers) - 2):
            layers_seq.append(('hidden_layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            layers_seq.append(('actvation_%d' % i, nn.Tanh()))
        layers_seq.append(('output_layer', nn.Linear(layers[-2], layers[-1])))

        self.layers = nn.Sequential(OrderedDict(layers_seq))
    def forward(self, x):
        """
        Forward pass through the network.

        Args:
            x (torch.Tensor): input tensor

        Returns:
            torch.Tensor: network output
        """
        out = torch.sin(self.lin1(x))
        output = self.layers(out)
        return output



class PINNInference():
    def __init__(self, 
                 x_max=1,
                 t_max=1,
                 measurement=None,
                 f=None,
                 init_xt=None,
                 layers=None,
                 Nf = 0,
                 a=0., b=0., c=0., d=0.,
                 device=None):
        """
        Initialize the Physics-Informed Neural Network (PINN) inference object.

        Args:
            x_max (float): maximum value of x for normalization
            t_max (float): maximum value of t for normalization
            measurement (np.array or torch.Tensor): observation points in the training set, shape (N, 4) [x, t, u, real]
            f (np.array or torch.Tensor): interior points in the training set for computing PDE residuals, shape (Nf, 3) [x, t, u]
            init_xt (np.array or torch.Tensor): initial condition points (x, t)
            layers (list[int]): network architecture defining neurons in each layer
            Nf (int): number of interior points for PDE residual evaluation
            a, b, c, d (float): coefficients for normalization
            device (torch.device): device to store tensors (CPU or GPU)
        """
        self.device=device        
        
        if measurement is not None:
            self.m_x = torch.tensor(measurement[:, 0:1], requires_grad=True).float().to(self.device)
            self.m_t = torch.tensor(measurement[:, 1:2], requires_grad=True).float().to(self.device)  
            self.m_u = torch.tensor(measurement[:, 2:3]).float().to(device)
            self.real = torch.tensor(measurement[:, 3:4]).float().to(device)

        if f is not None:
            self.f_x = torch.tensor(f[:, 0:1], requires_grad=True).float().to(self.device)
            self.f_t = torch.tensor(f[:, 1:2], requires_grad=True).float().to(self.device)  

        if init_xt is not None:
            self.init_x = torch.tensor(init_xt[:, 0:1], requires_grad=True).float().to(self.device)
            self.init_t = torch.tensor(init_xt[:, 1:2], requires_grad=True).float().to(self.device)

        self.layers = layers
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x_max = x_max
        self.t_max = t_max
        self.Nf = Nf
        self.gaussian_matrix =  torch.randn(self.Nf, self.Nf) 
        
        self.lambda_1 = torch.zeros(16, 1, requires_grad=True).to(device)
        
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        
        self.dnn = Net(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1) 
        self.loss_f_record = 1
        self.loss_record = 1
        self.optimizer = None        
        self.iter = 0
        

    def net_u(self, x, t):  
        x1 = 0.2*(x-self.a)/self.b
        t1 = 0.2*(t-self.c)/self.d
        uu = self.dnn(torch.cat([x1, t1], dim=1))
        u = uu[:,0:1]
        return u
    
    def net_f(self, x ,t):
        """ pde residual
        """
        # self.lambda_1 = torch.flatten(self.dnn.branch_conv[0].weight).unsqueeze(1) #self.dnn.layers.hidden_layer_1.bias.unsqueeze(1)
        #lambda_1 = self.lambda_1 ** 2    
        u = self.net_u(x,t)
        v = torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]#.unsqueeze(-1)
        v_t = torch.autograd.grad(
            v, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        w = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]#.unsqueeze(-1)
        w_x = torch.autograd.grad(
            w, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]#.unsqueeze(-1)
        Phi = torch.cat([torch.ones_like(u), u,u**2,u**3, 
                         w,u*w,u**2*w,u**3*w,
                         w_x, u*w_x, u**2*w_x,u**3*w_x,
                         v,u*v,u**2*v,u**3*v], 1) 
        # Phi = torch.cat([torch.ones_like(u), u,w_x,v], 1) 

        u_tt = v_t
        f = u_tt- torch.matmul(Phi, self.lambda_1)
        return f,Phi,u_tt
    
    def train(self, nIter,lam=None,Ir=0.001,lambda1=1.0, lambda2=1.0): # nIter is the number of ADO loop
            
        for it in range(nIter):
            # Loop of Adam optimization            
            print('Adam begins')             
            self.train_inverse(niter=2000,Ir=Ir, plotting=True,lambda1=lambda1, lambda2=lambda2,plot_num=200)
            # Loop of STRidge optimization
            print(self.dnn.lambda_1)
            self.callTrainSTRidge()
            print(self.dnn.lambda_1)
    
    def train_inverse(self, niter=0,Ir=0.001, plotting=False, lambda1=1.0, lambda2=1.0,lambda3=0.00001,lambda4=1.0,plot_num=40,freezing=False):
        """ mse = lambda1 * mse_u + lambda2 * mse_f
        """
        self.optimizer = torch.optim.AdamW(self.dnn.parameters(), lr=Ir, weight_decay=1e-2)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = Ir 
        self.dnn.train()
        start_time = time.time()
        
        loss_list = []
        
        t_list = range(niter)

        
        mask = (model.lambda_1 == 0)        
        print(mask)
        for it in range(niter):
            f_pred,phi,u_tt = self.net_f(self.f_x, self.f_t) #function
            m_pred = self.net_u(self.m_x, self.m_t) #measurement
            
            loss_m = torch.mean((1+torch.abs(self.m_u))*(m_pred - self.m_u) ** 2)
            loss_f = torch.mean((1)*f_pred ** 2)
            loss_l = torch.sum(torch.abs(self.lambda_1))
            loss =  lambda1*loss_m + lambda2*loss_f + lambda3*loss_l
            
            self.optimizer.zero_grad()
            loss.backward()  
                
            self.optimizer.step()
            if freezing:
                with torch.no_grad():
                    self.lambda_1[mask] = 0

            loss_list.append(loss.item())
            
            lambda_sum = torch.sum(torch.abs(self.lambda_1))
            # 将 para 转换为 NumPy 数组（或如果希望存储 Tensor，可以省略 .numpy()）
            para = self.lambda_1.clone()
            para_numpy = para.detach().cpu().numpy()
#             diff = torch.mean((m_pred - self.real)**2)
            # 将 para 存入列表
            lambda_list.append(para_numpy)
            loss_f_list.append(loss_f.item())   
            loss_l_list.append(loss_l.item())  
            loss_m_list.append(loss_m.item()) 
#             loss_ut_list.append(loss_ut.item())  
#             diff_list.append(diff.item())  
#             # phi_norm = torch.mean(phi**2, dim=0).detach()
#             u_tt_norm = torch.mean(u_tt**2).detach()
#             phi_norm = torch.mean(phi**2, dim=0).detach()
#             para2 = para.T*(phi_norm)/u_tt_norm
#             para_numpy2 = para2.detach().cpu().numpy()
#             # 将 para 存入列表
#             lambda_list2.append(para_numpy2)

            if it % plot_num == 0:
                elapsed = time.time() - start_time
                print(
                    'Iter %d, Loss: %.5e, Loss_m: %.5e, Loss_f: %.5e,Loss_l: %.3e,lam:%.5e, time: %.2f' % ( \
                     it, loss.item(), loss_m.item(), loss_f.item(),loss_l.item(),lambda_sum.item(),elapsed)
                )

                start_time = time.time()
                #u_tt_norm = torch.mean(u_tt**2).detach()
                #phi_norm = torch.mean(phi**2, dim=0).detach()
                # print(phi_norm)
                print('log_loss_f_min',np.log(self.loss_f_record),'log_loss_min',np.log(self.loss_record))
        
        loss_record_list.append(model.loss_f_record)

        begin_iter = 0
        if plotting:
            t_array = np.array(t_list)[begin_iter:]
            loss_array = np.array(loss_list)[begin_iter:]
            plt.figure(figsize=(10, 8))
            plt.title('Train loss')
            plt.plot(t_array, loss_array, color='red', label='loss')
            plt.legend()
            plt.xlabel('iter')
            plt.ylabel('mse')
            plt.ylim((0, loss_array.max() + 0.01))
            plt.show()
            
    def callTrainSTRidge(self,lam = 1e-5,d_tol = 0.05,maxit = 50,STR_iters = 50,normalize = 2):
        lam = lam
        d_tol = d_tol
        maxit = maxit
        STR_iters = STR_iters
         
        l0_penalty = None
        
        normalize = normalize
        split = 0.8
        print_best_tol = True 
        print_ls = True
        f_pred,Phi_pred,u_tt_pred  = self.net_f(self.f_x,self.f_t)
         
        lambda2 = self.TrainSTRidge(Phi_pred, u_tt_pred, lam, d_tol, maxit, STR_iters, l0_penalty, normalize, split,print_best_tol,print_ls)     
         
        print('lambda2',lambda2)
        self.lambda_1 = torch.tensor(lambda2, requires_grad=True).float().to(device)
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.dnn.register_parameter('lambda_1', self.lambda_1)                  
    
    def TrainSTRidge(self, R1, Ut1, lam, d_tol, maxit, STR_iters = 10, l0_penalty = None, normalize = 2, split = 0.8, 
                     print_best_tol = False,print_ls = False):            
# =============================================================================
#        Inspired by Rudy, Samuel H., et al. "Data-driven discovery of partial differential equations."
#        Science Advances 3.4 (2017): e1602614.
# =============================================================================    
        R0 = R1.detach().cpu().numpy()   
        Ut = Ut1.detach().cpu().numpy()
        self.lambda_1 = self.lambda_1.detach().cpu().numpy()
        #lam = lam1.numpy()
        # First normalize data 
        n,d = R0.shape
        R = np.zeros((n,d), dtype=np.float32)
        if normalize != 0:
            Mreg = np.zeros((d,1))
            for i in range(0,d):
                Mreg[i] = 1.0/(np.linalg.norm(R0[:,i],normalize))
                R[:,i] = Mreg[i]*R0[:,i]                
            normalize_inner = 0
        else: 
            R = R0
            Mreg = np.ones((d,1))*d
            normalize_inner = 2
        
        # Split data into 80% training and 20% test, then search for the best tolderance.
        np.random.seed(0) # for consistancy
        n,_ = R.shape
        train = np.random.choice(n, int(n*split), replace = False)
        test = [i for i in np.arange(n) if i not in train]
        TrainR = R[train,:]
        TestR = R[test,:]
        TrainY = Ut[train,:]
        TestY = Ut[test,:]
    
        # Set up the initial tolerance and l0 penalty
        d_tol = float(d_tol)
        tol = d_tol
        if l0_penalty == None: l0_penalty = 0.001*np.linalg.cond(R)
                
        # Or inherit Lambda
        w_best = self.lambda_1/Mreg
        print('initialw',w_best)
        
        err_f = np.linalg.norm(TestY - TestR.dot(w_best), 2)
        err_lambda = l0_penalty*np.count_nonzero(w_best)
        err_best = err_f + err_lambda
        tol_best = 0
    
        # Now increase tolerance until test performance decreases
        for iter in range(maxit):
    
            # Get a set of coefficients and error
            w = self.STRidge(TrainR, TrainY, lam, STR_iters, tol, Mreg,w_best, normalize = normalize_inner)
            err_f = np.linalg.norm(TestY - TestR.dot(w), 2)
            err_lambda = l0_penalty*np.count_nonzero(w)
            err = err_f + err_lambda
    
            # Has the accuracy improved?
            if err <= err_best:
                err_best = err
                w_best = w
                tol_best = tol
                tol = tol + d_tol
    
            else:
                tol = max([0,tol - 2*d_tol])
                d_tol = d_tol/1.618
                tol = tol + d_tol
    
        if print_best_tol: print ("Optimal tolerance:", tol_best)
        
        if print_ls: 
            w_ls = np.linalg.lstsq(TrainR,TrainY)[0]  
            print ("pinv", w_ls)
        
        print(w_best)
                
        return np.real(np.multiply(Mreg, w_best))     
    
    def STRidge(self, X0, y, lam, maxit, tol, Mreg, w_best, normalize = 2, print_results = False):
    
        n,d = X0.shape
        X = np.zeros((n,d), dtype=np.complex64)
        # First normalize data
        if normalize != 0:
            Mreg = np.zeros((d,1))
            for i in range(0,d):
                Mreg[i] = 1.0/(np.linalg.norm(X0[:,i],normalize))
                X[:,i] = Mreg[i]*X0[:,i]                
        else: 
            X = X0
        
        # Inherit lambda
        #w = self.lambda_1/Mreg  
        w = w_best #np.linalg.lstsq(X,y)[0]          
        
        biginds = np.where(abs(w) > tol)[0]
        num_relevant = d            
        
        # Threshold and continue
        for j in range(maxit):
    
            # Figure out which items to cut out
            smallinds = np.where(abs(w) < tol)[0]
            new_biginds = [i for i in range(d) if i not in smallinds]
                
            # If nothing changes then stop
            if num_relevant == len(new_biginds): break
            else: num_relevant = len(new_biginds)
                
            if len(new_biginds) == 0:
                if j == 0: 
                    if normalize != 0:
                        return np.multiply(Mreg, w)
                    else:
                        return w
                else: break
            biginds = new_biginds
            
            # Otherwise get a new guess
            w[smallinds] = 0
            
            if lam != 0: w[biginds] = np.linalg.lstsq(X[:, biginds].T.dot(X[:, biginds]) + lam*np.eye(len(biginds)),X[:, biginds].T.dot(y))[0]
            else: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
            
        #print(j)
    
        # Now that we have the sparsity pattern, use standard least squares to get w
        if len(biginds) > 0:
            w[biginds] = np.linalg.lstsq(X[:, biginds], y, rcond=None)[0]

#         if biginds != []: w[biginds] = np.linalg.lstsq(X[:, biginds],y)[0]
        
        if normalize != 0:
            return np.multiply(Mreg, w)
        else:
            return w
        
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        #x = torch.cat([x,t],dim=1)

        self.dnn.eval()
        u= self.net_u(x,t) #ped solution
        #f = self.net_f(x,t) #pde residual
        u = u.detach().cpu().numpy()
        #f = f.detach().cpu().numpy()
        return u


def sample(domain, num):
    min = domain[0]
    max = domain[1]
    x = min + (max-min) * np.random.uniform(0,1,size=num)
    return x

def figure_compare(name='1'):        
    t_lower = t[0]
    x_lower = x[0]
    t_upper = t[-1]
    x_upper = x[-1]
    B_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) 
    prediction = model.predict(B_star)
    #arr1 = np.array(prediction[0],dtype=float)
    u = prediction.reshape(X.shape[0], X.shape[1])
    # u = np.squeeze(u)
    
    h =  Exact
    
    vnorm = mpl.colors.Normalize(vmin=np.min(h), vmax=np.max(h))
    fig, ax = plt.subplots(2)
    
    ax[0].set_title("Results")
    ax[0].set_ylabel("model prediction")
    im1 = ax[0].imshow(
        u.T,
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
        norm = vnorm,
    )
    fig.colorbar(im1, ax=ax[0])
    ax[1].set_ylabel("real")
    im2 = ax[1].imshow(
        h.T,
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
        norm = vnorm,
    )
    fig.colorbar(im2, ax=ax[1])
    output_file = os.path.join(output_dir, f'{name}_whole.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    u2 = u[6,:]
    h2 = h[6,:]
     
    #x_sample_i = sample(x,50)
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(x,u2 , 'r-',linewidth = 2, label='Input')
    plt.plot(x, h2 , 'k--',linewidth =2, label='ture result')
    plt.legend() 
    plt.xlabel('x ')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, f'{name}_fixed_t.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(t,u[:,128] , 'r-',linewidth = 2, label='Input')
    plt.plot(t, h[:,128] , 'k--',linewidth =2, label='ture result')
    plt.legend() 
    plt.xlabel('t')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, f'{name}_fixed_x.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        

def figure_compare2():        
    t_lower = t[0]
    x_lower = x[0]
    t_upper = t[100]
    x_upper = x[-1]
    t_2 = np.linspace(t_lower, t_upper, 200)
    X_2, T_2 = np.meshgrid(x,t_2)
    B_star = np.hstack((X_2.flatten()[:, None], T_2.flatten()[:, None]/2)) 
    prediction = model.predict(B_star)
    #arr1 = np.array(prediction[0],dtype=float)
    u = prediction.reshape(X_2.shape[0], X_2.shape[1])
    # u = np.squeeze(u)
    
    h =  Exact[0:100,:]
    
    vnorm = mpl.colors.Normalize(vmin=np.min(h), vmax=np.max(h))
    fig, ax = plt.subplots(2)
    
    ax[0].set_title("Results")
    ax[0].set_ylabel("model prediction")
    im1 = ax[0].imshow(
        u.T,
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
        norm = vnorm,
    )
    fig.colorbar(im1, ax=ax[0])
    ax[1].set_ylabel("real")
    im2 = ax[1].imshow(
        h.T,
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
        norm = vnorm,
    )
    fig.colorbar(im2, ax=ax[1])
    output_file = os.path.join(output_dir, 'whole2.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    u2 = u[50,:]
    h2 = h[25,:]
     
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(x,u2 , 'r-',linewidth = 2, label='Input')
    plt.plot(x, h2 , 'k--',linewidth =2, label='ture result')
    plt.legend() 
     
    plt.xlabel('x ')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, 'fixed_t2.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(t_2,u[:,128] , 'r-',linewidth = 2, label='Input')
    plt.plot(t[0:100], h[:,128] , 'k--',linewidth =2, label='ture result')
    plt.legend() 
    plt.xlabel('t')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, 'fixed_x2.png')
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        

print("load module")

# -----------------------------
# Load data and define analytical solution
# -----------------------------
x_lower = 0
x_upper = np.pi
t_lower = 0
t_upper = 4
a=1
l=0.5
w_1= np.sqrt(a**2-l**2/4)
w_2= np.sqrt(4*a**2-l**2/4)

def func2(x,t):
    var1 = np.sin(x)*(np.exp(-l*t/2)*np.cos(w_1*t))
    var1 = var1 +0.5*np.sin(2*x)*(np.exp(-l*t/2)*np.cos(w_2*t))  #+2*w_1*np.exp(-l*t/2)*np.sin(w_1*t)/l
    return var1
x = np.linspace(x_lower, x_upper, 256)
t = np.linspace(t_lower, t_upper, 200)
X, T = np.meshgrid(x,t)
Exact = func2(X,T)

a0 = 0
length = 200
t = t[a0:a0+length]
x = x
t1 = [t[0],t[-1]]
t2 = sample(t1, length*4)
x1 = [x[0],x[-1]]
x2 = sample(x1,50)
X2, T2 = np.meshgrid(x2,t2)
Exact0 = Exact[a0:a0+length,:]
X, T = np.meshgrid(x,t)

# -----------------------------
# Create dataset for PINN training
# -----------------------------
noise = 0
#Exact = np.round(Exact,1)
Exact = Exact0 + noise*np.std(Exact0)*np.random.randn(Exact0.shape[0], Exact0.shape[1])

measurement = np.hstack((X.flatten()[:,None], T.flatten()[:,None],Exact.flatten()[:,None],Exact0.flatten()[:,None] )) 

Nm = 1000
m_sample = measurement[np.random.choice(measurement.shape[0],Nm),:]
f = np.hstack((X2.flatten()[:, None], T2.flatten()[:, None]))
Nf = 2*10**4
f_sample = f[np.random.choice(f.shape[0],Nf),:]

print(f"load experience data Nm = {Nm} Nf = {Nf} noise = {noise} start_point= {a0}")

# -----------------------------
# Training Process
# -----------------------------

lambda_1_history = []
layers = [2,50,100,100,100,1]
model = PINNInference(x_max=1,t_max=1,measurement=m_sample,f=f_sample ,Nf=Nf, layers=layers, device=device, a=1, b=1, c=1, d=1)
model.train_inverse(niter=5001,Ir=0.003,lambda1=1,lambda2=0.1,lambda3=0,lambda4=1,plot_num=1000)
# figure_compare(name='pertraining1')
model.train_inverse(niter=5001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1e-4,plot_num=1000)
model.callTrainSTRidge(lam = 1e-5,d_tol =1,maxit = 100,STR_iters = 10,normalize = 2)
#print_pde(model.lambda_1, rhs_des)
# figure_compare(name='pertraining4')
#model.train_inverse(niter=5001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1e-4,plot_num=1000,freezing=True)
model.train_inverse(niter=5001, Ir= 0.001, lambda1=1, lambda2=1, lambda3=0, lambda4=1.0, plot_num=1000,freezing=True)

# -----------------------------
# Plot Results
# -----------------------------
print_pde(model.lambda_1, rhs_des)
lambda_1_history_np = np.array(lambda_list)

plt.figure(figsize=(10, 6))

for i in range(lambda_1_history_np.shape[1]): 
    if i == 8:  
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='-', color='red')  
        plt.axhline(y=1, color='red', linestyle=':')
    elif i == 12:  
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='-', color='blue')  
        plt.axhline(y=-0.5, color='blue', linestyle=':')
    else: 
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='--')  


plt.xlabel('Iteration')
plt.ylabel('Value of Variables')
plt.title('u_tt=u_xx-0.1u_t')
plt.legend()
plt.grid(True)
output_file = os.path.join(output_dir, 'iter.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight') 
plt.show()
plt.close()        


figure_compare()

data_dict = {
    "lambda_list": lambda_list,
    "lambda_list2": lambda_list2,
    "loss_f_list": loss_f_list,
    "loss_l_list": loss_l_list,
    "loss_m_list": loss_m_list,
    "loss_ut_list": loss_ut_list,
    "term_list": term_list,
    "diff_list": diff_list,
    "loss_record_list": loss_record_list
}

torch.save(data_dict, os.path.join(output_dir, "training_data.pth"))

torch.save(model.dnn.state_dict(), os.path.join(output_dir, "net_state.pth"))

loss_f_history_np = np.array(loss_f_list)
fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(np.log(loss_f_history_np), 'b-', label='log(loss_f)')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('log(loss_f)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yscale('linear')  #

output_file = os.path.join(output_dir, 'Loss_f over Iterations.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight') 
plt.show()
plt.close()