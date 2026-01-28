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

        f = v_t- w_x
        return f,w_x,v_t
    
    def net_ut(self, x ,t):
        """ 计算v
        """
        # self.lambda_1 = torch.flatten(self.dnn.branch_conv[0].weight).unsqueeze(1) #self.dnn.layers.hidden_layer_1.bias.unsqueeze(1)
        #lambda_1 = self.lambda_1 ** 2    
        u = self.net_u(x,t)
        v = 1*torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]#.unsqueeze(-1)
        return v
    
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
            # loss_l = torch.sum(torch.abs(self.lambda_1))
            v_2 = self.net_ut(self.init_x, self.init_t)
            
            # loss_f = torch.mean((1)*f_pred ** 2)
            loss_ut = torch.mean(v_2 **2)
            loss =  lambda1*loss_m + lambda2*loss_f + lambda4*loss_ut
            
            self.optimizer.zero_grad()
            loss.backward()  
                
            self.optimizer.step()
            if freezing:
                with torch.no_grad():
                    self.lambda_1[mask] = 0

            loss_list.append(loss.item())
            
            # lambda_sum = torch.sum(torch.abs(self.lambda_1))
#             # 将 para 转换为 NumPy 数组（或如果希望存储 Tensor，可以省略 .numpy()）
#             para = self.lambda_1.clone()
#             para_numpy = para.detach().cpu().numpy()
# #             diff = torch.mean((m_pred - self.real)**2)
#             # 将 para 存入列表
#             lambda_list.append(para_numpy)
            loss_f_list.append(loss_f.item())   
            # loss_l_list.append(loss_l.item())  
            loss_m_list.append(loss_m.item()) 
            loss_ut_list.append(loss_ut.item())  
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
                    'Iter %d, Loss: %.5e, Loss_m: %.5e, Loss_f: %.5e, Loss_ut: %.5e, time: %.2f' % ( \
                    it, loss.item(), loss_m.item(), loss_f.item(), loss_ut.item(),elapsed)
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

x = np.linspace(x_lower, x_upper, 1000)
t = np.linspace(t_lower, t_upper, 1000)
X, T = np.meshgrid(x,t)
Exact = func2(X,T)

# -----------------------------
# Create dataset for PINN training
# -----------------------------
# noise = 0
# #Exact = np.round(Exact,1)
# Exact = Exact0 + noise*np.std(Exact0)*np.random.randn(Exact0.shape[0], Exact0.shape[1])

cols = [0, -1]

# 取空间两端
X_sides = X[:, cols].reshape(-1, 1)
T_sides = T[:, cols].reshape(-1, 1)
Exact_sides = Exact[:, cols].reshape(-1, 1)

# 取初始时刻
X_init = X[[0], :].reshape(-1, 1)
T_init = T[[0], :].reshape(-1, 1)
Exact_init = Exact[[0], :].reshape(-1, 1)

# 合并三部分
measurement = np.vstack((
    np.hstack((X_sides, T_sides, Exact_sides)),
    np.hstack((X_init, T_init, Exact_init))
))

# 若想去除重复行（例如角点重复）
measurement = np.unique(measurement, axis=0)

init = np.hstack((
    X[0, :].reshape(-1, 1),
    T[0, :].reshape(-1, 1)
))

x = np.linspace(x_lower, x_upper, 256)
t = np.linspace(t_lower, t_upper, 200)
X, T = np.meshgrid(x,t)
Exact = func2(X,T)

f = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
Nf = 10**4
f_sample = f[np.random.choice(f.shape[0],Nf),:]

print(f"load experience data  Nf = {Nf} ")

# -----------------------------
# Training Process
# -----------------------------

lambda_1_history = []
layers = [2,50,100,100,100,1]
model = PINNInference(x_max=1,t_max=1,measurement=measurement,init_xt=init,f=f_sample ,Nf=Nf, layers=layers, device=device, a=1, b=1, c=1, d=1)
model.train_inverse(niter=5001,Ir=0.003,lambda1=1,lambda2=0.1,lambda3=0,lambda4=1,plot_num=1000)
figure_compare(name='pertraining1')
model.train_inverse(niter=5001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1e-4,plot_num=1000)
#print_pde(model.lambda_1, rhs_des)
figure_compare(name='pertraining4')
#model.train_inverse(niter=5001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1e-4,plot_num=1000,freezing=True)
model.train_inverse(niter=5001, Ir= 0.001, lambda1=1, lambda2=1, lambda3=0, lambda4=1.0, plot_num=1000,freezing=True)

# -----------------------------
# Plot Results
# -----------------------------

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
loss_m_history_np = np.array(loss_m_list)
loss_ut_history_np = np.array(loss_ut_list)

fig, ax1 = plt.subplots(figsize=(10, 5))

ax1.plot(np.log(loss_f_history_np), 'b-', label='log(loss_f)')
ax1.plot(np.log(loss_m_history_np), 'r-', label='log(loss_m)')
ax1.plot(np.log(loss_ut_history_np), 'g-', label='log(loss_ut)')
ax1.set_ylabel('log(loss_f)', color='b')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('log(loss_f)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yscale('linear')  #

output_file = os.path.join(output_dir, 'Loss_f over Iterations.png')
plt.savefig(output_file, dpi=300, bbox_inches='tight') 
plt.show()
plt.close()