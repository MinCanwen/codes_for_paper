import os
pid = os.getpid()
print(f"Current process ID: {pid}")
# 使用字符串格式化来构造文件夹路径
# 获取 Python 文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# # 构建输出文件夹路径
noise=1.0
output_dir = os.path.join(current_dir, f"10G_{pid}")
# output_dir = os.path.join(f"./0_{pid}")
print(output_dir)
# 如果目录不存在，则创建
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
rhs_des = ['','u', 'u_x', 'u_xx/2(-1)', 'x^2', 'x^2*u(0.5)', 'x^2*u_x','x^2*u_xx',
                           'x','iu', 'iu_x', 'iu_xx', 'x^3', 'x^2*iu', 'x^2*iu_x','x^2*iu_xx']
def print_pde(w, rhs_description, ut = 'iu_t'):
    if isinstance(w, np.ndarray):
        lambda_1_value = w
    else:
        lambda_1_value = w.detach().cpu().numpy()  # 获取当前 lambda_1 并转为 numpy 数组
    #lambda_1_history.append(lambda_1_value)
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
from torch.optim.lr_scheduler import ExponentialLR
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.interpolate import griddata
import scipy.io

from typing import OrderedDict

if torch.backends.mps.is_available():
    device = torch.device('mps')  # macOS 上的 Apple Silicon (M1/M2) 使用 MPS
elif torch.cuda.is_available():
    device = torch.device('cuda:0')  # Nvidia GPU 使用 CUDA
else:
    device = torch.device('cpu')  # 默认使用 CPU

print(f"Using device: {device}")

lambda_list = []
lambda_list2 = []
loss_f_list=[]
loss_m_list=[]
loss_ut_list=[]
loss_l_list=[]
term_list = []
loss_record_list = []

class Net(nn.Module):
    def makedense(self,myinput,myoutput):#用函数创建重复的层
        dense=torch.nn.Sequential(torch.nn.Linear(myinput,myoutput),)
        return dense
    
    def __init__(self, layers):
        """
        Args:
            layers: 数组, 存放神经网络每一层的节点数
        """
        super(Net, self).__init__()
        layers_seq = list()
      
        self.layers_cnt = len(layers) - 1
      
        self.tanh = nn.Tanh()  # Tanh激活函数
      
                # 构建主干网络
        for i in range(0,len(layers) - 2):
            # if i == 2:
            #     layers_seq.append(('hidden_layer_%d' % i, nn.Linear(layers[i]+121, layers[i+1])))
            # else:
            layers_seq.append(('hidden_layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            layers_seq.append(('actvation_%d' % i, nn.Tanh()))
        layers_seq.append(('output_layer', nn.Linear(layers[-2], layers[-1])))

        self.layers = nn.Sequential(OrderedDict(layers_seq))
        self.prev_mask = torch.zeros(16, 1) <= 1
        self.mask = torch.zeros(16, 1) <= 1
        self.loss_f = 1
        self.thresholding=True
        print(self.prev_mask)

    def forward(self, x):
        output = self.layers(x)
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
        
        Args:
            measurement: 训练集中的观察点(x, t, u)
            f: 训练集中内部点的(x, t, u), 用于计算pde residual
            t_max,x_max: 时间最大值,用于归一化
            layers: 网络结构
            device: 设备
        """
        self.device=device        
        
        # 预先sample，直接读入device，在迭代中不变
        if measurement is not None:
            self.m_x = torch.tensor(measurement[:, 0:1], requires_grad=True).float().to(self.device)
            self.m_t = torch.tensor(measurement[:, 1:2], requires_grad=True).float().to(self.device)  
            self.m_u = torch.tensor(measurement[:, 2:3]).float().to(device)
            self.m_i = torch.tensor(measurement[:, 3:4]).float().to(device)

        # f data
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
        self.optimizer = None
        self.scheduler = None
        # settings lambda初值
        self.lambda_1 = torch.zeros(16, 1, requires_grad=True).to(device)
        #self.lambda_1 = torch.tensor([10.0], requires_grad=True).to(device)
        self.phi_norm = torch.ones(16, 1, requires_grad=False).to(device)
        #self.lambda_1 = torch.tensor([10.0], requires_grad=True).to(device)
        
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        self.loss_f_record = 1
        self.loss_record = 1
        
        # deep neural networks
        self.dnn = Net(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)       
        
        self.iter = 0
        

    def net_u(self, x, t):  
        x1 = 2*(x-self.a)/self.b -1
        t1 = 2*(t-self.c)/self.d -1
        uu = self.dnn(torch.cat([x1, t1], dim=1))#torch.cat拼接x,t矩阵，竖向
#         a = self.dnn(torch.cat([x, t], dim=1))#torch.cat拼接x,t矩阵，竖向
        u = uu[:,0:1]
        w = uu[:,1:2]
        v = uu[:,2:3]
        w_x = uu[:,3:4]        
        iu = uu[:,4:5]
        iw = uu[:,5:6]
        iv = uu[:,6:7]
        iw_x = uu[:,7:8]
        #print(u.shape)
        return u,w,v,w_x,iu,iw,iv,iw_x
    
    def net_f(self, x ,t):
        """ 计算pde residual
        """
        # self.lambda_1 = torch.flatten(self.dnn.branch_conv[0].weight).unsqueeze(1) #self.dnn.layers.hidden_layer_1.bias.unsqueeze(1)
        #lambda_1 = self.lambda_1 ** 2    
        u,w,v,w_x,iu,iw,iv,iw_x = self.net_u(x,t)
        Phi1 = torch.cat([torch.ones_like(u), u,w,w_x/2, 
                         x**2,x**2*u,x**2*w,x**2*w_x,
                         x, iu,iw,iw_x, 
                         x**3,x**2*iu,x**2*iw,x**2*iw_x,], 1) 
        Phi2 = torch.cat([torch.ones_like(u), iu,iw,iw_x/2, 
                         x**2,x**2*iu,x**2*iw,x**2*iw_x,
                         x,u,w,w_x, 
                         x**3,x**2*u,x**2*w,x**2*w_x,], 1) 
        u_tt1 = -iv
        f1 = u_tt1- torch.matmul(Phi1, self.lambda_1)
        u_tt2 = v
        f2 = u_tt2- torch.matmul(Phi2, self.lambda_1) #
        return f1**2+f2**2 ,Phi1,u_tt1

    def net_ut(self, x ,t,dt,dx):
        """ 计算pde residual
        """
        #lambda_1 = self.lambda_1 ** 2    
        u,w,v,w_x,iu,iw,iv,iw_x = self.net_u(x,t)
        u_t1,w_t1,v_t1,w_x_t1,iu_t1,iw_t1,iv_t1,iw_x_t1 = self.net_u(x,t-dt)
        u_t2,w_t2,v_t2,w_x_t2,iu_t2,iw_t2,iv_t2,iw_x_t2 = self.net_u(x,t+dt)
        u_x1,w_x1,v_x1,w_x_x1,iu_x1,iw_x1,iv_x1,iw_x_x1 = self.net_u(x-dx,t)
        u_x2,w_x2,v_x2,w_x_x2,iu_x2,iw_x2,iv_x2,iw_x_x2 = self.net_u(x+dx,t)
        f = (v-(u_t2-u_t1)/(dt*2))** 2 +(w- (u_x2-u_x1)/(dx*2))** 2 +((u_x2+u_x1-2*u)/(dx**2)-w_x)**2 \
        + (iv-(iu_t2-iu_t1)/(dt*2))** 2 +(iw- (iu_x2-iu_x1)/(dx*2))** 2 +((iu_x2+iu_x1-2*iu)/(dx**2)-iw_x)**2
        return f
    
    def train(self, nIter,lam=None,Ir=0.001,lambda1=1.0, lambda2=1.0): # nIter is the number of ADO loop
            
        for it in range(nIter):
            # Loop of Adam optimization            
            print('Adam begins')             
            self.train_inverse(niter=2000,Ir=Ir, plotting=True,lambda1=lambda1, lambda2=lambda2,plot_num=200)
            # Loop of STRidge optimization
            print(self.dnn.lambda_1)
            self.callTrainSTRidge()
            print(self.dnn.lambda_1)
    
    def train_inverse(self, niter=0,Ir=0.001, plotting=False, lambda1=1.0, lambda2=1.0,lambda3=0.00001,lambda4=1.0,plot_num=40,freezing=False,dt=0.1,dx=0.1):
        """ mse = lambda1 * mse_u + lambda2 * mse_f
        """
        if self.optimizer is None:  # 仅首次初始化优化器
            #self.optimizer = torch.optim.SGD(self.dnn.parameters(), lr=Ir, momentum=0.9, nesterov=True)
            self.optimizer = torch.optim.Adam(self.dnn.parameters(), lr=Ir)
            self.scheduler = ExponentialLR(self.optimizer, gamma=1)
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = Ir # 例如将学习率减少10倍
        self.dnn.train()
        start_time = time.time()
        
        loss_list = []
        
        t_list = range(niter)
        f_pred,phi,u_tt = self.net_f(self.f_x, self.f_t)
        u_tt_norm = torch.mean(u_tt**2).detach()
        phi_norm = torch.mean(phi**2, dim=0).detach()
        self.loss_f_record = 1 
        self.loss_record = 1
        mask = (model.lambda_1 == 0)        # 获取当前 mask（lambda == 0 的位置）

        for it in range(niter):
            f_pred,phi,u_tt = self.net_f(self.f_x, self.f_t) #function
            m_pred,_,_,_,i_pred,_,_,_ = self.net_u(self.m_x, self.m_t) #measurement
            ut_pred = self.net_ut(self.m_x, self.m_t,dt,dx) #function
            
            loss_m = torch.mean((1+torch.abs(self.m_u))*(m_pred - self.m_u) ** 2+(1+torch.abs(self.m_i))*(i_pred - self.m_i) ** 2)
            loss_f = torch.mean((1)*f_pred)
            loss_ut = torch.mean((1)*ut_pred)
            loss_l = torch.sum(torch.abs(self.lambda_1))
            loss =  lambda1*loss_m + lambda2*loss_f + lambda3*loss_l + lambda4*loss_ut
            if loss_f.item() < self.loss_f_record:
                self.loss_f_record = loss_f.item()
            if loss.item() < self.loss_record:
                self.loss_record = loss.item()
            self.optimizer.zero_grad()
            loss.backward()

            if freezing:
                # 获取层的权重并计算掩码
                layer_1_weight = self.lambda_1
                layer_1_weight.grad[mask] = 0  # 清除这些部分的梯度

            self.optimizer.step()
            
            if freezing:
                with torch.no_grad():
                    self.lambda_1[mask] = 0
            loss_list.append(loss.item())
            
            lambda_sum = torch.sum(torch.abs(self.lambda_1))
            # 将 para 转换为 NumPy 数组（或如果希望存储 Tensor，可以省略 .numpy()）
            para = self.lambda_1.clone()
            para_numpy = para.detach().cpu().numpy()
        
            # 将 para 存入列表
            lambda_list.append(para_numpy)
            loss_f_list.append(loss_f.item())   
            loss_l_list.append(loss_l.item())  
            loss_m_list.append(loss_m.item()) 
            loss_ut_list.append(loss_ut.item())      
            # phi_norm = torch.mean(phi**2, dim=0).detach()
            para2 = para.T*(phi_norm)/u_tt_norm
            para_numpy2 = para2.detach().cpu().numpy()
            # 将 para 存入列表
            lambda_list2.append(para_numpy2)

            if it % plot_num == 0:
                elapsed = time.time() - start_time
                print(
                    'Iter %d, Loss: %.5e, Loss_m: %.5e, Loss_f: %.5e,Loss_ut: %.5e,Loss_l: %.3e,lam:%.5e,u_tt_norm:%.3e, time: %.2f' % ( \
                     it, loss.item(), loss_m.item(), loss_f.item(),loss_ut.item(),loss_l.item(),lambda_sum.item(),u_tt_norm.item(),elapsed)
                )
                self.scheduler.step()
                start_time = time.time()
                u_tt_norm = torch.mean(u_tt**2).detach()
                phi_norm = torch.mean(phi**2, dim=0).detach()
                print('log_loss_f_min',np.log(self.loss_f_record),'log_loss_min',np.log(self.loss_record))
        
        loss_record_list.append(model.loss_f_record)
        # 画loss关于iter的图
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
        
        normalize = normalize#原本是2   
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
        u,_,_,_,iu,_,_,_ = self.net_u(x,t) #ped solution
        #f = self.net_f(x,t) #pde residual
        u = u.detach().cpu().numpy()
        iu = iu.detach().cpu().numpy()
        #f = f.detach().cpu().numpy()
        return u,iu


def sample(domain, num):
    min = domain[0]
    max = domain[1]
    x = min + (max-min) * np.random.uniform(0,1,size=num)
    return x

def figure_compare(name='1'):        
    ## 做预测
    t_lower = t[0]
    x_lower = x[0]
    t_upper = t[-1]
    x_upper = x[-1]
    B_star = np.hstack((X.flatten()[:, None], T.flatten()[:, None])) 
    prediction = model.predict(B_star)
    #arr1 = np.array(prediction[0],dtype=float)
    u = prediction[0].reshape(X.shape[0], X.shape[1])
    iu = prediction[1].reshape(X.shape[0], X.shape[1])
    # u = np.squeeze(u)
    
    h =  Exact_real
    ih =  Exact_imag
    vnorm = mpl.colors.Normalize(vmin=np.min(h), vmax=np.max(h))
    #scipy.io.savemat('diffusion.mat',{'u':model_prediction,'turth':h,'X':B,'T':C,'init_sample':init_xt,'bound_l':left_bound_xt,'bound_r':right_bound_xt,'f_sample':f_xt})
    # 绘制预测
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
    
    fig, ax = plt.subplots(2)
    ax[0].set_title("Results")
    ax[0].set_ylabel("model prediction")
    im1 = ax[0].imshow(
        iu.T,
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
        ih.T,
        interpolation="nearest",
        cmap="viridis",
        extent=[t_lower, t_upper, x_lower, x_upper],
        origin="lower",
        aspect="auto",
        norm = vnorm,
    )
    fig.colorbar(im2, ax=ax[1])
    output_file = os.path.join(output_dir, f'{name}_whole_imag.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    u2 = u[6,:]
    h2 = h[6,:]    
    iu2 = iu[6,:]
    ih2 = ih[6,:]
     
    #开始画图
    #x_sample_i = sample(x,50)
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(x,u2 , 'r-',linewidth = 2, label='Input')
    plt.plot(x, h2 , 'k--',linewidth =2, label='ture result')
    plt.plot(x,iu2 , 'r-',linewidth = 2, label='Input_i')
    plt.plot(x, ih2 , 'k--',linewidth =2, label='ture result_i')
    plt.legend() # 显示图例
     
    plt.xlabel('x ')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, f'{name}_fixed_t.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(t,u[:,128] , 'r-',linewidth = 2, label='Input')
    plt.plot(t, h[:,128] , 'k--',linewidth =2, label='ture result')
    plt.plot(t,iu[:,128] , 'r-',linewidth = 2, label='Input_i')
    plt.plot(t, ih[:,128] , 'k--',linewidth =2, label='ture result_i')
    plt.legend() # 显示图例
    plt.xlabel('t')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, f'{name}_fixed_x.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        

print("load module")


# In[load data] 
data = sio.loadmat('/data/home/liuyulong/Datasets/harmonic_osc.mat')
t = data['t'].flatten()[:,None]
x = data['x'].flatten()[:,None]
Exact_real = np.real(data['usol'])
Exact_imag = np.imag(data['usol'])

a0 = 0
length = t.shape[0]
t = t[a0:a0+length]
x = x
t1 = [t[0],t[-1]]
t2 = sample(t1, length*4)
x1 = [x[0],x[-1]]
x2 = sample(x1,50)
X, T = np.meshgrid(x,t)
X2, T2 = np.meshgrid(x2,t2)
Exact_real = Exact_real[a0:a0+length,:]
Exact_imag = Exact_imag[a0:a0+length,:]

# In[add Gaussian noise]
Exact_real = Exact_real + noise*np.std(Exact_real)*np.random.randn(Exact_real.shape[0], Exact_real.shape[1])
Exact_imag = Exact_imag + noise*np.std(Exact_imag)*np.random.randn(Exact_imag.shape[0], Exact_imag.shape[1])
###########sample by raws
num_cols = 100
cols = np.random.choice(Exact_imag.shape[1], num_cols, replace=False)
new_Exact_real = Exact_real[:, cols]
new_Exact_imag = Exact_imag[:, cols]
new_X = X[:, cols]
new_T = T[:, cols]

num_raws = 10
raws = np.random.choice(Exact_real.shape[0], num_cols, replace=False)
new_Exact_real = new_Exact_real [raws,:]
new_Exact_imag = new_Exact_imag [raws,:]
new_X = new_X [raws,:]
new_T = new_T [raws,:]
Nm = num_raws *num_cols

m_sample = np.hstack((new_X .flatten()[:,None], new_T .flatten()[:,None],new_Exact_real.flatten()[:,None],new_Exact_imag.flatten()[:,None]  )) 

# In[create datasets for neural network]
#measurement = np.hstack((X.flatten()[:,None], T.flatten()[:,None],Exact_real.flatten()[:,None] ,Exact_imag.flatten()[:,None])) 

#Nm = 1000
#m_sample = measurement[np.random.choice(measurement.shape[0],Nm),:]
f = np.hstack((X2.flatten()[:, None], T2.flatten()[:, None]))
Nf = 30000
f_sample = f[np.random.choice(f.shape[0],Nf),:]

print(f"load experience data Nm = {Nm} Nf = {Nf} length = {length},noise= {noise}")

a = float(x[0])
b = float(x[-1] - x[0])
c = float(t[0])
d = float(t[-1] - t[0])

# In[Pretaining]
lambda_1_history = []
#layers = [2,20,20,20,20,20,20,20,20,8]
layers = [2,50,50,50,50,50,50,50,50,8]
model = PINNInference(x_max=1,t_max=1,measurement=m_sample,f=f_sample ,Nf=Nf, layers=layers, device=device, a=a, b=b, c=c, d=d)
model.train_inverse(niter=10001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1e-8,lambda4=1,plot_num=1000)
figure_compare(name='pertraining1')
model.train_inverse(niter=30001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1e-8,plot_num=1000)
print_pde(model.lambda_1, rhs_des)
figure_compare(name='pertraining')
model.callTrainSTRidge(lam = 1e-5,d_tol =15,maxit = 100,STR_iters = 50,normalize = 2)
model.train_inverse(niter=30001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1e-8,plot_num=1000)
print_pde(model.lambda_1, rhs_des)
figure_compare(name='pertraining')
#model.callTrainSTRidge(lam = 1e-5,d_tol =15,maxit = 100,STR_iters = 50,normalize = 2)
model.optimizer = None
# In[adjust final results without l0 penalty]
model.train_inverse(niter=30001, Ir= 0.001, lambda1=1, lambda2=1,lambda3=0, lambda4=1.0, plot_num=1000,freezing=True,dt=0.01,dx=0.01)
print_pde(model.lambda_1, rhs_des)


# In[plot results]
# 转换为 NumPy 数组，便于后续操作
lambda_1_history_np = np.array(lambda_list)

# 绘制每个变量随迭代次数的变化曲线
plt.figure(figsize=(10, 6))

# 遍历每个变量
for i in range(lambda_1_history_np.shape[1]):  # 遍历每列
    if i == 3:  # 第2个变量（索引从0开始）
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='-', color='red')  # 红色虚线
        plt.axhline(y=-0.5, color='red', linestyle=':')
    elif i == 5:  # 第9个变量
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='-', color='blue')  # 蓝色点线
        plt.axhline(y=0.5, color='blue', linestyle=':')
    else:  # 其他变量
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='--')  # 黑色实线

# 添加标签和标题
plt.xlabel('Iteration')
plt.ylabel('Value of Variables')
plt.title('iu_t=-0.5u_xx+0.5x^2u')
plt.legend()
plt.grid(True)
output_file = os.path.join(output_dir, 'iter.png')
# 保存图片为高分辨率 PNG
plt.savefig(output_file, dpi=300, bbox_inches='tight') 
plt.show()
plt.close()        

# 转换为 NumPy 数组，便于后续操作
lambda_1_history_np = np.array(lambda_list2)[20000:,:]
lambda_1_history_np = np.squeeze(lambda_1_history_np, axis=1)
# 绘制每个变量随迭代次数的变化曲线
plt.figure(figsize=(10, 6))

# 遍历每个变量
for i in range(lambda_1_history_np.shape[1]):  # 遍历每列
    if i == 3:  # 第2个变量（索引从0开始）
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='-', color='red')  # 红色虚线
    elif i == 5:  # 第9个变量
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='-', color='blue')  # 蓝色点线
    else:  # 其他变量
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='--')  # 黑色实线

# 添加标签和标题
plt.xlabel('Iteration')
plt.ylabel('Value of Variables')
plt.title('Change of scaled Variables over Iterations(wave)')
plt.legend()
plt.grid(True)
output_file = os.path.join(output_dir, 'iter_scaled.png')
# 保存图片为高分辨率 PNG
plt.savefig(output_file, dpi=300, bbox_inches='tight') 
plt.show()
plt.close()        

figure_compare()
#figure_compare2()

# 组织数据
data_dict = {
    "lambda_list": lambda_list,
    "lambda_list2": lambda_list2,
    "loss_f_list": loss_f_list,
    "loss_l_list": loss_l_list,
    "loss_m_list": loss_m_list,
    "loss_ut_list": loss_ut_list,
    "term_list": term_list,
    "loss_record_list": loss_record_list
}

# 保存数据
torch.save(data_dict, os.path.join(output_dir, "training_data.pth"))

torch.save(model.dnn.state_dict(), os.path.join(output_dir, "net_state.pth"))

# 创建图表
loss_f_history_np = np.array(loss_f_list)
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左轴绘制 loss_f (log-scale) 折线图
ax1.plot(np.log(loss_f_history_np), 'b-', label='log(loss_f)')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('log(loss_f)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yscale('linear')  # 这里用线性，因为 loss_f 已经是 log10 了

output_file = os.path.join(output_dir, 'Loss_f over Iterations.png')
# 保存图片为高分辨率 PNG
plt.savefig(output_file, dpi=300, bbox_inches='tight') 
plt.show()
plt.close()