import os
pid = os.getpid()
print(f"Current process ID: {pid}")
# 使用字符串格式化来构造文件夹路径
# 获取 Python 文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建输出文件夹路径
output_dir = os.path.join(current_dir, f"r_3000_{pid}")
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
import random

rhs_des = ['','u', 'u**2', 'u**3', 'u_x', 'u*u_x', 'u**2*u_x',
                           'u3*u_x', 'u_xx.','u*u_xx', 'u**2*u_xx', 'u3*u_xx', 'u_t.', 'u*u_t', 'u**2*u_t', 'u3*u_t']
def print_pde(w, rhs_description, ut = 'u_tt'):
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

lambda_list = []
lambda_list2 = []
loss_f_list=[]
loss_m_list=[]
loss_ut_list=[]
loss_l_list=[]
term_list = []
loss_record_list = []

seed = 42  # 随机种子
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)  # 如果用多卡或者GPU
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

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
        self.lin1=nn.Linear(layers[0], layers[1])#torch.sin(torch.nn.Linear(layers[0], layers[1]))
        self.lin1.weight.data.normal_(0,1)
        self.lin1.bias.data.fill_(0.)
        self.lin1.weight.requires_grad = True
        self.lin1.bias.requires_grad = True
        self.layers_cnt = len(layers) - 1
        self.branch_conv = None  # 用于保存卷积分支
        self.merge_layer = None  # 用于特征融合
        self.lin2=nn.Linear(layers[1], layers[2])
        self.tanh = nn.Tanh()  # Tanh激活函数
        self.branch_conv = nn.Sequential( # 
            nn.Conv2d(1, 1, kernel_size=4, stride=1, padding=2),  # 4x4 卷积核
            nn.Tanh(),
            nn.Flatten()
        )
                # 构建主干网络
        for i in range(1,len(layers) - 2):
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
        out = torch.sin(self.lin1(x))
        # branch_input = out  # 保存分支输入
        # x = self.tanh( self.lin2(out) ) # 第一层

        # # 分支卷积操作
        # branch_output = self.branch_conv(branch_input.unsqueeze(1).reshape(-1,1, 10, 10))  # 添加通道维度 (N, 1, 10, 10)
        # merged_output = torch.cat([x, branch_output], dim=1)
        # 特征融合
        output = self.layers(out)
        return output

    def prune_and_freeze_weights(self, n=1):
        # 获取非零值（取绝对值不为零的元素）
        a = model.lambda_1*model.phi_norm
        non_zero_lambda_1 = a[torch.abs(model.lambda_1) != 0]
    
        if self.thresholding:
            if non_zero_lambda_1.numel() > 0 and np.log(model.loss_f_record) < np.log(self.loss_f) + 0.05:
                # 判断是否剩余的非零值大于 n
                if non_zero_lambda_1.numel() > n:
                    # 获取最小的 n 个非零值的索引
                    _, indices = torch.topk(torch.abs(non_zero_lambda_1), n, largest=False)
                    min_n_non_zero_values = non_zero_lambda_1[indices]
                    print(min_n_non_zero_values)
                    self.prev_mask = torch.abs(model.lambda_1*model.phi_norm) <= 0
                    print("prev_mask",self.prev_mask)
                    # 将小于最小非零值的元素置为0
                    self.mask = torch.abs(model.lambda_1*model.phi_norm) <= torch.abs( min_n_non_zero_values[-1])  # 保证压缩最小的 n 个值
                    model.lambda_1.data[self.mask] = 0
                    print("mask",self.mask)
                    self.loss_f = model.loss_f_record
                    print(f"Pruning the smallest {n} weights for lambda_1.")
                else:
                    # 若剩余的非零值小于或等于 n，则只压缩一个最小值
                    min_non_zero_value = torch.abs(non_zero_lambda_1).min().item()
                    self.prev_mask = torch.abs(model.lambda_1) <= 0
                    self.mask = torch.abs(model.lambda_1*model.phi_norm) <= min_non_zero_value
                    model.lambda_1.data[self.mask] = 0
                    self.loss_f = model.loss_f_record
                    print("Pruning the smallest weight for lambda_1.")
                
            elif model.loss_f_record > self.loss_f:
                # 若当前残差损失变差，恢复上次置0的位置为0.5
                self.mask = self.prev_mask
                self.thresholding = False
                # model.lambda_1.data[self.prev_mask] = 0.5
                print("Restoring previous mask")
        else:
            print("mask freezing")


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
        
        # settings lambda初值
        self.lambda_1 = 0.2 * torch.randn(16, 1, requires_grad=True).to(device)
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
        x1 = 0.2*(x-self.a)/self.b
        t1 = 0.2*(t-self.c)/self.d
        uu = self.dnn(torch.cat([x1, t1], dim=1))#torch.cat拼接x,t矩阵，竖向
#         a = self.dnn(torch.cat([x, t], dim=1))#torch.cat拼接x,t矩阵，竖向
        u = uu[:,0:1]
        w = uu[:,1:2]
        v = uu[:,2:3]
        w_x = uu[:,3:4]
        v_t = uu[:,4:5]
        #print(u.shape)
        return u,w,v,w_x,v_t
    
    def net_f(self, x ,t):
        """ 计算pde residual
        """
        # self.lambda_1 = torch.flatten(self.dnn.branch_conv[0].weight).unsqueeze(1) #self.dnn.layers.hidden_layer_1.bias.unsqueeze(1)
        #lambda_1 = self.lambda_1 ** 2    
        u,w,v,w_x,v_t = self.net_u(x,t)
        Phi = torch.cat([torch.ones_like(u), u,u**2,u**3, 
                         w,u*w,u**2*w,w*u**3,
                         w_x, u*w_x, u**2*w_x,w_x*u**3,
                         v,u*v,u**2*v,v*u**3], 1) 

        u_tt = v_t
        f = u_tt- torch.matmul(Phi, self.lambda_1)
        return f,Phi,u_tt

    def net_ut(self, x ,t,dt=0.1,dx=0.1):
        """ 计算pde residual
        """
        #lambda_1 = self.lambda_1 ** 2    
        u,w,v,w_x,v_t = self.net_u(x,t)
        u_t1,w_t1,v_t1,_,_ = self.net_u(x,t-dt)
        u_t2,w_t2,v_t2,_,_  =self.net_u(x,t+dt)
        u_x1,w_x1,v_x1,_,_  = self.net_u(x-dx,t)
        u_x2,w_x2,v_x2 ,_,_ = self.net_u(x+dx,t)
        f = (v-(u_t2-u_t1)/(dt*2))** 2 +(w- (u_x2-u_x1)/(dx*2))** 2 +((u_t2+u_t1-2*u)/(dt**2)-v_t)**2+((u_x2+u_x1-2*u)/(dx**2)-w_x)**2
        return f
    
    def train_inverse(self, niter=0,Ir=0.001, plotting=False, lambda1=1.0, lambda2=1.0,lambda3=0.00001,lambda4=1.0,plot_num=40,freezing=False,dx=0.1,dt=0.1):
        """ mse = lambda1 * mse_u + lambda2 * mse_f
        """
        optim = torch.optim.Adam(self.dnn.parameters(), lr=Ir)
        self.dnn.train()
        start_time = time.time()
        
        loss_list = []
        
        t_list = range(niter)
        f_pred,phi,u_tt = self.net_f(self.f_x, self.f_t)
        u_tt_norm = torch.mean(u_tt**2).detach()
        self.loss_f_record = 1 
        self.loss_record = 1

        for it in range(niter):
            f_pred,phi,u_tt = self.net_f(self.f_x, self.f_t) #function
            m_pred,w,v,_,_ = self.net_u(self.m_x, self.m_t) #measurement
            ut_pred = self.net_ut(self.f_x, self.f_t,dt,dx) #function
            
            loss_m = torch.mean((1+torch.abs(self.m_u))*(m_pred - self.m_u) ** 2)
            loss_f = torch.mean((1)*f_pred ** 2)
            loss_ut = torch.mean((1)*ut_pred)
            para2 = self.lambda_1*(self.phi_norm)/(u_tt_norm+1)
            loss_l = torch.sum(torch.abs(para2))
            loss =  lambda1*loss_m + lambda2*loss_f + lambda3*loss_l + lambda4*loss_ut
            if loss_f.item() < self.loss_f_record:
                self.loss_f_record = loss_f.item()
            if loss.item() < self.loss_record:
                self.loss_record = loss.item()
            
            optim.zero_grad()
            loss.backward()
            
            if freezing:
                # 获取层的权重并计算掩码
                layer_1_weight = self.lambda_1
                # mask = (torch.abs(layer_1_weight) == 0)  # 找到权重为零的部分
            
                # 在反向传播之前清除这些权重的梯度
                # `mask` 会是一个与 `layer_1_weight` 同大小的布尔矩阵
                # 对应为 `True` 的部分是需要冻结（清除梯度）的位置
                layer_1_weight.grad[self.dnn.mask] = 0  # 清除这些部分的梯度
                
            optim.step()
            
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
            #para2 = para.T*(phi_norm)/u_tt_norm
            para_numpy2 = para2.detach().cpu().numpy()
            # 将 para 存入列表
            lambda_list2.append(para_numpy2)

            if it % plot_num == 0:
                elapsed = time.time() - start_time
                print(
                    'Iter %d, Loss: %.5e, Loss_m: %.5e, Loss_f: %.5e,Loss_ut: %.5e,Loss_l: %.3e,lam:%.5e,u_tt_norm:%.3e, time: %.2f' % ( \
                     it, loss.item(), loss_m.item(), loss_f.item(),loss_ut.item(),loss_l.item(),lambda_sum.item(),u_tt_norm.item(),elapsed)
                )

                start_time = time.time()
                u_tt_norm = torch.sqrt(torch.mean(u_tt**2).detach())
                self.phi_norm = torch.sqrt(torch.mean(phi**2, dim=0).detach()).view(16, 1)
                print(self.phi_norm)
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
        
    def predict(self, X):
        x = torch.tensor(X[:, 0:1], requires_grad=True).float().to(self.device)
        t = torch.tensor(X[:, 1:2], requires_grad=True).float().to(self.device)
        #x = torch.cat([x,t],dim=1)

        self.dnn.eval()
        u,w,v,_,_ = self.net_u(x,t) #ped solution
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
    ## 做预测
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
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    u2 = u[6,:]
    h2 = h[6,:]
     
    #开始画图
    #x_sample_i = sample(x,50)
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(x,u2 , 'r-',linewidth = 2, label='Input')
    plt.plot(x, h2 , 'k--',linewidth =2, label='ture result')
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
    plt.legend() # 显示图例
    plt.xlabel('t')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, f'{name}_fixed_x.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        

def figure_compare2():        
    ## 做预测
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
    output_file = os.path.join(output_dir, 'whole2.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    u2 = u[50,:]
    h2 = h[25,:]
     
    #开始画图
    #x_sample_i = sample(x,50)
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(x,u2 , 'r-',linewidth = 2, label='Input')
    plt.plot(x, h2 , 'k--',linewidth =2, label='ture result')
    plt.legend() # 显示图例
     
    plt.xlabel('x ')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, 'fixed_t2.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(t_2,u[:,128] , 'r-',linewidth = 2, label='Input')
    plt.plot(t[0:100], h[:,128] , 'k--',linewidth =2, label='ture result')
    plt.legend() # 显示图例
    plt.xlabel('t')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, 'fixed_x2.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        

print("load module")


# In[load data] 
data = scipy.io.loadmat('/data/home/liuyulong/Datasets/wave_33_1.mat')
t = 2*5*np.real(data['t'].flatten()[:,None])/10 #100 *  -85
x = np.real(data['x'].flatten()[:,None])/10 # -25
a0 = 3000
length = 100
t = t[a0:a0+length]
x = x
t1 = [t[0],t[-1]]
t2 = sample(t1, length*4)
x1 = [x[0],x[-1]]
x2 = sample(x1,50)
X2, T2 = np.meshgrid(x2,t2)
Exact = np.real(data['u3']).T
Exact = 2*np.transpose(Exact)
Exact = Exact[a0:a0+length,:]
X, T = np.meshgrid(x,t)
# 创建 2D 域（用于绘图和输入）
c=np.mean(t)
d=np.std(t)
a=np.mean(x)
b=np.std(x)

num_cols = 100
cols = np.random.choice(Exact.shape[1], num_cols, replace=False)

# 生成新的子矩阵
new_Exact = Exact[:, cols]
new_X = X[:, cols]
new_T = T[:, cols]

measurement = np.hstack((new_X.flatten()[:,None], new_T.flatten()[:,None],new_Exact.flatten()[:,None] )) 
#measurement = np.hstack((X.flatten()[:,None], T.flatten()[:,None],Exact.flatten()[:,None] )) 

Nm = 5000
m_sample = measurement[np.random.choice(measurement.shape[0],Nm),:]
f = np.hstack((X2.flatten()[:, None], T2.flatten()[:, None]))
Nf = 5000
f_sample = f[np.random.choice(f.shape[0],Nf),:]

print(f"load experience data Nm = {Nm} Nf = {Nf} length = {length} start_point= {a0}")

lambda_1_history = []
layers = [2,50,100,100,5]
model = PINNInference(x_max=1,t_max=1,measurement=m_sample,f=f_sample ,Nf=Nf, layers=layers, device=device, a=1, b=1, c=1, d=1)
model.train_inverse(niter=10001,Ir=0.001,lambda1=1,lambda2=0.1,lambda3=0,lambda4=1,plot_num=5000)
figure_compare(name='pertraining1')
model.train_inverse(niter=30001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1e-9,plot_num=5000)
print_pde(model.lambda_1, rhs_des)
figure_compare(name='pertraining')
model.train_inverse(niter=30001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1e-7,lambda4=1,plot_num=1000)
figure_compare(name='pertraining2')

for i in range(5):  # 第一次训练循环
    # 训练模型
    model.train_inverse(niter=6001, Ir= 0.001, lambda1=1, lambda2=1, lambda3=1e-5, lambda4=1,plot_num=5000,dx=0.03,dt=0.03)
    
    # 打印结果
    print_pde(model.lambda_1, rhs_des)

for i in range(2):  # 第3次训练循环
    # 剪枝和冻结权重
    model.dnn.prune_and_freeze_weights(n=4)
    
    # 训练模型2
    model.train_inverse(niter=10001, Ir= 0.001, lambda1=1, lambda2=1, lambda3=1e-5, lambda4=1.0, plot_num=5000,freezing=True,dx=0.03,dt=0.03)
    
    # 打印结果2
    print_pde(model.lambda_1, rhs_des)
 

for i in range(7):  # 第3次训练循环
    # 剪枝和冻结权重
    model.dnn.prune_and_freeze_weights(n=1)
    
    # 训练模型2
    model.train_inverse(niter=10001, Ir= 0.0005, lambda1=1, lambda2=1, lambda3=1e-5, lambda4=1.0, plot_num=1000,freezing=True,dx=0.03,dt=0.03)
    
    # 打印结果2
    print_pde(model.lambda_1, rhs_des)

figure_compare(name='end')
model.train_inverse(niter=30001, Ir= 0.0005, lambda1=1, lambda2=1, lambda3=0, lambda4=1.0, plot_num=5000,freezing=True,dx=0.03,dt=0.03)
print_pde(model.lambda_1, rhs_des)
# 转换为 NumPy 数组，便于后续操作
lambda_1_history_np = np.array(lambda_list)

# 绘制每个变量随迭代次数的变化曲线
plt.figure(figsize=(10, 6))

# 遍历每个变量
for i in range(lambda_1_history_np.shape[1]):  # 遍历每列
    if i == 8:  # 第2个变量（索引从0开始）
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='-', color='red')  # 红色虚线
        plt.axhline(y=1, color='red', linestyle=':')
    elif i == 12:  # 第9个变量
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='-', color='blue')  # 蓝色点线
        plt.axhline(y=-0.5, color='blue', linestyle=':')
    else:  # 其他变量
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='--')  # 黑色实线

# 添加标签和标题
plt.xlabel('Iteration')
plt.ylabel('Value of Variables')
plt.title('u_tt=u_xx-0.1u_t')
plt.legend()
plt.grid(True)
output_file = os.path.join(output_dir, 'iter.png')
# 保存图片为高分辨率 PNG
plt.savefig(output_file, dpi=300, bbox_inches='tight') 
plt.show()
plt.close()        

# 转换为 NumPy 数组，便于后续操作
lambda_1_history_np = np.array(lambda_list2)[20000:,:]
lambda_1_history_np = np.squeeze(lambda_1_history_np)
# 绘制每个变量随迭代次数的变化曲线
plt.figure(figsize=(10, 6))

# 遍历每个变量
for i in range(lambda_1_history_np.shape[1]):  # 遍历每列
    if i == 8:  # 第2个变量（索引从0开始）
        plt.plot(lambda_1_history_np[:, i], label=rhs_des[i], linestyle='-', color='red')  # 红色虚线
    elif i == 12:  # 第9个变量
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
loss_m_history_np = np.array(loss_m_list)
fig, ax1 = plt.subplots(figsize=(10, 5))

# 左轴绘制 loss_f (log-scale) 折线图
ax1.plot(np.log(loss_f_history_np), 'b-', label='log(loss_f)')
ax1.plot(np.log(loss_m_history_np), 'k-', label='log(loss_m)')
ax1.set_xlabel('Iterations')
ax1.set_ylabel('log(loss_f)', color='b')
ax1.tick_params(axis='y', labelcolor='b')
ax1.set_yscale('linear')  # 这里用线性，因为 loss_f 已经是 log10 了

output_file = os.path.join(output_dir, 'Loss_f over Iterations.png')
# 保存图片为高分辨率 PNG
plt.savefig(output_file, dpi=300, bbox_inches='tight') 
plt.show()
plt.close()