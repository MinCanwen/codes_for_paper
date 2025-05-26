import os
pid = os.getpid()
print(f"Current process ID: {pid}")
# 使用字符串格式化来构造文件夹路径
# 获取 Python 文件所在的目录
current_dir = os.path.dirname(os.path.abspath(__file__))
# 构建输出文件夹路径
output_dir = os.path.join(current_dir, f"stage1_{pid}")
if not os.path.exists(output_dir):
    os.makedirs(output_dir)  # 如果目录不存在，则创建

print(output_dir)

mat_file_path = './wave_initial3.mat'
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
rhs_des =['','u', 'u**2', 'x','t', 'u_x', 'x*u_x', 'x**2*u_x','t*u_x', 't**2*u_x', 'u_xx.','x*u_xx', 'x**2*u_xx', 't*u_xx', 't**2*u_xx', 'u_t.', 'x*u_t', 'x**2*u_t', 't*u_t', 't**2*u_t']
# ['','u', 'u**2', 'u**3', 'u_x', 'u*u_x', 'u**2*u_x',
#                           'u**3*u_x', 'u_xx.','u*u_xx', 'u**2*u_xx', 'u**3*u_xx', 'u_t.', 'u*u_t', 'u**2*u_t', 'u**3*u_t']
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
loss_f2_list=[]
term_list = []
loss_l_list = []

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

    def prune_and_freeze_weights(self, threshold=1e-4):       
         mask = torch.abs (model.lambda_1) < 0.001
         model.lambda_1.data[mask] = 0
         print(f"Pruning weights for lambda_1 with small values.")


class PINNInference():
    def __init__(self, 
                 x_max=1,
                 t_max=1,
                 measurement=None,
                 f=None,
                 init_xt=None,
                 init_xt2=None,
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
            self.m_x2 = torch.tensor(init_xt[:, 0:1], requires_grad=True).float().to(self.device)
            self.m_t2 = torch.tensor(init_xt[:, 1:2], requires_grad=True).float().to(self.device)
            self.m_v = torch.tensor(init_xt[:, 2:3], requires_grad=True).float().to(self.device)

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
        self.lambda_1 = 0.2 * torch.randn(20, 1, requires_grad=True).to(device)
        #self.lambda_1 = torch.tensor([10.0], requires_grad=True).to(device)
        
        self.lambda_1 = torch.nn.Parameter(self.lambda_1)
        
        # deep neural networks
        self.dnn = Net(layers).to(device)
        self.dnn.register_parameter('lambda_1', self.lambda_1)       
        
        self.iter = 0
        

    def net_u(self, x, t):  
        x1 = 0.2*(x)/self.b
        t1 = 0.2*(t)/self.d
        uu = self.dnn(torch.cat([x1, t1], dim=1))#torch.cat拼接x,t矩阵，竖向
#         a = self.dnn(torch.cat([x, t], dim=1))#torch.cat拼接x,t矩阵，竖向
        u = x*(x-self.a)*uu[:,0:1]
        w = uu[:,1:2]
        v = uu[:,2:3]
        #print(u.shape)
        return u,w,v
    
    def net_f(self, x ,t):
        """ 计算pde residual
        """
        # self.lambda_1 = torch.flatten(self.dnn.branch_conv[0].weight).unsqueeze(1) #self.dnn.layers.hidden_layer_1.bias.unsqueeze(1)
        #lambda_1 = self.lambda_1 ** 2    
        u,w1,v1 = self.net_u(x,t)
        v = 1*torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]#.unsqueeze(-1)
        w = 1*torch.autograd.grad(
            u, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]#.unsqueeze(-1)
        v_t = 1*torch.autograd.grad(
            v, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]#.unsqueeze(-1)
        w_x = 1*torch.autograd.grad(
            w, x, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]#.unsqueeze(-1)
        u_tt = v_t
        Phi = w_x
        #f = u_tt   -1.038220*w_x  #discovered equation for stage1
        f = u_tt   + 0.117071*u**3-1.835131*w_x+ 0.076053*v #discovered equation for stage2
        return f,Phi,v


    def net_ut(self, x ,t):
        """ 计算v
        """
        # self.lambda_1 = torch.flatten(self.dnn.branch_conv[0].weight).unsqueeze(1) #self.dnn.layers.hidden_layer_1.bias.unsqueeze(1)
        #lambda_1 = self.lambda_1 ** 2    
        u,w1,v1 = self.net_u(x,t)
        v = 1*torch.autograd.grad(
            u, t, 
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]#.unsqueeze(-1)
        return v
        
    def train(self, niter=0,Ir=0.001, plotting=False, lambda1=1.0, lambda2=1.0,lambda3=1.0,lambda4=1e-7,lambda5=1.0,plot_num=40,freezing=False):
        """ mse = lambda1 * mse_u + lambda2 * mse_f
        """
        # 只优化其他参数
        optim = torch.optim.Adam(filter(lambda p: p is not self.lambda_1, self.dnn.parameters()), lr=Ir)
        self.dnn.train()
        start_time = time.time()
        
        loss_list = []
        
        t_list = range(niter)
        f_pred,phi,u_tt = self.net_f(self.f_x, self.f_t)
        
        u_tt_norm = torch.mean(u_tt**2).detach()
        phi_norm = torch.mean(phi**2, dim=0).detach()

        for it in range(niter):
            # f_pred,phi,u_tt = self.net_f(self.f_x, self.f_t) #function
            m_pred,w,v = self.net_u(self.m_x, self.m_t) #measurement
            # ut_pred = self.net_ut(self.f_x, self.f_t) #function
            f_pred2,phi2,v2 = self.net_f(self.f_x, self.f_t)
            v_2 = self.net_ut(self.m_x2, self.m_t2)
            
            loss_m = torch.mean((1+torch.abs(self.m_u))*(m_pred - self.m_u) ** 2)
            # loss_f = torch.mean((1)*f_pred ** 2)
            loss_f2 = torch.mean((1)*f_pred2 ** 2)
            loss_ut = torch.mean((v_2 -self.m_v)**2)
            loss =  lambda1*loss_m  + lambda2*loss_f2 + lambda3*loss_ut #+ lambda2*loss_f
            
            optim.zero_grad()
            loss.backward()                
            optim.step()
            
            loss_list.append(loss.item())
            
            lambda_sum = torch.sum(torch.abs(self.lambda_1))
            # 将 para 转换为 NumPy 数组（或如果希望存储 Tensor，可以省略 .numpy()）
            para = self.lambda_1.clone()
            para_numpy = para.detach().cpu().numpy()
        
            # 将 para 存入列表
            lambda_list.append(para_numpy)
            # loss_f_list.append(loss_f.item())  
            loss_f2_list.append(loss_f2.item()) 
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
                    'Iter %d, Loss: %.5e, Loss_m: %.5e,Loss_f2: %.3e,loss_ut:%.5e,u_tt_norm:%.3e, time: %.2f' % ( \
                     it, loss.item(), loss_m.item(), loss_f2.item(),loss_ut.item(),u_tt_norm.item(),elapsed)
                )

                start_time = time.time()
                u_tt_norm = torch.mean(u_tt**2).detach()
                phi_norm = torch.mean(phi**2, dim=0).detach()

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
        u,w,v = self.net_u(x,t) #ped solution
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
    tb = np.linspace(t_lower, t_upper, 800)
    Xb, Tb = np.meshgrid(x,tb)
    B_star = np.hstack((Xb.flatten()[:, None], Tb.flatten()[:, None])) 
    prediction = model.predict(B_star)
    #arr1 = np.array(prediction[0],dtype=float)
    u = prediction.reshape(Xb.shape[0], Xb.shape[1])
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
    
    u2 = u[0,:]
    h2 = h[0,:]
    dtb = tb[1] - tb[0]
    u3 = (u[1,:]-u[0,:])/dtb
    h3 = Exact_v[0,:]
     
    #开始画图
    #x_sample_i = sample(x,50)
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(x,u2 , 'r-',linewidth = 2, label='Output')
    plt.plot(x, h2 , 'k--',linewidth =2, label='maesurement')
    plt.plot(x, h3 , 'k-',linewidth =2, label='maesurement v')
    plt.plot(x,u3 , 'b--',linewidth = 2, label='Output v')
    plt.legend() # 显示图例     
    plt.xlabel('x ')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, f'{name}_init.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(tb,u[:,100] , 'r-',linewidth = 2, label='Output')
    plt.plot(t, h[:,100] , 'k--',linewidth =2, label='measurement')
    plt.legend() # 显示图例
    plt.xlabel('t')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, f'{name}_fixed_x.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()    

    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(tb,u[:,400] , 'r-',linewidth = 2, label='Output')
    plt.plot(t, h[:,400] , 'k--',linewidth =2, label='Measurement')
    plt.legend() # 显示图例
    plt.xlabel('t')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, f'{name}_center.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()     


def figure_compare2(name='1'):        
    ## 做预测
    t_lower = t[length1]
    x_lower = x[0]
    t_upper = t[-1]
    x_upper = x[-1]
    t_2 = np.linspace(t_lower, t_upper, length2-length1)
    X_2, T_2 = np.meshgrid(x,t_2)
    B_star = np.hstack((X_2.flatten()[:, None], T_2.flatten()[:, None])) 
    prediction = model.predict(B_star)
    #arr1 = np.array(prediction[0],dtype=float)
    u = prediction.reshape(X_2.shape[0], X_2.shape[1])
    # u = np.squeeze(u)
    
    h =  Exact[length1:length2,:]
    
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
    output_file = os.path.join(output_dir, f'{name}_whole2.png')
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
    output_file = os.path.join(output_dir, f'{name}_fixed_t2.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close()        
    
    sub_axix = filter(lambda x:x%200 == 0, x)
    #plt.title('Initial condition')
    plt.plot(t_2,u[:,128] , 'r-',linewidth = 2, label='Input')
    plt.plot(t_2, h[:,128] , 'k--',linewidth =2, label='ture result')
    plt.legend() # 显示图例
    plt.xlabel('t')
    plt.ylabel('value ')
    output_file = os.path.join(output_dir, f'{name}_fixed_x2.png')
# 保存图片为高分辨率 PNG
    plt.savefig(output_file, dpi=300, bbox_inches='tight') 
    plt.show()
    plt.close() 

print("load module")

def central_difference(Exact, dt):
    dExact_dt = np.zeros_like(Exact)  # 初始化导数矩阵
    N = Exact.shape[0]  # 时间点数量
    
    # 使用中心差分法计算内部点的导数
    dExact_dt[1:-1] = (Exact[2:] - Exact[:-2]) / (2 * dt)
    
    # 处理边界点
    dExact_dt[0] = (Exact[1] - Exact[0]) / dt  # 前向差分
    dExact_dt[-1] = (Exact[-1] - Exact[-2]) / dt  # 后向差分
    
    return dExact_dt

# In[load data] 
data = scipy.io.loadmat(mat_file_path)
t = 2*5*np.real(data['t'].flatten()[:,None])/10 #100 *  -85
x = np.real(data['x'].flatten()[:,None])/10 # -25
Exact = 2*np.real(data['u3'])
dt = t[1] - t[0]
Exact_v = central_difference(Exact, dt)
a0 = 0
length1 = 0
length2 = 40
t = t[a0:a0+length2]
t_domain = [t[0],t[-1]]
points = 10000
tf = sample(t_domain, points)
x_domain = [x[0],x[-1]]
xf = sample(x_domain, 200)
x = x
t1= [t[0]]


Exact = Exact[a0:a0+length2,:]
Exact_v = Exact_v[a0:a0+length2,:]
Exact1 = np.vstack((Exact[0, :]))

X, T = np.meshgrid(xf,tf)

X1, T1 = np.meshgrid(x,t1)
Exact_v1 = np.vstack((Exact_v[0, :]))


measurement = np.hstack((X1.flatten()[:,None], T1.flatten()[:,None],Exact1.flatten()[:,None] )) 
init = np.hstack((X1.flatten()[:,None], T1.flatten()[:,None],Exact_v1.flatten()[:,None])) #pde
m_sample = measurement#[np.random.choice(measurement.shape[0],Nm),:]
init_sample = init

Nf = 60000
f = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
f_sample = f[np.random.choice(f.shape[0],Nf),:]

print(f"load experience data Nf = {Nf} pred_length= {length2} start_point= {a0}")

lambda_1_history = []
layers = [2,50,100,3]
model = PINNInference(x_max=1,t_max=1,measurement=m_sample,f=f_sample ,Nf=Nf,init_xt=init_sample, layers=layers, device=device, a=x[-1].item(), b=1, c=1, d=1)

figure_compare(name='per')
# lambda1*loss_m + lambda2*loss_f + lambda3*loss_f2 + lambda4*loss_ut + lambda5*loss_b
model.train(niter=20001,Ir=0.001,lambda1=1,lambda2=0.01,lambda3=1,lambda5=1,plot_num=1000,freezing=True)
figure_compare(name='pertraining2')

model.train(niter=30001,Ir=0.001,lambda1=1,lambda2=0.1,lambda3=1,plot_num=1000,freezing=True)
figure_compare(name='train2')
model.train(niter=30001,Ir=0.001,lambda1=1,lambda2=1,lambda3=1,plot_num=1000,freezing=True)
figure_compare(name='train3')
model.train(niter=30001,Ir=0.0005,lambda1=1,lambda2=1,lambda3=1,plot_num=1000,freezing=True)
model.train(niter=30001,Ir=0.0005,lambda1=1,lambda2=10,lambda3=10,plot_num=1000,freezing=True)
figure_compare(name='train3')
model.train(niter=30001,Ir=0.0001,lambda1=1,lambda2=100,lambda3=10,plot_num=1000,freezing=True)
figure_compare(name='train4')
model.train(niter=30001,Ir=0.0001,lambda1=1,lambda2=100,lambda3=10,plot_num=1000,freezing=True)
figure_compare(name='train5')
model.train(niter=30001,Ir=0.00005,lambda1=1,lambda2=1000,lambda3=10,plot_num=1000,freezing=True)
figure_compare(name='train6')
for j in range(0):
    # 使用路径加载 .mat 文件
    data = scipy.io.loadmat(mat_file_path)
    t = np.real(data['t'].flatten()[:,None]) #100 *  -85
    x = np.real(data['x'].flatten()[:,None])/10 # -25
    length2 = length2+5
    t = t[a0:a0+length2]
    t_domain = [t[0],t[-1]]
    points =points + 200
    tf = sample(t_domain, points)
    x_domain = [x[0],x[-1]]
    xf = sample(x_domain, 200)
    x = x
    Exact = np.real(data['u3']).T
    Exact = 2*np.transpose(Exact)
    Exact = Exact[a0:a0+length2,:]
    
    X, T = np.meshgrid(xf,tf)
    
    Nf = Nf + 2000
    f = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    f_sample = f[np.random.choice(f.shape[0],Nf),:]

    model.f_x = torch.tensor(f_sample[:, 0:1], requires_grad=True).float().to(model.device)
    model.f_t = torch.tensor(f_sample[:, 1:2], requires_grad=True).float().to(model.device)
    print(f"load experience data Nf = {Nf} pred_length= {length2} start_point= {a0}")
    model.train(niter=10001,Ir=0.00005,lambda1=1,lambda2=10,plot_num=1000,freezing=True)
    model.train(niter=10001,Ir=0.00005,lambda1=1,lambda2=10,plot_num=1000,freezing=True)
    model.train(niter=10001,Ir=0.00001,lambda1=1,lambda2=100,plot_num=1000,freezing=True)
    name = f"train_{j+7}"
    figure_compare(name)

# 4. 保存模型
torch.save(model.dnn.state_dict(), os.path.join(output_dir, "model.pth"))