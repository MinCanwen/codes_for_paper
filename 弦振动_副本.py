import torch
from torch import nn
#from torch.autograd import Variable

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import time
from scipy.interpolate import griddata

from typing import OrderedDict

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class Net(nn.Module):
    def __init__(self, layers):
        """
        Args:
            layers: 数组, 存放神经网络每一层的节点数
        """
        super(Net, self).__init__()

        self.layers_cnt = len(layers) - 1
        layers_seq = list()
        #layers_seq.append(('input_layer', nn.Linear(layers[0], layers[1])))
        for i in range(len(layers) - 2):
            layers_seq.append(('hidden_layer_%d' % i, nn.Linear(layers[i], layers[i+1])))
            layers_seq.append(('actvation_%d' % i, nn.Sigmoid()))
        layers_seq.append(('output_layer', nn.Linear(layers[-2], layers[-1])))

        self.layers = nn.Sequential(OrderedDict(layers_seq))
        

    def forward(self, x):
        output = self.layers(x)
        return output


class PINNInference():
    def __init__(self, 
                 x_max=1,
                 t_max=1,
                 init_xt=None, 
                 init_xt1=None,
                 left_bound_xt=None,
                 right_bound_xt=None,
                 f_xt=None,
                 layers=None,
                 a=0., b=0., c=0., d=0.,
                 device=None):
        """
        
        Args:
            init_xt: 训练集中的初始点(x, t)
            init_xt1: 训练集中的第二个初始条件的初始点(x, t)
            left_bound_xt: 训练集中的左边界点(x, t)
            right_bound_xt: 训练集中右边界点(x, t)
            f_xt: 训练集中内部点的(x, t), 用于计算pde residual
            layers: 网络结构
            a,b,c,d为方程参数，a,h,l
            device: 设备
        """
        self.device=device        
        
        # 预先sample，直接读入device，在迭代中不变
        # init data  全部使用, 直接放入device中
        if init_xt is not None:
            self.init_x = torch.tensor(init_xt[:, 0:1], requires_grad=True).float().to(self.device)
            self.init_t = torch.tensor(init_xt[:, 1:2], requires_grad=True).float().to(self.device)
        
        if init_xt1 is not None:
            self.init_x1 = torch.tensor(init_xt1[:, 0:1], requires_grad=True).float().to(self.device)
            self.init_t1 = torch.tensor(init_xt1[:, 1:2], requires_grad=True).float().to(self.device)
        
        # left bound data  全部使用
        if left_bound_xt is not None:
            self.left_bound_x = torch.tensor(left_bound_xt[:, 0:1], requires_grad=True).float().to(self.device)
            self.left_bound_t = torch.tensor(left_bound_xt[:, 1:2], requires_grad=True).float().to(self.device)

        # right bound data 全部使用
        if right_bound_xt is not None:
            self.right_bound_x = torch.tensor(right_bound_xt[:, 0:1], requires_grad=True).float().to(self.device)
            self.right_bound_t = torch.tensor(right_bound_xt[:, 1:2], requires_grad=True).float().to(self.device)

        # f data
        if f_xt is not None:
            self.f_x = torch.tensor(f_xt[:, 0:1], requires_grad=True).float().to(self.device)
            self.f_t = torch.tensor(f_xt[:, 1:2], requires_grad=True).float().to(self.device)

        self.layers = layers
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.x_max = x_max
        self.t_max = t_max
 
        # deep neural networks
        self.dnn = Net(layers).to(device)
        
        self.iter = 0
        

    def net_u(self, x): 
        x[:,0:1]=x[:,0:1]/self.x_max
        x[:,1:2]=x[:,1:2]/self.t_max
        u = self.dnn(x)#torch.cat拼接x,t矩阵，竖向
        return u
    
    def net_left_bound(self, x):
        """ 计算左边界点条件的residual u(x=0)=0
        """
        u = self.net_u(x)
        return u
    
    def net_right_bound(self, x):
        """ 计算右边界点条件的residual u(x=l)=0
        """
        u = self.net_u(x)
        return u
        
    def net_init(self, x):
        """ 计算起始点条件的residual u(0)= 4h(x-l)l/l^2
        """
        u = self.net_u(x)
        return u - 4 * self.b * (x-self.c) * x / self.c**2
    
    def net_init1(self, x):
        """ 计算起始点条件的residual u_t(t=0)=0
        """
        u = self.net_u(x)
        u_xt = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_t = u_xt[:,1].unsqueeze(-1)
        return u_t

    def net_f(self, x):
        """ 计算pde residual u_t - a^2*u_xx=0
        """
        u = self.net_u(x)
        u_xt = torch.autograd.grad(
            u, x,
            grad_outputs=torch.ones_like(u),
            retain_graph=True,
            create_graph=True
        )[0]
        u_x = u_xt[:,0].unsqueeze(-1)
        u_t = u_xt[:,1].unsqueeze(-1)
        u_xx = torch.autograd.grad(
            u_x, x, 
            grad_outputs=torch.ones_like(u_x),
            retain_graph=True,
            create_graph=True
        )[0][:,0].unsqueeze(-1)
        u_tt = torch.autograd.grad(
            u_t, x, 
            grad_outputs=torch.ones_like(u_t),
            retain_graph=True,
            create_graph=True
        )[0][:,1].unsqueeze(-1)
        f = u_tt - self.a **2 * u_xx
        return f
    
    def train_0_b_f(self, Ir=0.05,niter=0, plotting=False, lambda1=1.0, lambda2=1.0, lambda3=1.0, lambda4=1.0):
        """ mse = lambda1 * mse_ic + lambda2 * mse_b + lambda3 * mse_f + lambda4 * mse_ic1
        """
        init_xt1 = torch.cat([self.init_x, self.init_t], dim=1)
        init_xt2 = torch.cat([self.init_x1, self.init_t1], dim=1)
        left_bound_xt1 = torch.cat([self.left_bound_x, self.left_bound_t], dim=1)
        right_bound_xt1 = torch.cat([self.right_bound_x, self.right_bound_t], dim=1)
        f_xt1 = torch.cat([self.f_x, self.f_t], dim=1)
        optim = torch.optim.Adam(self.dnn.parameters(), Ir)
        self.dnn.train()
        start_time = time.time()
        
        loss_list = []
        t_list = range(niter)
        min_loss = 100

        for it in range(niter):
            init_pred = self.net_init(init_xt1)                             #越接近0越好
            left_bound_pred = self.net_left_bound(left_bound_xt1) 
            right_bound_pred = self.net_right_bound(right_bound_xt1)    #越接近0越好
            init2_pred = self.net_init1(init_xt2) #越接近0越好
            f_pred = self.net_f(f_xt1)

            loss_ic = torch.mean((init_pred) ** 2)
            loss_b = torch.mean((left_bound_pred) ** 2) + torch.mean((right_bound_pred) ** 2)
            loss_ic1 = torch.mean((init2_pred) ** 2)
            loss_f = torch.mean(f_pred ** 2)
            loss =  lambda3 * loss_f + lambda1 * loss_ic +  lambda2 * loss_b + lambda4 * loss_ic1
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            loss_list.append(loss.item())
            
            if loss < min_loss:
                min_loss = loss
                #print("save model")
                # 保存模型语句
                torch.save(self.dnn.state_dict(),"model.pth")
                epoch = it
            

            if it % 500 == 0:
                elapsed = time.time() - start_time
                print(
                    'Iter %d, Loss: %.5e, Loss_ic: %.5e, Loss_b: %.5e,Loss_ic1: %.5e, Loss_f: %.5e, time: %.2f' % ( \
                     it, loss.item(), loss_ic.item(), loss_b.item(),loss_ic1.item(), loss_f.item(), elapsed)
                )

                start_time = time.time()

        print('The best model is at iteration %d'% ( \
                     epoch))        
        # 画loss关于iter的图
        begin_iter = 200
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
        x = torch.cat([x,t],dim=1)
        
        #load 最佳模型
        best_model = Net(layers).to(device)
        best_model.load_state_dict(torch.load("model.pth"))
        best_model.eval()
        u = best_model(x) #pde solution
        f = self.net_f(x) #pde residual
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()

        '''self.dnn.eval()
        u = self.net_u(x) #ped solution
        f = self.net_f(x) #pde residual
        u = u.detach().cpu().numpy()
        f = f.detach().cpu().numpy()'''
        return u, f

def sample(domain, num):
    min = domain[0]
    max = domain[1]
    x = min + (max-min) * np.random.uniform(0,1,size=num)
    return x

x = [0,0.5]
t = [0,0.3]
init_num = 500
bound_num = 500

t_sample = sample(t,bound_num) # 时间区间上随机取样，对应边界条件
x_l = x[0] * np.zeros_like(t_sample)
left_bound_xt = np.squeeze(np.stack((x_l.flatten()[:, None],t_sample.flatten()[:, None]),-1))

t_sample = sample(t,bound_num)
x_r = x[1] * np.ones_like(t_sample)
right_bound_xt = np.squeeze(np.stack((x_r.flatten()[:, None],t_sample.flatten()[:, None]),-1))

x_sample = sample(x,init_num) # 空间上取样，对应初始条件
t_1 = np.zeros_like(x_sample)
init_xt = np.squeeze(np.stack((x_sample.flatten()[:, None],t_1.flatten()[:, None]),-1))

x_sample = sample(x,init_num) # 空间上取样，对应初始条件
init_xt1 = np.squeeze(np.stack((x_sample.flatten()[:, None],t_1.flatten()[:, None]),-1))

#设置内部点采样
t_sample = sample(t,60)
#t_sample = np.append(sample(t,300),sample([1,4], 50))
x_sample = sample(x,60)
X,T = np.meshgrid(x_sample,t_sample)
f_xt = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))

layers = [2,20,20,20,1]
a = (86/4.67e-4)**0.5 #张力86N/线密度4.67e-4kg/m 钢琴琴弦
b = 0.01 #h
c = 0.50 #l
model = PINNInference(init_xt=init_xt, init_xt1=init_xt1,left_bound_xt=left_bound_xt, 
                      right_bound_xt=right_bound_xt,f_xt=f_xt, 
                      layers=layers, device=device, a=a, b=b, c=c, d=1)
#参数意义 a=a,b=A,c=w,d=l
model.train_0_b_f(Ir=0.05,niter=20000, plotting=True, 
                  lambda1=1, lambda2=1, lambda3=1,lambda4=1)
#改变下网络结构和激活函数，能不能光滑一下loss曲线
#学习率
# In['plot'] 

x_lower = 0
x_upper = 0.5
t_lower = 0
t_upper =0.3

# 创建 2D 域（用于绘图和输入）
x1=  np.linspace(x_lower, x_upper, 256)
t1= np.linspace(t_lower, t_upper, 201)
B, C = np.meshgrid(x1,t1)

# 整个域变平
B_star = np.hstack((B.flatten()[:, None], C.flatten()[:, None]))

# 做预测
prediction = model.predict(B_star)
arr1 = np.array(prediction,dtype=float)
u = griddata(B_star, arr1[0,:], (B, C), method="cubic")#
u = np.squeeze(u)

#计算误差
plt.title('Error')
plt.imshow(
    u.T,
    interpolation="nearest",
    cmap="viridis",
    extent=[t_lower, t_upper, x_lower, x_upper],
    origin="lower",
    aspect="auto",
    #norm = mpl.colors.Normalize(vmin=0, vmax=np.max(h)),
)
plt.colorbar()
plt.show()


#model.train_0_b_f(Ir=0.005,niter=20000, plotting=True, 
#                  lambda1=1, lambda2=1, lambda3=100,lambda4=1)




