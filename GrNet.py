import torch
from torch import nn

class fFRMapLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        ctx.save_for_backward(x,weight)
        return torch.matmul(weight,x)
    
    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_x = torch.matmul(weight.transpose(-1,-2), grad_output)
        grad_w = torch.matmul(grad_output, x.transpose(-1,-2))
        return grad_x, grad_w

class FRMapLayer(nn.Module):
    def __init__(self, channel, in_datadim, out_datadim):
        super().__init__()
        self.w = torch.nn.Parameter(self.init_w(channel,in_datadim,out_datadim))
    def init_w(self,channel,in_datadim,out_datadim):
        A = torch.rand(channel,in_datadim,in_datadim)
        U,_1,_2 = torch.svd(A)
        return U[:,:,:out_datadim].transpose(2,1)
    def forward(self,x):
        return fFRMapLayer.apply(x,self.w)
        
class fQRDepLayar(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, x):
        Q,R=torch.qr(x)
        ctx.save_for_backward(Q,R)
        return Q
    
    @staticmethod
    def backward(ctx, grad_output):
        Q, R = ctx.saved_tensors
        S = torch.eye(Q.shape[-2])-torch.matmul(Q,Q.transpose(-1,-2))
        mid_1 = torch.matmul(Q.transpose(-1,-2),grad_output)
        mid_2 = torch.tril(mid_1) - torch.tril(mid_1.transpose(-1,-2))
        mid_3 = torch.matmul(S.transpose(-1,-2),grad_output) + torch.matmul(Q,mid_2)
        mid_4 = torch.inverse(R).transpose(-1,-2)
        return torch.matmul(mid_3, mid_4)

class QRDepLayer(nn.Module):
    '''
    Do QR decomposition layer-wisely
    '''
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return fQRDepLayar.apply(x)
        
        
class fProjmapLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return torch.matmul(x,x.transpose(-1,-2))
    @staticmethod
    def backward(ctx, grad_output):
        x, = ctx.saved_tensors
        return 2 * torch.matmul(grad_output, x)
    
class ProjMapLayer(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self,x):
        return fProjmapLayer.apply(x)
        
        
def calcuK(S):
    b,c,h = S.shape
    Sr = S.reshape(b,c,1,h)
    Sc = S.reshape(b,c,h,1)
    K = Sc-Sr
    K = 1.0/K
    K[torch.isinf(K)]=0
    return K

class fOrthmapLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, p):
        S, U = torch.symeig(x,eigenvectors=True)
        ctx.save_for_backward(U,S)
        if(len(x.shape)==3):
            res = U[:,:,1:p]
        else:
            res = U[:,:,:,1:p]
        return res
    
    @staticmethod
    def backward(ctx, grad_output):
        U, S = ctx.saved_tensors
        b,c,h,w = grad_output
        p = h-w
        pad_zero = torch.zeros(b,c,h,p)
        grad_output = torch.cat((grad_output,pad_zero),3)
        Ut = U.transpose(-1,-2)
        K = calcuK(S)
        mid_1 = K.transpose(-1,-2)*torch.matmul(Ut,grad_output)
        mid_2 = torch.matmul(U,mid_1)
        return torch.matmul(mid_2,Ut)
        
        
class OrthMapLayer(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.p = p
    def forward(self, x):
        return fOrthmapLayer.apply(x,self.p)
        
        

class fProjPoolLayer_A(torch.autograd.Function):
    # AProjPooling  c/n ==0 
    @staticmethod
    def forward(ctx, x, n=4):
        b,c,h,w = x.shape
        ctx.save_for_backward(n)
        new_c =int(ceil(c/n))
        new_x = [ x[:,i:i+n].mean(1) for i in range(0,c,n)]
        return torch.cat(new_x,1).reshape(b,new_c,h,w)

    @staticmethod
    def backward(ctx, grad_output):
        n = ctx.saved_variables
        return torch.repeat_interleave(grad_output/n,n,1)
        

        
class ProjPoolLayer(nn.Module):
    ''' W-ProjPooling'''
    def __init__(self, n=4):
        super().__init__()
        self.n = n
    def forward(self, x):
        avgpool = torch.nn.AvgPool2d(int(sqrt(n)))
        return avgpool(x)
   
   
class ManifoldNet(nn.Module):
    def __init__(self, channel, in_datadim, out_datadim, embeddim):
        super().__init__()
        self.p = embeddim
        self.QR = QRDepLayer()
        self.ProjMap = ProjMapLayer()
        self.FR = FRMapLayer(channel, in_datadim, out_datadim)
        self.Orth = OrthMapLayer(self.p)
        self.Pool = ProjPoolLayer()
    def forward(self,x):
        x=self.FR(x)
        x=self.QR(x)
        x=self.ProjMap(x)
        x= self.Pool(x)
        x=self.Orth(x)
        return x
