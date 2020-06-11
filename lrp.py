import torch

'''
this tool box support pytorch only, yet.
'''

__version__ = '1.3\nFix: R of different layers differ. -> Now they are the same. (if ignore epsilon)'
class lrp():
    def __init__(self,epsilon):
        self.epsilon = epsilon
        self.R = None
        self.x = None
        self.y = None
        self.weight = None
        self.bias = None
        self.hstride = None
        self.wstride = None
        self.hpool = None
        self.wpool = None
        return

    def clean(self):
        self.x = None
        self.y = None
        self.weight = None
        self.bias = None
        self.hstride = None
        self.wstride = None
        self.hpool = None
        self.wpool = None
        return

    def R0(self,net,inputs):
        y = net.forward(inputs)
        y = torch.nn.functional.softmax(y,1)
        _,pred = torch.max(y.detach(),1)
        self.R = torch.zeros_like(y)
        mask = torch.zeros_like(y)
        for i,item in enumerate(pred,0):
            mask[i,item.item()] = 1
        self.R = mask * y
        return

    def set_(self,layer,*arg):
        '''
        layer: 'conv', 'fc', 'avepool', 'maxpool', 'flatten'
        conv: x, weight, bias, hstride, wstride
        fc: x, weight, bias
        avepool/maxpool: x, y, hpool, wpool, hstride, wstride
        flatten: x
        '''
        self.clean()
        l = len(arg)
        if layer not in ['conv','fc','avepool','maxpool','flatten']:
            raise Exception('cannot manipulate "%s" layer.'%layer)
        if layer == 'conv':
            if l != 5:
                raise Exception('convolution layer should have had 5 arguments: x, weight, bias, hstride, wstride, but %d are given.'%l)
            self.x, self.weight, self.bias, self.hstride, self.wstride = arg
        elif layer == 'fc':
            if l != 3:
                raise Exception('fc layer should have had 3 arguments: x, weight, bias, but %d are given.'%l)
            self.x, self.weight, self.bias= arg
        elif layer == 'flatten':
            if l != 1:
                raise Exception('flatten layer should have had 1 arguments: x, but %d are given.'%l)
            self.x = arg[0]
        else:
            if l != 6:
                raise Exception('pool layer should have had 6 arguments: x, y, hpool, wpool, hstride, wstride, but %d are given.'%l)
            self.x, self.y, self.hpool, self.wpool, self.hstride, self.wstride= arg
        return

    def fc(self):
        #N: batch size
        #a: units number of this layer
        #n: units number of next layer

        #R [N,n]
        with torch.no_grad():
            w = self.weight#[n,a]
            b = self.bias#[n]
            x = self.x#[N,a]
            b.unsqueeze_(0)
            b.unsqueeze_(0)#[1,1,n]
            w = torch.transpose(w,0,1)#[a,n]
            x.unsqueeze_(2)#[N,a,1]
            w.unsqueeze_(0)#[1,a,n]
            z = w * x#[N,a,n]
            #zs = z.sum(dim=1,keepdim=True) + b#[N,1,n]
            zs = z.sum(dim=1,keepdim=True)#[N,1,n]
            zs += self.epsilon * ((zs>=0)*2-1)#[N,1,n]
            self.R = (z / zs * self.R.unsqueeze(1)).sum(dim=2)#[N,a]
        return

    def fc_v(self):
        with torch.no_grad():
            w = self.weight
            b = self.bias
            x = self.x
            w = torch.transpose(w,0,1)
            z = torch.zeros_like(x)
            for i in range(w.shape[0]):
                up = 0
                for j in range(w.shape[1]):
                    down = 0
                    up = w[i,j] * x[:,i]
                    for k in range(w.shape[0]):
                        down += w[k,j] * x[:,k]
                    down += self.epsilon * ((down>=0)*2-1)
                    z[:,i] += up / down * self.R[:,j]
            self.R = z
        return

                    

    def flatten(self):
        with torch.no_grad():
            R = self.R.reshape_as(self.x)
            self.R = R
        return

    def maxpool(self):
        #N: batch size
        #C: channel number
        #H: filter's row number
        #W: filter's column number
        with torch.no_grad():
            hpool = self.hpool
            wpool = self.wpool
            hstride = self.hstride
            wstride = self.wstride
            N, C, H, W = self.x.shape
            Hout = (H - hpool) // hstride + 1
            Wout = (W - wpool) // wstride + 1
            Rx = torch.zeros_like(self.x)
            for i in range(Hout):
                for j in range(Wout):
                    z = self.y[:,:,i:i+1,j:j+1] == self.x[:,:,i*hstride:i*hstride+hpool,j*wstride:j*wstride+wpool]
                    zs = z.sum(dim=(2,3),keepdims=True)
                    zs += self.epsilon * ((zs>=0)*2-1)
                    Rx[:,:,i*hstride:i*hstride+hpool,j*wstride:j*wstride+wpool] += (z / zs) * self.R[:,:,i:i+1,j:j+1]
            self.R = Rx
        return

    def avepool(self):
        #N: batch size
        #C: channel number
        #H: filter's row number
        #W: filter's column number
        with torch.no_grad():
            hpool = self.hpool
            wpool = self.wpool
            hstride = self.hstride
            wstride = self.wstride
            N, C, H, W = self.x.shape
            Hout = (H - hpool) // hstride + 1
            Wout = (W - wpool) // wstride + 1
            Rx = torch.zeros_like(self.x)
            for i in range(Hout):
                for j in range(Wout):
                    z = self.x[:,:,i*hstride:i*hstride+hpool,j*wstride:j*wstride+wpool]
                    zs = z.sum(dim=(2,3),keepdims=True)
                    #zs = zs / (H * W)
                    zs += self.epsilon * ((zs>=0)*2-1)
                    Rx[:,:,i*hstride:i*hstride+hpool,j*wstride:j*wstride+wpool] += (z / zs) * self.R[:,:,i:i+1,j:j+1]
            self.R = Rx
        return

    def conv(self):
        #N: batch size
        with torch.no_grad():
            w = self.weight#[cout,cin,hfilter,wfilter]
            b = self.bias#[cin]
            x = self.x
            cout, cin, hfilter, wfilter = w.shape
            hstride = self.hstride
            wstride = self.wstride
            N, C, H, W =self.x.shape
            Hout = (H - hfilter) // hstride + 1
            Wout = (W - wfilter) // wstride + 1
            Rx = torch.zeros_like(self.x)#[N,cin,H,W]
            w = w.unsqueeze(0)#[1,cout,cin,hfilter,wfilter]
            x = x.unsqueeze(1)#[N,1,cin,H,W]
            b = b.view(1,-1,1,1,1)
            R = self.R.unsqueeze(2)#[N,cout,1,Hout,Wout]
            for i in range(Hout):
                for j in range(Wout):
                    z = w * x[:,:,:,i*hstride:i*hstride+hfilter,j*wstride:j*wstride+wfilter]
                    zs = z.sum(dim=(2,3,4),keepdims=True) + b
                    zs += self.epsilon * ((zs>=0)*2-1)
                    Rx[:,:,i*hstride:i*hstride+hfilter,j*wstride:j*wstride+wfilter] += ((z / zs) * R[:,:,:,i:i+1,j:j+1]).sum(dim=1)
            self.R = Rx
        return
