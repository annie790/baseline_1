def matlab_style_gauss2D(shape=(3,3),sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m,n = [(ss-1.)/2. for ss in shape]
    y,x = np.ogrid[-m:m+1,-n:n+1]
    h = np.exp( -(x*x + y*y) / (2.*sigma*sigma) )
    h[ h < np.finfo(h.dtype).eps*h.max() ] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


class Gauss_Conv2d(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size,stride,same_padding=True):
        super(Gauss_Conv2d, self).__init__()
        padding = int((kernel_size - 1) / 2) if same_padding else 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding)

    def forward(self, x):
        x = self.conv(x)
        return x

def cc():
    gauss_kernel = matlab_style_gauss2D(shape=[11,11],sigma=1.5)
    gauss_kernel = np.expand_dims(gauss_kernel,-1)
    gauss_kernel = np.expand_dims(gauss_kernel,-1)
    gauss_kernel = gauss_kernel.swapaxes(0,2)
    gauss_kernel = gauss_kernel.swapaxes(1,3)
    # (1, 1, 11, 11)
    gauss_kernel_1 = torch.from_numpy(gauss_kernel).float().cuda()

    return gauss_kernel_1
    

class GaussNet(nn.Module):
    def __init__(self):
        super(GaussNet,self).__init__()
        self.Gauss_conv = Gauss_Conv2d(1, 1, kernel_size=11,stride=1)
        self.Gauss_conv.conv.weight.data = cc()

        # self.sum = Gauss_Conv2d(1, 1, kernel_size=512,stride=512)
        # self.sum.conv.weight.data = a.view(1,1,512,512)

        # Gauss_conv.weight
        # self.Gauss_conv.conv.weight
        

        
    def forward(self, pred,input_gt):

        # gauss_kernel = matlab_style_gauss2D(shape=[11,11],sigma=1.5)
        # gauss_kernel = np.expand_dims(gauss_kernel,-1)
        # gauss_kernel = np.expand_dims(gauss_kernel,-1)
        # gauss_kernel = gauss_kernel.swapaxes(0,2)
        # gauss_kernel = gauss_kernel.swapaxes(1,3)
        # # (1, 1, 11, 11)
        # pdb.set_trace()
        # gauss_kernel_1 = torch.from_numpy(gauss_kernel).float().cuda()
        


        # pdb.set_trace()
        x = input_gt.view(BATCH_SIZE,-1,512,512)
        y = pred.view(BATCH_SIZE,-1,512,512)
        u_x = self.Gauss_conv(x)
        u_y = self.Gauss_conv(y)
        siga_x = self.Gauss_conv(torch.pow(x-u_x,2))
        siga_y = self.Gauss_conv(torch.pow(y-u_y,2))
        siga_xy = self.Gauss_conv((x-u_x)*(y-u_y))

        # pdb.set_trace()
        C1 = (0.01*255)*(0.01*255)
        C2 = (0.03*255)*(0.03*255)
        a = (2*u_x*u_y + C1)*(2*siga_xy + C2)
        bbb = (torch.pow(u_x,2)+torch.pow(u_y,2)+C1)*(torch.pow(siga_x,2)+torch.pow(siga_y,2)+C2)

        ssim = a/bbb


        ssim = ssim.view(BATCH_SIZE,-1,512,512)
        # mssim = self.sum(ssim)

        mssim = torch.mean(torch.sum(ssim, 1))
        # batch avg


        return mssim

