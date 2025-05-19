import torch
import torch.nn.functional as F

try:
    from pytorch_msssim import ssim
    use_ssim = True
except ImportError:
    use_ssim = False

try:
    import lpips
    use_lpips = True
except ImportError:
    use_lpips = False

try:
    from geomloss import SamplesLoss
    use_sinkhorn = True
except ImportError:
    use_sinkhorn = False

class VGGPerceptualLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import vgg16
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        vgg = vgg16(pretrained=True).features[:16].to(self.device)
        for p in vgg.parameters():
            p.requires_grad = False
        self.vgg = vgg
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=self.device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=self.device).view(1, 3, 1, 1)

    def forward(self, x, y):
        x = (x - self.mean) / self.std
        y = (y - self.mean) / self.std
        return F.mse_loss(self.vgg(x.to(self.device)), self.vgg(y.to(self.device)))


class LPIPSLoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.lpips_model = lpips.LPIPS(net='vgg').to(self.device)
        self.lpips_model.eval()
        for p in self.lpips_model.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        x = x.to(self.device)
        y = y.to(self.device)
        return self.lpips_model(x, y).mean()


class PSNRLoss(torch.nn.Module):
    def forward(self, x, y):
        mse = F.mse_loss(x, y)
        psnr = -10 * torch.log10(mse + 1e-8)
        return -psnr


class TVLoss(torch.nn.Module):
    def forward(self, x):
        batch_size = x.size(0)
        h_tv = torch.pow(x[:, :, 1:, :] - x[:, :, :-1, :], 2).sum()
        w_tv = torch.pow(x[:, :, :, 1:] - x[:, :, :, :-1], 2).sum()
        return (h_tv + w_tv) / batch_size


class SinkhornLoss(torch.nn.Module):
    def __init__(self, p=2, blur=0.05, scaling=0.9):
        super().__init__()
        self.sinkhorn = SamplesLoss("sinkhorn", p=p, blur=blur, scaling=scaling)

    def forward(self, x, y):
        B, C, H, W = x.shape
        x_flat = x.view(B, C, -1).transpose(1, 2)
        y_flat = y.view(B, C, -1).transpose(1, 2)

        loss = 0
        for xb, yb in zip(x_flat, y_flat):
            a = torch.ones(xb.shape[0], device=xb.device) / xb.shape[0]
            b = torch.ones(yb.shape[0], device=yb.device) / yb.shape[0]
            loss += self.sinkhorn(a, xb, b, yb)
        return loss / B



def get_loss_fn(name):
    if name == "mse":
        return F.mse_loss
    elif name == "l1":
        return F.l1_loss
    elif name == "ssim":
        if use_ssim:
            return lambda x, y: 1 - ssim(x, y, data_range=1.0)
        else:
            raise ImportError("ssim loss requires `pytorch-msssim`")
    elif name == "vgg":
        return VGGPerceptualLoss()
    elif name == "lpips":
        if use_lpips:
            return LPIPSLoss()
        else:
            raise ImportError("lpips loss requires `lpips`")
    elif name == "psnr":
        return PSNRLoss()
    elif name == "tv":
        return TVLoss()
    elif name == "sinkhorn":
        if use_sinkhorn:
            return SinkhornLoss()
        else:
            raise ImportError("sinkhorn loss requires `geomloss`")
    else:
        raise ValueError(f"Unknown loss function: {name}")