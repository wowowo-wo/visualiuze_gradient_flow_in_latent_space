import torch
import argparse
from diffusers import AutoencoderKL
from torchvision.models import vgg16
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
from .losses import get_loss_fn
from .utils import load_and_preprocess
import imageio
import numpy as np
from PIL import Image
import random

vgg_preprocess = Compose([
    Resize((224, 224)), 
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406],
              std=[0.229, 0.224, 0.225]),
])

def extract_vgg_features(img_tensor, device):
    vgg = vgg16(pretrained=True).features[:16].to(device)
    vgg.eval()
    with torch.no_grad():
        return vgg(img_tensor)


def run_optimization(args):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
    vae.eval()

    target_img, target_img_pil = load_and_preprocess(args.image_path)
    target_img = target_img.to(device)
    
    with torch.no_grad():
        latent_target = vae.encode(target_img).latent_dist.mean

    seed = random.randint(0, 2**32 - 1)
    torch.manual_seed(seed)
    z = torch.empty_like(latent_target).uniform_(-3, 3)
    z = z.detach()
    z.requires_grad_()

    optimizer = torch.optim.Adam([z], lr=args.lr)
    loss_fn = get_loss_fn(args.loss_type)

    if args.loss_type == "sinkhorn":
        vgg_ready = vgg_preprocess(target_img_pil).unsqueeze(0).to(device)
        target_img = extract_vgg_features(vgg_ready, device)


    frames = []

    for step in range(args.steps):
        optimizer.zero_grad()
        decoded = vae.decode(z).sample
        loss = loss_fn(decoded, target_img)
        loss.backward()
        optimizer.step()
        print(f"Step {step}, Loss: {loss.item():.4f}")

        if step % args.skip_step == 0:
            img_np = decoded.clamp(0, 1).detach().cpu().squeeze().permute(1, 2, 0).numpy()
            img_np = (img_np * 255).astype(np.uint8)
            frame = Image.fromarray(img_np) 
            frames.append(frame)          
        
    images = frames
    imageio.mimsave(args.output_path, images, duration=0.3)
    print(f"saved in {args.output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path", type=str, required=True)
    parser.add_argument("--loss_type", type=str, default="mse", choices = ["mse", "l1", "ssim", "vgg", "lpips", "psnr", "tv", "sinkhorn"])
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--skip_step", type=int, default = 1)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--output_path", type=str, default="grad_result.gif")
    args = parser.parse_args()
    run_optimization(args)