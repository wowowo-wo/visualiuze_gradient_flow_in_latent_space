from PIL import Image
from torchvision import transforms

def load_and_preprocess(image_path):
    img_pil = Image.open(image_path).convert("RGB").resize((512,512))
    transform=transforms.ToTensor()
    return transform(img_pil).unsqueeze(0), img_pil