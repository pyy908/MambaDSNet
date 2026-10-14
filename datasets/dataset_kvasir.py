import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF

class KvasirDataset(Dataset):
    def __init__(self, data_root, img_size=352, mode='train'):
        self.data_root = data_root
        self.img_size = img_size
        self.mode = mode
        
        self.image_dir = os.path.join(data_root, 'image')
        self.mask_dir = os.path.join(data_root, 'masks')
        
        self.samples = [f for f in os.listdir(self.image_dir) if f.endswith(('.png', '.jpg'))]
        
    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        name = self.samples[idx]
        img_path = os.path.join(self.image_dir, name)
        m_name = name if name.endswith('.png') else name.replace('.jpg', '.png')
        if not os.path.exists(os.path.join(self.mask_dir, m_name)):
            # Fallback if names match exactly
            m_name = name
            
        mask_path = os.path.join(self.mask_dir, m_name)
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
        
        if self.mode == 'train':
            # Random Scale
            scale = random.choice([0.75, 1.0, 1.25])
            if scale != 1.0:
                w, h = image.size
                image = image.resize((int(w * scale), int(h * scale)), Image.BILINEAR)
                mask = mask.resize((int(w * scale), int(h * scale)), Image.NEAREST)
                
            # Resize
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            
            # Random Horizontal Flip
            if random.random() > 0.5:
                image = TF.hflip(image)
                mask = TF.hflip(mask)
            
            # Random Vertical Flip
            if random.random() > 0.5:
                image = TF.vflip(image)
                mask = TF.vflip(mask)
                
            # Color Jitter
            if random.random() > 0.5:
                # brightness
                enhancer = ImageEnhance.Brightness(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                # contrast
                enhancer = ImageEnhance.Contrast(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
            if random.random() > 0.5:
                # saturation
                enhancer = ImageEnhance.Color(image)
                image = enhancer.enhance(random.uniform(0.8, 1.2))
                
        else:
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
            mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)
            
        image = np.array(image, dtype=np.float32) / 255.0
        mask = np.array(mask, dtype=np.float32) / 255.0
        mask[mask >= 0.5] = 1.0
        mask[mask < 0.5] = 0.0
        
        # Normalize
        image = (image - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        image = image.transpose((2, 0, 1))
        
        image = torch.from_numpy(image).float()
        mask = torch.from_numpy(mask).float()
        
        return {'image': image, 'label': mask}
