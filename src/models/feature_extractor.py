"""
Unified feature extraction wrapper for ReID models.
"""

import logging
from pathlib import Path
from typing import List, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
from torchvision.models.resnet import resnet50, ResNet50_Weights

logger = logging.getLogger(__name__)

class ReIDFeatureExtractor:
    """Unified feature extraction wrapper for all 3 ReID architectures.
    
    Loads a trained model checkpoint and extracts L2-normalized embeddings
    from person crop images. Works on CUDA, MPS (Apple Silicon), and CPU.
    """
    
    def __init__(self, model_name: str, weights_path: str = None, device: str = 'auto'):
        """
        Args:
            model_name: One of 'resnet50', 'osnet', 'transreid'
            weights_path: Path to trained .pth checkpoint
            device: 'auto' (picks best available), 'cuda', 'mps', 'cpu'
        """
        self.model_name = model_name.lower()
        self.weights_path = weights_path
        
        if device == 'auto':
            if torch.cuda.is_available():
                self.device = torch.device('cuda')
            elif torch.backends.mps.is_available():
                self.device = torch.device('mps')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)
            
        logger.info(f"Using device: {self.device}")
        
        self._build_model()
        
        self.transform = T.Compose([
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _build_model(self):
        """Builds the selected model and loads weights if provided."""
        if self.model_name == 'resnet50':
            self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1 if not self.weights_path else None)
            # Remove classification head
            self.model.fc = torch.nn.Identity()
        elif self.model_name == 'osnet':
            raise NotImplementedError("OSNet will be implemented when trained.")
        elif self.model_name == 'transreid':
            raise NotImplementedError("TransReID will be implemented when trained.")
        else:
            raise ValueError(f"Unknown model: {self.model_name}")
            
        if self.weights_path and Path(self.weights_path).exists():
            checkpoint = torch.load(self.weights_path, map_location='cpu')
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            self.model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights from {self.weights_path}")
            
        self.model.to(self.device)
        self.model.eval()
        
    @torch.no_grad()
    def extract(self, images: Union[np.ndarray, List[np.ndarray], torch.Tensor], batch_size: int = 64) -> np.ndarray:
        """Extract features from one or more person crop images.
        
        Args:
            images: Single image (H,W,3 BGR numpy), list of images, or batch tensor
            batch_size: Configurable batch size
            
        Returns:
            L2-normalized embeddings of shape (N, embedding_dim)
        """
        # Convert inputs to a list of tensors
        if isinstance(images, np.ndarray):
            if images.ndim == 3:
                images = [images]
            elif images.ndim == 4:
                images = list(images)
                
        if isinstance(images, list) and isinstance(images[0], np.ndarray):
            tensor_images = []
            for img in images:
                # Convert BGR to RGB if needed (assuming OpenCV BGR input)
                if img.shape[2] == 3:
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img)
                tensor_images.append(self.transform(pil_img))
            batch_tensor = torch.stack(tensor_images)
        elif isinstance(images, torch.Tensor):
            batch_tensor = images
            if batch_tensor.ndim == 3:
                batch_tensor = batch_tensor.unsqueeze(0)
        else:
            raise TypeError("Unsupported image input type")

        all_features = []
        num_samples = batch_tensor.size(0)
        
        for i in range(0, num_samples, batch_size):
            batch = batch_tensor[i:i+batch_size].to(self.device)
            features = self.model(batch)
            features = F.normalize(features, p=2, dim=1)
            all_features.append(features.cpu())
            
        return torch.cat(all_features, dim=0).numpy()
        
    def extract_from_paths(self, image_paths: List[str], batch_size: int = 64) -> np.ndarray:
        """Extract features from image file paths with automatic batching."""
        images = []
        for path in image_paths:
            img = cv2.imread(str(path))
            if img is not None:
                images.append(img)
            else:
                logger.warning(f"Failed to read image: {path}")
        if not images:
            return np.empty((0, 0))
        return self.extract(images, batch_size=batch_size)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    extractor = ReIDFeatureExtractor('resnet50')
    dummy_img = np.random.randint(0, 255, (256, 128, 3), dtype=np.uint8)
    features = extractor.extract(dummy_img)
    print(f"Extracted features shape: {features.shape}")
