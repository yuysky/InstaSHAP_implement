import torch
import torch.nn as nn
import numpy as np
from tqdm.auto import tqdm
from torch.utils.data import TensorDataset, DataLoader

class DataAugmenter:
    """
    Responsible for data augmentation strategies to generate samples near the data manifold.
    Strategies include:
    1. Mixup: Linear interpolation to fill gaps between samples.
    2. Local Noise: Local Gaussian perturbation to improve robustness.
    """
    def __init__(self, noise_scale=0.05, mixup_alpha=0.2):
        self.noise_scale = noise_scale
        self.mixup_alpha = mixup_alpha
        self.rng = np.random.default_rng(seed=37)

    def augment(self, x, strategy='mixup'):
        """
        Input tensor x, return augmented tensor x_aug.
        Strategy: 'noise', 'mixup', or 'mixed' (randomly choose).
        """
        batch_size = x.shape[0]
        device = x.device
        
        if strategy == 'mixed':
            # 50% probability for mixup, 50% for noise
            strategy = 'mixup' if self.rng.random() > 0.5 else 'noise'

        if strategy == 'noise':
            # Add small Gaussian noise to stay within the local manifold
            noise = torch.randn_like(x) * self.noise_scale
            return x + noise
            
        elif strategy == 'mixup':
            # Mixup: x' = lambda * x_i + (1-lambda) * x_j
            # Randomly shuffle indices
            indices = torch.randperm(batch_size).to(device)
            lam = self.rng.beta(self.mixup_alpha, self.mixup_alpha)
            # Limit lambda to avoid extreme values, ensuring mixture
            lam = max(0.1, min(lam, 0.9))
            
            x_aug = lam * x + (1 - lam) * x[indices]
            return x_aug
        
        return x

class TeacherWrapper:
    """
    Wraps a black-box model (e.g., XGBoost, LightGBM, or sklearn models) 
    to behave like a PyTorch model for easier integration in the training loop.
    """
    def __init__(self, model, device='cpu', is_torch=False):
        self.model = model
        self.device = device
        self.is_torch = is_torch

    def predict(self, x_tensor):
        """
        Input Tensor, output prediction as Tensor (logits or values).
        """
        if self.is_torch:
            self.model.eval()
            with torch.no_grad():
                return self.model(x_tensor)
        else:
            # For XGBoost/Sklearn, convert back to CPU numpy
            x_np = x_tensor.cpu().numpy()
            try:
                # Try to get predictions
                # Note: For regression, use predict directly. 
                # For classification, consider using predict_proba or raw output margins.
                preds = self.model.predict(x_np)
            except:
                preds = self.model.predict(x_np)
            
            # Convert back to Tensor
            return torch.tensor(preds, dtype=torch.float32).to(self.device).view(-1, 1)

class DistillationTrainer:
    """
    Distillation Trainer.
    Uses the output of a Teacher model to train the InstaSHAP (Student) model.
    """
    def __init__(self, student_model, teacher_model, optimizer, sampler, device='cpu'):
        self.student = student_model
        self.teacher = TeacherWrapper(teacher_model, device=device, is_torch=isinstance(teacher_model, nn.Module))
        self.optimizer = optimizer
        self.sampler = sampler
        self.augmenter = DataAugmenter(noise_scale=0.02, mixup_alpha=0.2)
        self.device = device
        self.criterion = nn.MSELoss() # Distillation usually uses MSE to match Logits/Values

    def train_epoch(self, train_loader, augment_factor=1):
        """
        augment_factor: How many times to generate augmented data per batch.
        If factor=0, use only original data but labels still come from Teacher (correction).
        """
        self.student.train()
        epoch_loss = 0.0
        total_samples = 0

        # Use tqdm to show progress
        pbar = tqdm(train_loader, desc="Distilling", leave=False)
        
        for x_batch, _ in pbar:
            # Note: We ignore y_batch (true labels) here, trusting the Teacher completely.
            # Alternatively, a hybrid loss can be used: L = L_teacher + lambda * L_true
            
            x_batch = x_batch.to(self.device)
            current_batch_size = x_batch.shape[0]

            # --- 1. Data Augmentation ---
            x_train_list = [x_batch]
            if augment_factor > 0:
                for _ in range(augment_factor):
                    x_aug = self.augmenter.augment(x_batch, strategy='mixup')
                    x_train_list.append(x_aug)
            
            # Concatenate original and augmented data
            x_input = torch.cat(x_train_list, dim=0)
            
            # --- 2. Get Teacher's 'Soft Labels' ---
            # This is the core of distillation: fitting what the Teacher thinks f(x) is.
            with torch.no_grad():
                y_targets = self.teacher.predict(x_input)

            # --- 3. Sampling Mask (Shapley Sampler) ---
            # InstaSHAP requires mask S for training.
            # Note: Sampler needs to generate masks based on the current batch size.
            S_batch = self.sampler.sample(batch_size=x_input.shape[0], paired_sampling=True).to(self.device)

            # --- 4. Train Student ---
            self.optimizer.zero_grad()
            
            # Call InstaSHAP's forward (x, S)
            student_outputs = self.student(x_input, S_batch)
            
            loss = self.criterion(student_outputs, y_targets)
            loss.backward()
            self.optimizer.step()

            epoch_loss += loss.item() * x_input.size(0)
            total_samples += x_input.size(0)
            
            pbar.set_postfix({'distill_loss': loss.item()})

        return epoch_loss / total_samples

    def train(self, train_loader, num_epochs=10, augment_factor=1):
        print(f"Start Distillation Training for {num_epochs} epochs...")
        loss_history = []
        for epoch in range(num_epochs):
            loss = self.train_epoch(train_loader, augment_factor=augment_factor)
            loss_history.append(loss)
            print(f"Epoch {epoch+1}/{num_epochs}, Distillation Loss: {loss:.6f}")
        return loss_history