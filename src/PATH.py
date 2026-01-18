"""
Gene Expression Prediction Model with LoRA Fine-tuning
Uses CTransPath for image embeddings and MLP head for gene expression prediction.
Supports LoRA fine-tuning of the embedding model.
"""

import sys
sys.path.append('./TransPath')
sys.path.append('./src')
import torch
import torch.nn as nn
import torch.nn.functional as F
from TransPath.ctran import ctranspath
from typing import Optional, Tuple
import os
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
import src.Loss as Loss
from sklearn.metrics import roc_auc_score
import numpy as np
import gc
from scipy.stats import spearmanr
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#DEVICE = torch.device("cpu")
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class HierarchicalConfounderDiscriminator(nn.Module):
    """
    Option A: p(dataset|z) * p(sample|dataset,z) * p(slide|sample,dataset,z)

    Usage (training):
        d_logits, s_logits, l_logits = disc(z, dataset_id=d, sample_id=s)
        loss_adv = CE(d_logits, d) + CE(s_logits, s) + CE(l_logits, l)
    """
    def __init__(
        self,
        input_dim: int,
        num_datasets: int,
        num_samples: int,
        num_slides: int,
        hidden_dim: int = 250,
        emb_dim: int = 32,
        lambda_dataset: float = 1.0,
        lambda_sample: float = 1.0,
        lambda_slide: float = 1.0,
    ):
        super().__init__()
        self.num_datasets = num_datasets
        self.num_samples = num_samples
        self.num_slides = num_slides
        self.lambda_dataset = lambda_dataset
        self.lambda_sample = lambda_sample
        self.lambda_slide = lambda_slide

        # Shared feature extractor
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(input_dim // 4, hidden_dim),
            nn.LeakyReLU(0.2, inplace=True)
            )


        # Heads
        self.dataset_head = nn.Linear(hidden_dim, num_datasets)
        self.sample_head = nn.Linear(hidden_dim , num_samples)              # conditioned on dataset
        self.slide_head  = nn.Linear(hidden_dim , num_slides)           # conditioned on dataset + sample

        # Optional: init (kept simple)

    def forward(
        self,
        x: torch.Tensor,
        dataset_id: torch.Tensor | None = None,
        sample_id: torch.Tensor | None = None,
    ):
        """
        Args:
            x: (B, input_dim) latent embeddings (after GRL upstream)
            dataset_id: (B,) long tensor of dataset labels (teacher forcing)
            sample_id: (B,) long tensor of sample labels (teacher forcing)

        Returns:
            dataset_logits: (B, num_datasets)
            sample_logits:  (B, num_samples)  (requires dataset_id)
            slide_logits:   (B, num_slides)   (requires dataset_id and sample_id)
        """
        h = self.trunk(x)

        dataset_logits = self.dataset_head(h)

        dataset_id = dataset_logits.argmax(dim=-1)
        sample_logits = self.sample_head(h)
        sample_id = sample_logits.argmax(dim=-1)
        slide_logits = self.slide_head(h)

        return dataset_logits, sample_logits, slide_logits

    def calculate_loss(
        self,
        logits,
        dataset_id: torch.Tensor,
        sample_id: torch.Tensor,
        slide_id: torch.Tensor,
    ):
        """
        Computes hierarchical adversarial CE loss.

        Args:
            logits: tuple returned by forward()
            dataset_id, sample_id, slide_id: ground truth labels

        Returns:
            total_loss (scalar)
            dict of individual losses (for logging)
        """
        dataset_logits, sample_logits, slide_logits = logits
        dataset_id = dataset_id.to(torch.long)
        sample_id = sample_id.to(torch.long)
        slide_id = slide_id.to(torch.long)

        losses = {}

        # Dataset
        loss_dataset = F.cross_entropy(dataset_logits, dataset_id)
        losses["adv_dataset"] = loss_dataset

        total_loss = self.lambda_dataset * loss_dataset

        # Sample (conditional)
        if sample_logits is not None:
            loss_sample = F.cross_entropy(sample_logits, sample_id)
            losses["adv_sample"] = loss_sample
            total_loss += self.lambda_sample * loss_sample

        # Slide (conditional)
        if slide_logits is not None:
            loss_slide = F.cross_entropy(slide_logits, slide_id)
            losses["adv_slide"] = loss_slide
            total_loss += self.lambda_slide * loss_slide

        return total_loss, losses



class LoRALinear(nn.Module):
    """
    LoRA-wrapped Linear layer for efficient fine-tuning.
    Replaces a standard Linear layer with LoRA adaptation.
    """
    def __init__(self, linear_layer: nn.Linear, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        self.linear = linear_layer
        in_features = linear_layer.in_features
        out_features = linear_layer.out_features
        
        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.randn(rank, in_features) * 0.02)
        self.lora_B = nn.Parameter(torch.randn(out_features, rank) * 0.02)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Original output
        original_output = self.linear(x)
        
        # LoRA adaptation: x @ A^T @ B^T * scaling
        lora_output = F.linear(F.linear(x, self.lora_A), self.lora_B)
        
        return original_output + lora_output * self.scaling


def list_linear_layers(model: nn.Module, print_output: bool = True) -> list:
    """
    List all Linear layers in the model with their full names.
    Useful for determining target_module_names for LoRA.
    
    Args:
        model: The model to inspect
        print_output: Whether to print the list
    
    Returns:
        List of (name, in_features, out_features) tuples
    """
    linear_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            linear_layers.append((name, module.in_features, module.out_features))
            if print_output:
                print(f"  {name}: Linear({module.in_features} -> {module.out_features})")
    
    if print_output:
        print(f"\nTotal Linear layers found: {len(linear_layers)}")
    
    return linear_layers


def apply_lora_to_linear_layers(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                                target_module_names: Optional[list] = None) -> nn.Module:
    """
    Apply LoRA to Linear layers in the model, targeting attention and MLP layers.
    
    Args:
        model: The transformer model (e.g., ctranspath)
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        target_module_names: List of module name patterns (substrings) to target.
                            The function searches all Linear layers and applies LoRA
                            to any layer whose full path name contains any of these patterns.
                            
                            Examples:
                            - ['qkv', 'proj', 'mlp.fc1', 'mlp.fc2']: Targets attention and MLP layers
                            - ['qkv', 'proj']: Targets only attention layers
                            - ['attn']: Targets all layers with 'attn' in their name
                            - ['mlp']: Targets all MLP layers
                            - None: Uses default ['qkv', 'attn.qkv', 'proj', 'attn.proj', 'mlp.fc1', 'mlp.fc2']
                            
                            Use list_linear_layers() to see all available layer names.
    
    Returns:
        Modified model with LoRA layers
    """
    if target_module_names is None:
        # Default: target attention and MLP linear layers
        target_module_names = ['qkv', 'attn.qkv', 'proj', 'attn.proj', 'mlp.fc1', 'mlp.fc2']
    
    replaced_layers = []
    
    def _replace_module(parent, name, module):
        """Recursively replace target modules with LoRA versions"""
        for child_name, child_module in list(module.named_children()):
            full_name = f"{name}.{child_name}" if name else child_name
            
            # Check if this is a target Linear layer
            if isinstance(child_module, nn.Linear):
                # Check if name matches target patterns
                should_replace = any(pattern in full_name.lower() for pattern in target_module_names)
                
                if should_replace:
                    # Replace with LoRA version
                    lora_layer = LoRALinear(child_module, rank=rank, alpha=alpha)
                    setattr(module, child_name, lora_layer)
                    replaced_layers.append(full_name)
            
            # Recursively process children
            _replace_module(module, full_name, child_module)
    
    _replace_module(None, "", model)
    
    print(f"\nTotal LoRA layers applied: {len(replaced_layers)}")
    return model


class PATH(nn.Module):
    """
    Model that embeds images using CTransPath and predicts gene expression using MLP head.
    Supports LoRA fine-tuning of the embedding model.
    """
    def __init__(
        self,
        num_slides: int,
        num_samples: int,
        num_datasets: int,
        kegg_dim: int,
        embedding_dim: int = 768,
        hidden_dims: list = [256,100],
        dropout: float = 0.1,
        use_lora: bool = True,
        lora_rank: int = 8,
        lora_alpha: float = 16.0,
        freeze_backbone: bool = True,
        model_path: str = './TransPath/ctranspath.pth',
        classification_mode: bool = True,
        lambda_adv: float = 0.2,
        num_epochs_adv: int = 4,
        device: Optional[torch.device] = DEVICE,
    ):
        """
        Args:
            num_genes: Number of genes to predict
            embedding_dim: Dimension of image embeddings (fixed at 768 for Swin Tiny, parameter kept for compatibility)
            hidden_dims: List of hidden layer dimensions for MLP head
            dropout: Dropout rate for MLP head
            use_lora: Whether to use LoRA for fine-tuning
            lora_rank: LoRA rank parameter
            lora_alpha: LoRA alpha scaling parameter
            freeze_backbone: If True, freeze backbone (only train LoRA/MLP)
            model_path: Path to pretrained ctranspath weights
        """
        super().__init__()
        
        # Load CTransPath backbone
        self.backbone = ctranspath()
        self.backbone.head = nn.Identity()
        self.kegg_dim = kegg_dim
        td = torch.load(model_path)
        self.backbone.load_state_dict(td['model'], strict=True)
        # Set embedding dimension (768 for Swin Tiny)
        self.embedding_dim = 768
        self.num_slides = num_slides
        self.num_samples = num_samples
        self.num_datasets = num_datasets
        self.classification_mode = classification_mode
        self.lambda_adv = lambda_adv
        self.num_epochs_adv = num_epochs_adv
        self.device = device
        # Apply LoRA if requested
        self.use_lora = use_lora
        if use_lora:
            print("Applying LoRA to attention and MLP layers...")
            self.backbone = apply_lora_to_linear_layers(
                self.backbone, 
                rank=lora_rank, 
                alpha=lora_alpha,
                target_module_names=['qkv', 'proj', 'mlp.fc1', 'mlp.fc2']  # Target attention and MLP layers
            )
        
        # Freeze backbone if requested (LoRA parameters will still be trainable)
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                # Keep LoRA parameters trainable
                if 'lora' not in name.lower():
                    param.requires_grad = False
                else:
                    param.requires_grad = True
        
        # MLP head for gene expression prediction
        
        kegg_layers = []
        kegg_layers.append(nn.Linear(embedding_dim, 512))
        kegg_layers.append(nn.ReLU())
        kegg_layers.append(nn.Linear(512, kegg_dim))
        #kegg_layers.append(nn.BatchNorm1d(kegg_dim))
        self.kegg_head = nn.Sequential(*kegg_layers)

        self.slide_number_discriminator = HierarchicalConfounderDiscriminator(embedding_dim, self.num_datasets, self.num_samples, self.num_slides)

        # Print parameter statistics
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_ratio = trainable_params / total_params if total_params > 0 else 0.0
        
        print(f"\nParameter Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Trainable ratio: {trainable_ratio:.4f} ({trainable_ratio*100:.2f}%)")
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: Input images tensor of shape (B, C, H, W)
        
        Returns:
            gene_expressions: Predicted gene expressions of shape (B, num_genes)
        """
        # Get image embeddings
        embeddings = self.backbone(images)  # (B, embedding_dim)
        
        # Predict gene expression
        kegg_expressions = self.kegg_head(embeddings)  # (B, num_genes)
        return kegg_expressions, embeddings
    
    def get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get image embeddings without predicting gene expression.
        
        Args:
            images: Input images tensor of shape (B, C, H, W)
        
        Returns:
            embeddings: Image embeddings of shape (B, embedding_dim)
        """
        with torch.no_grad():
            embeddings = self.backbone(images)
        return embeddings
    
    def get_trainable_parameters(self):
        """
        Get trainable parameters (useful for checking LoRA setup).
        """
        trainable = []
        frozen = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable.append(name)
            else:
                frozen.append(name)
        return trainable, frozen


    
    def train_withno_adversarial(self, train_loader: DataLoader, val_loader: DataLoader, num_epochs: int = 10, learning_rate: float = 1e-4, freeze_backbone: bool = False, auc_training: bool = True) -> float:
        """
        Train the model without adversarial training.
        """
        self.train()
        self.eval()
        self.backbone.eval()
        self.kegg_head.train()
        self.slide_number_discriminator.eval()

        device = self.device
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        if auc_training:
            loss_fn = Loss.AUCMarginLoss(margin=1.0)
        else:
            loss_fn = nn.MSELoss(reduction='mean')
        print(loss_fn)
        best_auc = 0.0

        for epoch in range(num_epochs):
            self.train()
            if not freeze_backbone:
                self.backbone.train()
            self.kegg_head.train()
            self.slide_number_discriminator.eval()
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")
            all_train_predictions = []
            all_train_targets = []
            for i, batch in enumerate(train_pbar):
                if len(batch) == 2:
                    images, kegg_expressions = batch
                else:
                    images, kegg_expressions, slide_number, sample_number, dataset_number = batch

                images = images.to(device)
                kegg_expressions = kegg_expressions.to(device)

                kegg_predictions, embeddings = self(images.float())
                kegg_loss = loss_fn(kegg_predictions.float(), kegg_expressions.float())
                kegg_loss.backward()
                optimizer.step()
                train_pbar.set_postfix({"kegg_loss": f"{kegg_loss.item():.4f}"})
                all_train_predictions.append(kegg_predictions.float().detach().cpu().numpy())
                all_train_targets.append(kegg_expressions.float().detach().cpu().numpy())

            all_train_predictions = np.concatenate(all_train_predictions, axis=0)
            all_train_targets = np.concatenate(all_train_targets, axis=0)
            train_roc_auc = roc_auc_score(all_train_targets, all_train_predictions, average='macro')
            print(f"Epoch {epoch+1}/{num_epochs} - Train ROC-AUC: {train_roc_auc:.4f}")

            all_predictions = []
            all_targets = []
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]")
            self.eval()
            self.backbone.eval()
            self.kegg_head.eval()
            self.slide_number_discriminator.eval()
            for i, batch in enumerate(val_pbar):
                if len(batch) == 2:
                    images, kegg_expressions = batch
                else:
                    images, kegg_expressions, slide_number, sample_number, dataset_number = batch
                images = images.to(device)
                kegg_expressions = kegg_expressions.to(device)

                with torch.no_grad():
                    kegg_predictions, embeddings = self(images.float())
                    kegg_loss = loss_fn(kegg_predictions.float(), kegg_expressions.float())
                    all_predictions.append(kegg_predictions.float().cpu().numpy())
                    all_targets.append(kegg_expressions.float().cpu().numpy())
                val_pbar.set_postfix({"kegg_loss": f"{kegg_loss.item():.4f}"})
            
            all_predictions = np.concatenate(all_predictions, axis=0)
            all_targets = np.concatenate(all_targets, axis=0)
            mask = (all_targets.sum(axis=0) > all_targets.shape[0] * 0.05) * (all_targets.sum(axis=0) < all_targets.shape[0] * 0.95) 
            val_roc_auc = roc_auc_score(all_targets[:, mask], all_predictions[:, mask], average='macro')
            if val_roc_auc > best_auc:
                best_auc = val_roc_auc
                self.best_model = copy.deepcopy(self.state_dict())
                print("saving best model")
                print(f"*** new best ROC-AUC: {best_auc:.4f}")
            print(f"Epoch {epoch+1}/{num_epochs} - Val ROC-AUC: {val_roc_auc:.4f}")
        self.load_state_dict(self.best_model)
        return best_auc

    def train_model(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        num_epochs: int = 10,
        learning_rate: float = 1e-4,
        loss_fn: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        verbose: bool = True,
        auc_training: bool = False,
    ) -> dict:
        """
        Train the model on training data and evaluate on validation data each epoch.
        
        Args:
            train_loader: DataLoader for training data (returns images, gene_expressions)
            val_loader: DataLoader for validation data (returns images, gene_expressions)
            num_epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            loss_fn: Loss function to use (default: MSE Loss)
            device: Device to train on (default: cuda if available, else cpu)
            verbose: Whether to print training progress
        
        Returns:
            Dictionary with training history containing:
                - train_losses: List of training losses per epoch
                - val_losses: List of validation losses per epoch
        """
        print(f"Training model with lambda_adv: {self.lambda_adv} and num_epochs_adv: {self.num_epochs_adv}")
        
        self.to(self.device)
        
        # Use MSE loss by default, or PearsonCorrelationLoss if available
        if self.classification_mode and not auc_training:
            loss_fn_kegg = nn.BCEWithLogitsLoss(reduction='mean', pos_weight=torch.tensor([10]).to(self.device))
            #loss_fn_kegg =  Loss.AUCMarginLoss(margin=1.0, reduction='mean')
            print(loss_fn_kegg)
        elif self.classification_mode and auc_training:
            loss_fn_kegg = Loss.AUCMarginLoss(margin=1.0, reduction='mean')
            #self.num_epochs_adv = 0
        else:
            loss_fn_kegg = Loss.PearsonCorrelationLoss(reduction='mean')

        # Split parameters: main vs discriminator
        disc_params = list(self.slide_number_discriminator.parameters())
        main_params = [p for p in self.backbone.parameters() if p.requires_grad ] + list(self.kegg_head.parameters())

        optimizer_main = optim.Adam(main_params, lr=learning_rate)
        optimizer_disc = optim.Adam(disc_params, lr=learning_rate)
                # Training history
        history = {
            'train_losses': [],
            'val_losses': [],
            'disc_losses': [],
            'val_roc_auc': []  # Macro ROC-AUC for classification mode
        }
        
        if verbose:
            print(f"\nStarting training on {self.device}")
            print(f"Number of trainable parameters: {sum(p.numel() for p in main_params):,}")
            print(f"Number of trainable parameters: {sum(p.numel() for p in disc_params):,}")
            print(f"Training for {num_epochs} epochs\n")
        
        best_auc=0.0
        for epoch in range(num_epochs):
            # Training phase
            self.train()
            train_loss = 0.0
            disc_loss = 0.0
            num_train_batches = 0
            train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]") if verbose else train_loader
            
            for i, (images,kegg_expressions, slide_number, sample_number, dataset_number) in enumerate(train_pbar):
                images = images.to(self.device)
                slide_number = slide_number.to(self.device)
                sample_number = sample_number.to(self.device)
                dataset_number = dataset_number.to(self.device)
                kegg_expressions = kegg_expressions.to(self.device)
                # =======================
                # 1) Update discriminator
                # =======================
                optimizer_disc.zero_grad()

                with torch.no_grad():
                    _, embeddings = self(images)  # detach later anyway

                slide_number_logits, sample_number_logits, dataset_number_logits = self.slide_number_discriminator(embeddings.detach())
                logits = ( slide_number_logits, sample_number_logits, dataset_number_logits)
                disc_loss, _ = self.slide_number_discriminator.calculate_loss(logits, dataset_number, sample_number, slide_number)
                #disc_loss = disc_loss / 3.0
                disc_loss.backward()
                optimizer_disc.step()

                # ========================
                # 2) Update main model (backbone + MLP) adversarially
                # ========================
                self.train()
                optimizer_main.zero_grad()
                
                kegg_predictions, embeddings = self(images.float())
                
                kegg_loss = loss_fn_kegg(kegg_predictions.float(), kegg_expressions.float())
                # adversarial term: *maximize* disc loss ⇒ subtract it
                slide_number_logits_adv, sample_number_logits_adv, dataset_number_logits_adv = self.slide_number_discriminator(embeddings)
                logits = ( slide_number_logits_adv, sample_number_logits_adv, dataset_number_logits_adv)
                disc_loss_adv, _ = self.slide_number_discriminator.calculate_loss(logits, dataset_number, sample_number, slide_number)
               # disc_loss_adv = disc_loss_adv / 3.0

                if epoch >= self.num_epochs_adv and i % 1 == 0:
                    total_loss = kegg_loss - self.lambda_adv * disc_loss_adv
                    num_train_batches += 1
                    train_loss += total_loss.item()
                    disc_loss += disc_loss_adv.item()
                    total_loss.backward()
                    optimizer_main.step()

                else:
                    total_loss =  - self.lambda_adv * disc_loss_adv
                    total_loss.backward()
                    optimizer_main.step()

          
        
                gc.collect()
                torch.cuda.empty_cache()
                if verbose and isinstance(train_pbar, tqdm):
                    train_pbar.set_postfix({"total_loss": f'{total_loss.item():.4f}', 'kegg_loss': f'{kegg_loss.item():.4f}', 'disc_loss': f'{disc_loss_adv.item():.4f}'})
            
            avg_train_loss = train_loss / num_train_batches if num_train_batches > 0 else 0.0
            avg_disc_loss = disc_loss / num_train_batches if num_train_batches > 0 else 0.0
            history['train_losses'].append(avg_train_loss)
            history['disc_losses'].append(avg_disc_loss)
            # Validation phase
            self.eval()
            val_loss = 0.0
            num_val_batches = 0
            
            # For classification mode: collect predictions and targets for ROC-AUC
            all_predictions = []
            all_targets = []
            all_kegg_predictions = []
            all_kegg_targets = []
            
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Val]") if verbose else val_loader
            
            with torch.no_grad():
                for batch in val_pbar:

                    images,kegg_expressions, slide_number, sample_number, dataset_number = batch

                    images = images.to(self.device)
                    kegg_expressions = kegg_expressions.to(self.device)
                    slide_number = slide_number.to(self.device)
                    sample_number = sample_number.to(self.device)
                    dataset_number = dataset_number.to(self.device)
                    # Forward pass
                    kegg_predictions, _  = self(images)
                    kegg_loss = loss_fn_kegg(kegg_predictions, kegg_expressions)
                    total_loss = kegg_loss #+ kegg_loss
                    
                    val_loss += total_loss.item()
                    num_val_batches += 1
                    val_pbar.set_postfix({'kegg_loss': f'{kegg_loss.item():.4f}'})

                    
                    # Collect predictions and targets for ROC-AUC calculation (classification mode)
                    # Convert logits to probabilities using softmax
                    probs = F.sigmoid(kegg_predictions)
                    all_predictions.append(probs.cpu().numpy())
                    all_kegg_predictions.append(kegg_predictions.cpu().numpy())
                    all_kegg_targets.append(kegg_expressions.cpu().numpy())
            
            avg_val_loss = val_loss / num_val_batches if num_val_batches > 0 else 0.0
            history['val_losses'].append(avg_val_loss)
            
            # Calculate macro ROC-AUC for classification mode
            
            if len(all_predictions) > 0:
                all_kegg_predictions = np.concatenate(all_kegg_predictions, axis=0)
                all_kegg_targets = np.concatenate(all_kegg_targets, axis=0)
                mask = (all_kegg_targets.sum(axis=0) > all_kegg_targets.shape[0] * 0.05)

                
                # Calculate macro ROC-AUC (one-vs-rest)
                if self.classification_mode:    
                    try:
                        val_roc_auc = roc_auc_score(
                            all_kegg_targets[:, mask], 
                            all_kegg_predictions[:, mask], 
                            average='macro'
                        )

                        if val_roc_auc > best_auc:
                            best_auc = val_roc_auc
                            self.best_model = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                            print("saving best model")
                            print(f"*** new best ROC-AUC: {best_auc:.4f}")
                    except ValueError as e:
                        # Handle case where some classes might not be present
                        val_roc_auc = np.nan
                        if verbose:
                            print(f"Warning: Could not calculate ROC-AUC: {e}")
                
                else:

                    corr_list = []
                    for i in range(all_kegg_targets.shape[1]):
                        pred_col = all_kegg_predictions[:, i]
                        target_col = all_kegg_targets[:, i]
                        if np.std(pred_col) > 0 and np.std(target_col) > 0:
                            corr = np.corrcoef(pred_col, target_col)[0, 1]
                            corr_list.append(corr)
                                
                    val_roc_auc = np.mean(corr_list) if corr_list else np.nan

                    if val_roc_auc > best_auc:
                        self.best_model = {k: v.detach().cpu().clone() for k, v in self.state_dict().items()}
                        best_auc = val_roc_auc
                        print("saving best model")
                        print(f"*** new best ROC-AUC: {best_auc:.4f}")
                
                history['val_roc_auc'].append(val_roc_auc)
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Val ROC-AUC: {val_roc_auc:.4f}")
            else:
                print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_train_loss:.4f}, Disc Loss: {avg_disc_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        self.load_state_dict(self.best_model)
        return history

    def copy_model(self, model: 'PATH', backbone: bool = True, kegg_head: bool = True, slide_number_discriminator: bool = True) -> 'PATH':
        """
        Copy the state dict from another model to this model.
        
        Args:
            model: Source model to copy from
            
        Returns:
            self: Returns self for method chaining
        """
        # Copy state dicts for all submodules
        if backbone:
            self.backbone.load_state_dict(model.backbone.state_dict())
        if kegg_head:
            self.kegg_head.load_state_dict(model.kegg_head.state_dict())
        if slide_number_discriminator:
            self.slide_number_discriminator.load_state_dict(model.slide_number_discriminator.state_dict())
        
        # Copy other attributes if they exist
        if hasattr(model, 'use_lora'):
            self.use_lora = model.use_lora
        if hasattr(model, 'kegg_dim'):
            self.kegg_dim = model.kegg_dim
        if hasattr(model, 'num_slides'):
            self.num_slides = model.num_slides
        if hasattr(model, 'num_samples'):
            self.num_samples = model.num_samples
        if hasattr(model, 'num_datasets'):
            self.num_datasets = model.num_datasets
        if hasattr(model, 'classification_mode'):
            self.classification_mode = model.classification_mode
        
        return self

    def save_model(self, path: str):
        backbone_dict = self.backbone.state_dict()
        kegg_head_dict = self.kegg_head.state_dict()
        slide_number_discriminator_dict = self.slide_number_discriminator.state_dict()
        torch.save({
            'backbone': backbone_dict,
            'kegg_head': kegg_head_dict,
            'slide_number_discriminator': slide_number_discriminator_dict
        }, path)
        print(f"Model saved to {path}")

    def load_model(self, path: str, backbone: bool = True, kegg_head: bool = True, slide_number_discriminator: bool = False):
        checkpoint = torch.load(path)

        if backbone:
            self.backbone.load_state_dict(checkpoint['backbone'])
        
        if kegg_head:
            self.kegg_head.load_state_dict(checkpoint['kegg_head'])
        if slide_number_discriminator:
            self.slide_number_discriminator.load_state_dict(checkpoint['slide_number_discriminator'])
        print(f"Model loaded from {path}")

    def get_embeddings(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get image embeddings without predicting gene expression.
        
        Args:
            images: Input images tensor of shape (B, C, H, W)
        """
        with torch.no_grad():
            embeddings = self.backbone(images)
        return embeddings.detach().cpu().numpy()


def create_model(
    kegg_dim: int,
    num_slides: int = 0,
    num_samples: int = 0,
    num_datasets: int = 0,
    use_lora: bool = True,
    lora_rank: int = 8,
    lora_alpha: float = 16.0,
    freeze_backbone: bool = False,
    classification_mode: bool = True,
    lambda_adv: float = 0.2,
    num_epochs_adv: int = 4,
    device: Optional[torch.device] = DEVICE,
    **kwargs
) -> PATH:
    """
    Factory function to create a PATH model.
    
    Args:
        kegg_dim: Number of Pathways to predict
        use_lora: Whether to use LoRA fine-tuning
        lora_rank: LoRA rank
        lora_alpha: LoRA alpha
        freeze_backbone: Whether to freeze backbone weights
        **kwargs: Additional arguments passed to GeneExpressionPredictor
    
    Returns:
        GeneExpressionPredictor model
    """
    model = PATH(
        kegg_dim=kegg_dim,
        num_datasets=num_datasets,
        num_samples=num_samples,
        num_slides=num_slides,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        freeze_backbone=freeze_backbone,
        classification_mode=classification_mode,
        lambda_adv=lambda_adv,
        num_epochs_adv=num_epochs_adv,
        device=device,
        **kwargs
    )
    model.to(device)
    return model

