 
from numpy.lib.function_base import append
import torch.nn as nn
import torch
from Models import  IMAGE_DIM
import random
import numpy as np
from Utils import BATCH_SIZE
from torch.autograd import Function
import torch.nn.functional as F
from Utils import grad_reverse
import seaborn as sns 
import matplotlib.pyplot as plt
from Loss import PearsonCorrelationLoss
from scipy.stats import spearmanr
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import imp
import Datasets
imp.reload(Datasets)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMBED_DIM = 100

class SimpleMLP(nn.Module):
    def __init__(self, image_embedding_dim=IMAGE_DIM, gene_expression_dim=12000, n_slides=10,p=0.3):
        super(SimpleMLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_embedding_dim, gene_expression_dim //2),
            nn.Dropout(p),
            #nn.BatchNorm1d(gene_expression_dim // 2),
           # nn.ReLU(inplace=True),
           # nn.ReLU(inplace=True),
            #nn.Dropout(p),
            nn.Linear(gene_expression_dim // 2, gene_expression_dim),
        )

    def forward(self, x):
        return self.model(x)

class MLPModel():
    def __init__(self,gene_expression_dim, image_embedding_dim, n_slides, lambda_slides=0.1, simple_mlp=False, classification_mode=False):
        self.gene_expression_dim = gene_expression_dim
        self.image_embedding_dim = image_embedding_dim
        self.init_optimizers_flag = False
        self.set_seed()
        self.classification_mode = classification_mode
        #self.model = ImageToGeneGenerator(gene_expression_dim=gene_expression_dim, image_embedding_dim=image_embedding_dim, P_mat=P).to(DEVICE)
        self.n_slides = n_slides
        self.simple_mlp = simple_mlp
        #self.model = SimpleMLP(image_embedding_dim=image_embedding_dim, gene_expression_dim=gene_expression_dim).to(DEVICE)
        if self.simple_mlp:
            self.model = SimpleMLP(image_embedding_dim=image_embedding_dim, gene_expression_dim=gene_expression_dim).to(DEVICE)

        if classification_mode:
            self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        else:   
            self.criterion = nn.MSELoss(reduction='mean')
        #self.criterion_slides = nn.CrossEntropyLoss(reduction='mean')
        self.lambda_slides = lambda_slides

    @staticmethod
    def set_seed(seed=42):
        np.random.seed(seed)
        torch.manual_seed(seed)
        random.seed(seed)
        if torch.cuda.is_available():
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False   


    def test_model(self, val_dataloader):
        self.model.eval()
        corrs_exp = []
        exps_list = []
        pred_exp_list = []
        loss_list = []
        with torch.no_grad():
            #for _, batch in enumerate(zip(dataloader_image, dataloader_ge)):
            for batch in val_dataloader:
                    if len(batch) == 3:
                        imgs, exps, _ = batch
                    else:
                        imgs, exps = batch
                    imgs, exps = imgs.to(DEVICE), exps.to(DEVICE)
                    if self.simple_mlp:
                        predicted = self.model(imgs)
                    else:
                        predicted,_ = self.model(imgs)
                    loss = self.criterion(predicted, exps.float())
                    loss_list.append(loss.item())
                    exps_list.append(exps.detach().cpu().numpy())
                    pred_exp_list.append(predicted.detach().cpu().numpy())

        
            if self.classification_mode:
                mask = (exps.sum(dim=0) > exps.shape[0] * 0.05).detach().cpu().numpy()

                val_roc_auc = roc_auc_score(
                    np.concatenate(exps_list, axis=0)[:, mask],
                    np.concatenate(pred_exp_list, axis=0)[:, mask], 
                    average='macro'
                )
                return val_roc_auc, np.concatenate(exps_list, axis=0), np.concatenate(pred_exp_list, axis=0)
            else:
                corrs_exp = []
                for i in range(exps.shape[1]):
                    exps_col = exps[:, i].detach().cpu().numpy()
                    predicted_col = predicted[:, i].detach().cpu().numpy()
                    if np.std(exps_col) > 0 and np.std(predicted_col) > 0:
                        corrs_exp.append(np.corrcoef(exps_col, predicted_col)[0,1])

        return  np.array(corrs_exp).mean(), np.concatenate(exps_list, axis=0), np.concatenate(pred_exp_list, axis=0)
    

    def plot_smooth_loss(self, loss_list, loss_name,  window_size=50):

        loss_array = np.array(loss_list)
        
        loss_smooth = np.convolve(loss_array, np.ones(window_size)/window_size, mode='valid')

        x_smooth = np.arange(window_size-1, len(loss_list))
        
        plt.figure(figsize=(12, 8))
        plt.plot(x_smooth, loss_smooth, label=f"{loss_name} Loss (smoothed)", linewidth=2)
        plt.xlabel('Training Steps')
        plt.ylabel('Loss')
        plt.title(f'{loss_name} Losse (Rolling Mean, window={window_size})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()

    def train(self,train_dataloader, val_dataloader, max_epochs=3, lr=1e-4, weight_decay=1e-5, train_every=1):
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        loss_list = []
        loss_slides_list = []
        acc_slids = []
        best_corr = 0
        for epoch in range(max_epochs):
            self.model.train()
            for batch_idx, (images, gene_expression) in enumerate(train_dataloader):
                images = images.to(DEVICE)
                gene_expression = gene_expression.to(DEVICE)

                self.optimizer.zero_grad()
                #indices = torch.randperm(images.shape[0])
                #images = torch.index_select(images, dim=0, index=indices.to(DEVICE))
                if not self.simple_mlp:
                    fake_gene_expressions, _ = self.model(images)
                else:
                    fake_gene_expressions = self.model(images)
                
                loss = self.criterion(fake_gene_expressions, gene_expression.float())
                loss.backward()
                self.optimizer.step()
                loss_list.append(loss.item())

        
                #if batch_idx % 10 == 0:
                #    print(f'Epoch [{epoch+1}/{max_epochs}], Step [{batch_idx+1}/{len(train_dataloader)}], Loss: {loss.item():.4f}')
                
            # Validation after each epoch
            val_corr, _, _ = self.test_model(val_dataloader)
            if val_corr > best_corr:
                best_corr = val_corr
            else:
                if self.classification_mode:
                    print(f'Epoch [{epoch+1}/{max_epochs}], Validation ROC AUC: {val_corr:.4f}, Best: {best_corr:.4f}')
                else:
                    print(f'Epoch [{epoch+1}/{max_epochs}], Validation Correlation: {val_corr:.4f}, Best: {best_corr:.4f}')
            if self.classification_mode:
                print(f'Epoch [{epoch+1}/{max_epochs}], Validation ROC AUC: {val_corr:.4f}')
            else:
                print(f'Epoch [{epoch+1}/{max_epochs}], Validation Correlation: {best_corr:.4f}')
        
        self.plot_smooth_loss(loss_list, 'MSE')
        if not self.simple_mlp:
            self.plot_smooth_loss(loss_slides_list, 'Slide Number')
            self.plot_smooth_loss(acc_slids, 'Discriminator ACC')
        print("Training complete.")
        return best_corr

    def train_maml_simple(self, train_dataset, val_dataset, test_dataloader , max_epochs=3, lr=1e-4, inner_lr=1e-3, weight_decay=1e-5, inner_steps=5,
                          num_of_steps=5000, test_every=500):
        """
        Model-Agnostic Meta-Learning (MAML) training loop for SimpleMLP
        """
        # Replace current model with SimpleMLP for this training
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        inner_loss_list = []
        meta_loss_list = []
        best_corr = 0
        
        
        self.model.train()
    

        for batch_idx in tqdm(range(num_of_steps)):
            slide_idx = random.choice(train_dataset.unique_slides)

            #query_slide_idx = slide_idx
            query_slide_idx = random.choice(val_dataset.unique_slides)
            query_batch = val_dataset.get_slide_batch(query_slide_idx)

            query_images, query_gene_exp, _ = query_batch
            query_images, query_gene_exp = query_images.to(DEVICE), query_gene_exp.to(DEVICE)

            fast_weights = {}
            for name, param in self.model.named_parameters():
                fast_weights[name] = param.clone()
            
            # Inner loop updates
            inner_loss_step_list = []
            for inner_step in range(inner_steps):
                # Forward pass with current fast weights
                batch = train_dataset.get_slide_batch(slide_idx)

                support_images, support_gene_exp, _ = batch
                support_images, support_gene_exp = support_images.to(DEVICE), support_gene_exp.to(DEVICE)
                fake_gene_expressions = self._forward_simple_mlp_with_weights(support_images, fast_weights)
                
                # Compute inner loss
                inner_loss = self.criterion(fake_gene_expressions, support_gene_exp.float())
                inner_loss_step_list.append(inner_loss.item())
                
                # Compute gradients
                grads = torch.autograd.grad(inner_loss, fast_weights.values(), create_graph=True)
                
                # Update fast weights
                for (name, param), grad in zip(fast_weights.items(), grads):
                    fast_weights[name] = param - inner_lr * grad
            
            # Outer loop: meta-update
            query_fake_gene_exp = self._forward_simple_mlp_with_weights(query_images, fast_weights)
            
            # Compute meta loss
            meta_loss = self.criterion(query_fake_gene_exp, query_gene_exp.float())
            
            # Update model parameters
            self.optimizer.zero_grad()
            meta_loss.backward()
            self.optimizer.step()

            inner_loss_list.append(np.mean(inner_loss_step_list))
            meta_loss_list.append(meta_loss.item())
               
            del fast_weights
        # Validation after each epoch
            if batch_idx % test_every == 0:
                val_corr, _, _ = self.test_model(test_dataloader)
                print("val_corr", val_corr)
                if val_corr > best_corr:
                    best_corr = val_corr
        
        
        self.plot_smooth_loss(inner_loss_list, 'MAML Simple MLP Gene Expression')
        self.plot_smooth_loss(meta_loss_list, 'MAML Simple MLP Meta')
        print("MAML Simple MLP Training complete.")
        return best_corr

    def _forward_simple_mlp_with_weights(self, x, weights):
        out = x
        model = self.model.model
        for name, layer in model.named_children():
            if isinstance(layer, nn.Linear):
                W = weights[f"model.{name}.weight"]
                b_key = f"model.{name}.bias"
                b = weights[b_key] if b_key in weights else None
                # Ensure dtype/device match (optional but safer)
                out = F.linear(out.to(W.dtype).to(W.device), W, b)
            elif isinstance(layer, nn.ReLU):
                # Avoid inplace to keep graph clean for higher-order grads
                out = F.relu(out, inplace=False)
            else:
                raise ValueError(f"Unsupported layer in meta-forward: {layer}")
        return out

    def _forward_with_weights(self, x, weights):
        """
        Forward pass using custom weights for MAML
        """
        # Encoder forward pass
        encoded = x
        encoder_layers = [
            (f'encoder.0.weight', f'encoder.0.bias'),
            (f'encoder.2.weight', f'encoder.2.bias'), 
            (f'encoder.4.weight', f'encoder.4.bias')
        ]
        
        for i, (weight_name, bias_name) in enumerate(encoder_layers):
            encoded = torch.nn.functional.linear(encoded, weights[weight_name], weights[bias_name])
            if i < len(encoder_layers) - 1:  # Apply ReLU except for last layer
                encoded = torch.nn.functional.relu(encoded)
        
        # Decoder forward pass
        decoded = encoded
        decoder_layers = [
            (f'decoder.0.weight', f'decoder.0.bias'),
            (f'decoder.2.weight', f'decoder.2.bias'),
            (f'decoder.4.weight', f'decoder.4.bias'),
            (f'decoder.6.weight', f'decoder.6.bias')
        ]
        
        for i, (weight_name, bias_name) in enumerate(decoder_layers):
            decoded = torch.nn.functional.linear(decoded, weights[weight_name], weights[bias_name])
            if i < len(decoder_layers) - 1:  # Apply ReLU except for last layer
                decoded = torch.nn.functional.relu(decoded)
        
        return decoded, encoded