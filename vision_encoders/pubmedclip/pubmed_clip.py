import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

class Pubmedclip(nn.Module):
    def __init__(self, aug = None, device='cuda', project=True):
        """``project=False`` skips the 768->1408 MLP head.

        SharedVisualTokenProjector now owns every encoder's projection to the
        shared visual dimension, so callers that merge downstream must not build
        this head: it would be a trainable module that receives no gradient and
        costs a projection on every forward.
        """
        super(Pubmedclip, self).__init__()  # Initialize nn.Module
        # Load the pre-trained PubMedCLIP model and processor
        self.device = device
        self.aug = aug
        self.project = project
        self.model_name = "flaviagiammarino/pubmed-clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(self.model_name).to(self.device)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval()
        self.processor = CLIPProcessor.from_pretrained(self.model_name)
        
        # Define the MLP to project patch embeddings to 1408 dimensions
        self.mlp = (
            nn.Sequential(
                nn.Linear(768, 1024),  # Bottleneck layer: from 768 (input) to 1024 (hidden)
                nn.ReLU(inplace=True),
                nn.Linear(1024, 1408)  # Final projection to 1408 dimensions
            ).to(self.device)
            if project
            else None
        )

    def train(self, mode=True):
        super().train(mode)
        self.model.eval()
        return self

    def forward(self, image, apply_aug = True):
        # Input is already a [0,1] float tensor from dataset ToTensor(); skip
        # the processor's /255 rescale or it produces near-constant features.
        inputs = self.processor(images=image, return_tensors="pt", do_rescale=False)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}  # Move to device
        
        # inputs = inputs['pixel_values'].squeeze(0)
        inputs = inputs['pixel_values']

        if apply_aug and self.aug is not None:
             inputs = self.aug(inputs)
             
        # Obtain patch embeddings
        with torch.no_grad():
            vision_outputs = self.model.vision_model(pixel_values=inputs)
            patch_embeddings = vision_outputs.last_hidden_state
        
        if self.mlp is None:
            return patch_embeddings, None

        # Project the patch embeddings to 1408 dimensions using the MLP
        batch_size, num_patches, embedding_dim = patch_embeddings.shape  # Expected: (batch_size, num_patches, 768)
        patch_embeddings_clone = patch_embeddings.view(batch_size * num_patches, embedding_dim)  # Flatten for MLP
        projected_embeddings = self.mlp(patch_embeddings_clone)  # Project to 1408 dimensions
        projected_embeddings = projected_embeddings.view(batch_size, num_patches, -1)  # Reshape back to (batch_size, num_patches, 1408)

        return patch_embeddings, projected_embeddings
