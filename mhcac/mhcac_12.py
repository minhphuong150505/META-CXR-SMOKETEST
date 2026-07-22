import torch
import torch.nn as nn
import torch.nn.functional as F

from .loss import AbnormalitySpecificLoss, AttentionLoss

class DownsamplePatches(nn.Module):
    def __init__(self, input_patch_count, output_patch_count, embed_dim, method="conv"):
        """
        Downsamples patch embeddings to match the target patch count.
        
        Args:
            input_patch_count (int): Number of input patches (e.g., 196 for CNN).
            output_patch_count (int): Number of target patches (e.g., 49 for ViT).
            embed_dim (int): Dimensionality of the patch embeddings.
            method (str): Downsampling method ('conv' for convolution, 'pool' for pooling).
        """
        super(DownsamplePatches, self).__init__()
        self.input_patch_count = input_patch_count
        self.output_patch_count = output_patch_count
        self.embed_dim = embed_dim
        self.method = method
        
        # Define the method for downsampling
        if method == "conv":
            self.downsampler = nn.Conv2d(
                embed_dim, embed_dim, kernel_size=2, stride=2, padding=0
            )
        elif method == "pool":
            self.downsampler = None  # Will use F.adaptive_avg_pool2d
        else:
            raise ValueError("Unsupported method. Choose 'conv' or 'pool'.")
    
    def forward(self, patches):
        """
        Forward pass to downsample patches.
        
        Args:
            patches (Tensor): Patch embeddings of shape [batch_size, input_patch_count, embed_dim].
        
        Returns:
            Tensor: Downsampled patch embeddings of shape [batch_size, output_patch_count, embed_dim].
        """
        batch_size = patches.size(0)
        
        # Reshape patches into a grid (assumes square grid)
        grid_size = int(self.input_patch_count ** 0.5)  # e.g., 14x14
        target_size = int(self.output_patch_count ** 0.5)  # e.g., 7x7
        patches = patches.view(batch_size, grid_size, grid_size, self.embed_dim)
        patches = patches.permute(0, 3, 1, 2)  # Shape: [batch_size, embed_dim, grid_size, grid_size]
        
        if self.method == "conv":
            # Apply convolutional downsampling
            patches_downsampled = self.downsampler(patches)  # Shape: [batch_size, embed_dim, target_size, target_size]
        elif self.method == "pool":
            # Apply average pooling
            patches_downsampled = F.adaptive_avg_pool2d(patches, (target_size, target_size))
        
        # Reshape back to patch embedding format
        patches_downsampled = patches_downsampled.permute(0, 2, 3, 1)  # Shape: [batch_size, target_size, target_size, embed_dim]
        patches_downsampled = patches_downsampled.view(batch_size, self.output_patch_count, self.embed_dim)  # Flatten
        
        return patches_downsampled

# CrossModalEmbeddingAlignment to project image, text, and query embeddings into a common space
class CrossModalEmbeddingAlignment(nn.Module):
    """Aligns the shared visual tokens, report text and expert tokens to ``common_dim``.

    The image side is a **single** projection from ``visual_dim``. It used to hold
    one Linear per encoder, which meant MHCAC re-projected
    the raw encoder outputs itself and therefore trained on a different visual
    representation than META-Former. Encoders are now merged once, upstream, by
    ``SharedVisualTokenProjector``.
    """

    def __init__(
        self,
        common_dim,
        visual_dim=1408,
        txt_dim=768,
        expert_dim=768,
    ):
        super(CrossModalEmbeddingAlignment, self).__init__()
        self.visual_proj = nn.Linear(visual_dim, common_dim)
        self.text_proj = nn.Linear(txt_dim, common_dim)
        self.expert_proj = nn.Linear(expert_dim, common_dim)

        self.expert_norm = nn.LayerNorm(common_dim) # For expert tokens

    def forward(
        self,
        visual_tokens=None,
        text_embeddings=None,
        expert_tokens=None,
    ):
        # Project image and text embeddings
        visual_proj = F.normalize(self.visual_proj(visual_tokens), dim=-1) if visual_tokens is not None else None
        txt_proj = F.normalize(self.text_proj(text_embeddings), dim=-1) if text_embeddings is not None else None
        expert_proj = self.expert_norm(self.expert_proj(expert_tokens)) if expert_tokens is not None else None
        return visual_proj, txt_proj, expert_proj

# Define trainable positional encoding
class TrainablePositionalEncoding(nn.Module):
    def __init__(self, num_patches, embed_dim):
        super(TrainablePositionalEncoding, self).__init__()
        self.positional_encoding = nn.Parameter(torch.randn(1, num_patches, embed_dim))  # Learnable parameter

    def forward(self, x):
        return x + self.positional_encoding  # Add positional encoding to input
    
# ExpertTokenCrossAttention layer that performs both image and query cross-attention in a single pass
class ExpertTokenCrossAttention(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        num_abnormalities=14,
        dropout=0.1,
        text_dropout_rate=0.2,
        use_text_attention=True,
    ):
        super(ExpertTokenCrossAttention, self).__init__()
        
        # Expert-to-image cross-attention
        self.expert_to_image_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.expert_to_text_attention = (
            nn.MultiheadAttention(
                embed_dim, num_heads, dropout=dropout, batch_first=True
            )
            if use_text_attention
            else None
        )

        # Feed-forward networks for modality-specific features
        self.ffn_expert = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

        # Self-attention among expert tokens for knowledge sharing
        self.self_attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)

        # Layer normalization for residual connections
        self.norm_expert_text = (
            nn.LayerNorm(embed_dim) if use_text_attention else None
        )
        self.norm_self_attention = nn.LayerNorm(embed_dim)
        self.norm_expert_image = nn.LayerNorm(embed_dim)
        self.norm_ff = nn.LayerNorm(embed_dim)
        
        self.text_dropout_rate = text_dropout_rate

    def forward(
        self,
        expert_tokens,
        image_patches,
        text_embeddings=None,
        text_attention_mask=None,
    ):
        if text_embeddings is not None and self.expert_to_text_attention is not None:
            # Report text is privileged teacher information.  Whole-report
            # dropout prevents that branch from ignoring the image entirely.
            if self.training and self.text_dropout_rate > 0:
                keep_text = torch.rand(
                    text_embeddings.size(0), 1, 1, device=text_embeddings.device
                ) >= self.text_dropout_rate
                text_embeddings = text_embeddings * keep_text.to(text_embeddings.dtype)
            
            # Cross-attend image patches with text embeddings
            expert_text, _ = self.expert_to_text_attention(
                query=expert_tokens,
                key=text_embeddings,
                value=text_embeddings,
                # PyTorch MHA expects True for positions to ignore.  Without
                # this, the privileged-text teacher attended BERT padding and
                # learnt a length-dependent shortcut instead of report content.
                key_padding_mask=(
                    ~text_attention_mask.to(dtype=torch.bool)
                    if text_attention_mask is not None
                    else None
                ),
            )
            expert_text = self.norm_expert_text(expert_text + expert_tokens)  # Residual connection
        
        else:
            # In inference mode or when text_embeddings is unavailable, rely on image patches alone
            expert_text = expert_tokens
        
        # Self-attention among expert tokens
        expert_refined, _ = self.self_attention(
            query=expert_text, key=expert_text, value=expert_text
        )
        expert_refined = self.norm_self_attention(expert_text + expert_refined)  # Residual connection and normalization
        
        # Expert-to-ViT cross-attention
        expert_image, attention_weights = self.expert_to_image_attention(query = expert_refined, key = image_patches, value = image_patches)
        expert_image = self.norm_expert_image(expert_image + expert_refined)  # Residual connection
        expert_image = self.norm_ff(self.ffn_expert(expert_image) + expert_image)  # Feed-forward layer

        return expert_image, attention_weights


# Main Abnormality Classification Model
class AbnormalityClassificationModel(nn.Module):
    def __init__(
        self,
        embed_dim=768,
        num_heads=8,
        num_abnormalities=14,
        num_classes=3,
        num_layers=2,
        num_commmon_tokens=8,
        dropout=0.2,
        initial_expert_tokens=None,
        visual_dim=1408,
        txt_dim=768,
        target_patch_count=49,
        text_dropout_rate=0.2,
        num_text_teacher_layers=2,
        use_cnn=True,
    ):
        super(AbnormalityClassificationModel, self).__init__()

        self.embed_dim = embed_dim
        self.visual_dim = visual_dim
        self.num_abnormalities = num_abnormalities
        self.num_layers = num_layers
        self.target_patch_count = target_patch_count
        if not 0 <= num_text_teacher_layers <= num_layers:
            raise ValueError("num_text_teacher_layers must be between 0 and num_layers")
        # Initial projection layer to align image, text, and query embeddings.
        # One image projection, from the shared visual dimension.
        self.embedding_alignment = CrossModalEmbeddingAlignment(
            embed_dim,
            visual_dim=visual_dim,
            txt_dim=txt_dim,
            # Expert tokens are allocated at embed_dim below, so their projection
            # must read embed_dim -- not the 768 default, which only happened to
            # match because the shipped config also uses embed_dim=768.
            expert_dim=embed_dim,
        )

        if initial_expert_tokens is not None:
            self.expert_tokens = nn.Parameter(initial_expert_tokens)
        else:
            self.expert_tokens = nn.Parameter(torch.randn(num_commmon_tokens, embed_dim))
            nn.init.xavier_uniform_(self.expert_tokens)

        # Stack multiple ExpertTokenCrossAttention layers
        self.attention_layers = nn.ModuleList(
            [
                ExpertTokenCrossAttention(
                    embed_dim,
                    num_heads,
                    num_abnormalities,
                    dropout,
                    text_dropout_rate=text_dropout_rate,
                    use_text_attention=layer_idx < num_text_teacher_layers,
                )
                for layer_idx in range(num_layers)
            ]
        )

        # Classification heads for each expert token
        # self.classifiers = nn.ModuleList([
        #     nn.Sequential(
        #         nn.Linear(embed_dim * (num_commmon_tokens + 1), embed_dim * 2),  # Expand feature space
        #         nn.ReLU(),  # Non-linearity
        #         nn.Dropout(0.2),  # Regularization
        #         nn.Linear(embed_dim * 2, num_classes)  # Final classification layer
        #     )
        #     for _ in range(num_abnormalities)
        # ])
        
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),  # Expand feature space
                nn.ReLU(),  # Non-linearity
                nn.Dropout(0.2),  # Regularization
                nn.Linear(embed_dim * 2, num_classes)  # Final classification layer
            )
            for _ in range(num_abnormalities)
        ])

        self.expert_loss = AbnormalitySpecificLoss(
            temperature=0.05,
            margin=0.5,
            d_embedding=embed_dim,
            num_abnormalities=num_abnormalities,
        )
        # self.attention_loss = AttentionLoss(lambda_sparsity=0.3)
        
        self.expert_token_norm = nn.LayerNorm(embed_dim)
        
        # self.w_cnn = nn.Parameter(torch.tensor(1.0))
        # self.w_vit = nn.Parameter(torch.tensor(1.0))
        
        self.pos_enc = TrainablePositionalEncoding(num_patches=target_patch_count, embed_dim=embed_dim)
        # Runs on the biovil span *after* the shared projection, hence embed_dim.
        self.cnn_downsampler = (
            DownsamplePatches(196, target_patch_count, embed_dim, method="conv")
            if use_cnn
            else None
        )
        
        if self.cnn_downsampler is not None and isinstance(
            self.cnn_downsampler.downsampler, nn.Conv2d
        ):
            nn.init.xavier_uniform_(self.cnn_downsampler.downsampler.weight)
            nn.init.constant_(self.cnn_downsampler.downsampler.bias, 0)

    def _resize_patch_sequence(self, patches):
        if patches is None:
            return None

        # Drop a CLS token when the remaining tokens form the actual patch grid.
        num_tokens = patches.size(1)
        if num_tokens == self.target_patch_count + 1:
            patches = patches[:, 1:, :]
        else:
            without_cls = num_tokens - 1
            full_grid = int(num_tokens ** 0.5)
            patch_grid = int(without_cls ** 0.5)
            if (
                without_cls > 0
                and full_grid * full_grid != num_tokens
                and patch_grid * patch_grid == without_cls
            ):
                patches = patches[:, 1:, :]

        if patches.size(1) == self.target_patch_count:
            return patches

        grid_size = int(patches.size(1) ** 0.5)
        target_size = int(self.target_patch_count ** 0.5)
        if grid_size * grid_size == patches.size(1) and target_size * target_size == self.target_patch_count:
            bsz, _, dim = patches.shape
            patches = patches.view(bsz, grid_size, grid_size, dim).permute(0, 3, 1, 2)
            patches = F.adaptive_avg_pool2d(patches, (target_size, target_size))
            return patches.permute(0, 2, 3, 1).reshape(bsz, self.target_patch_count, dim)

        return F.interpolate(
            patches.transpose(1, 2),
            size=self.target_patch_count,
            mode="linear",
            align_corners=False,
        ).transpose(1, 2)

    def forward(
        self,
        shared_visual_tokens,
        text_embeddings=None,
        text_attention_mask=None,
        labels=None,
        sample_mask=None,
    ):
        """Classify from the shared visual tokens produced upstream.

        ``shared_visual_tokens`` is a ``SharedVisualTokens``: one ``[B, N, visual_dim]``
        tensor plus the span each encoder occupies. Spans are used only to give each
        encoder its own within-stream positional encoding and its own resize to
        ``target_patch_count``; the projection to ``embed_dim`` happens once, on the
        merged tensor, so MHCAC and META-Former share one visual representation.
        """
        tokens = shared_visual_tokens.tokens
        spans = shared_visual_tokens.spans
        if tokens.ndim != 3:
            raise ValueError(
                f"shared_visual_tokens must be [B, N, D]; got {tuple(tokens.shape)}"
            )
        if tokens.shape[-1] != self.visual_dim:
            raise ValueError(
                f"shared_visual_tokens dim {tokens.shape[-1]} != MHCAC visual_dim {self.visual_dim}"
            )
        if not spans:
            raise ValueError("shared_visual_tokens carries no encoder spans")
        batch_size = tokens.size(0)

        # Expand expert tokens to match batch size
        expert_tokens = self.expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)  # [B, N_expert, D]

        # One projection for every encoder, applied to the merged sequence.
        visual_proj, txt_proj, expert_tokens = self.embedding_alignment(
            visual_tokens=tokens,
            text_embeddings=text_embeddings,
            expert_tokens=expert_tokens,
        )

        # Slice the *projected shared* sequence per encoder. This is a view of the
        # shared representation, not a second encoding path.
        image_streams = []
        for name, span in sorted(spans.items(), key=lambda item: item[1].start):
            stream = visual_proj[:, span, :]
            if name == "biovil" and self.cnn_downsampler is not None:
                # Learned 196 -> target_patch_count reduction, kept from the
                # original CNN path. It now runs at embed_dim, after the shared
                # projection, so it no longer constitutes a separate projection.
                stream = self.cnn_downsampler(stream)
            else:
                stream = self._resize_patch_sequence(stream)
            image_streams.append(self.pos_enc(stream))
        if not image_streams:
            raise ValueError("No image stream was provided to MHCAC.")
        image_patches = torch.cat(image_streams, dim=1)

        # Pass through multiple attention layers
        attention_weights_list = []
        for i, layer in enumerate(self.attention_layers):
            if layer.expert_to_text_attention is not None:
                expert_tokens, attention_weights = layer(
                    expert_tokens,
                    image_patches,
                    txt_proj,
                    text_attention_mask=text_attention_mask,
                )
            elif i == self.num_layers - 2: #last before layer
                normalized_expert_tokens = self.expert_token_norm(self.expert_tokens)
                # Add normalized initial expert tokens back
                expert_tokens = expert_tokens + normalized_expert_tokens.unsqueeze(0).expand(batch_size, -1, -1)
                expert_tokens, attention_weights = layer(expert_tokens, image_patches, text_embeddings = None)
            else:
                expert_tokens, attention_weights = layer(expert_tokens, image_patches, text_embeddings = None)
                
            attention_weights_list.append(attention_weights)

        pooled_representations, orth_loss, contrastive_loss, sparsity_loss = self.expert_loss(
            expert_tokens,
            attention_weights_list,
            labels,
            sample_mask=sample_mask,
        )
        
        # Classification for each abnormality
        logits = []
        for i in range(len(self.classifiers)):
            logits.append(self.classifiers[i](pooled_representations[:, i, :]))  # Shape: [batch_size, num_classes] for each abnormality
            # combined_features = torch.cat([expert_tokens.flatten(1), pooled_representations[:, i, :]], dim=1)  # Shape: [batch_size, embed_dim * (num_tokens + 1)]
            # logits.append(self.classifiers[i](combined_features))
        logits = torch.stack(logits, dim=1)  # Shape: [batch_size, num_abnormalities, num_classes]
        
            
        return logits, attention_weights_list, contrastive_loss, orth_loss, sparsity_loss
        
