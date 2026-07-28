"""
 Copyright (c) 2023, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""
import logging

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn import functional as F

from model.lavis.common.registry import registry
from model.lavis.models.base_model import concat_all_gather
from model.lavis.models.blip2_models.blip2 import (
    Blip2Base,
    compute_sim_matrix,
    disabled_train,
)
from model.lavis.models.blip_models.blip_outputs import BlipOutput, BlipOutputFeatures
from mhcac.loss import (
    ClassificationLoss,
    MultiPositiveContrastiveLoss,
    graph_connected_zero,
    soft_target_kl_loss,
    view_consistency_loss,
)
from mhcac.mhcac_12 import AbnormalityClassificationModel
from mhcac.view_fusion import ViewFusionModule
from smoke.sampling import negative_sampling_weights
from vision_encoders.pubmedclip.pubmed_clip import Pubmedclip
from vision_encoders.shared_visual_tokens import SharedVisualTokenProjector
from vision_encoders.swin.swin_encoder import SwinEncoder

# from vision_encoders.medclip.medclip import Medclip

# Common dimension every encoder is projected to before the merge.
VISUAL_DIM = 1408

chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

@registry.register_model("blip2")
@registry.register_model("blip2_feature_extractor")
class Blip2Qformer(Blip2Base):
    """
    BLIP2 first-stage model with Q-former and ViT.
    Supported model types:
        - pretrained: pretrained model with vit-g
        - pretrain_vitL: pretrained model with vit-large
        - coco: fintuned model on coco
    Usage:
        >>> from lavis.models import load_model
        >>> model = load_model("blip2", "pretrain")
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain": "configs/models/blip2/blip2_pretrain.yaml",
        "pretrain_vitL": "configs/models/blip2/blip2_pretrain_vitL.yaml",
        "coco": "configs/models/blip2/blip2_coco.yaml",
    }

    def __init__(
        self,
        vit_model="eva_clip_g",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        num_query_token=32,
        cross_attention_freq=2,
        embed_dim=256,
        max_txt_len=32,
        use_biovil=True,
        use_pubmedclip=True,
        use_swin=False,
        swin_model_name="ChayanM/SwinV2-GPT2_Mimic",
        swin_backend="hf",
        swin_pretrained=True,
        swin_frozen=True,
        swin_normalize=None,
        multi_view=False,
        view_fusion_cfg=None,
        lambda_mpc=0.0,
        lambda_view_consistency=0.0,
        lambda_itc=1.0,
        lambda_itm=1.0,
        lambda_lm=1.0,
        lambda_cls=1.0,
        lambda_teacher_cls=0.5,
        lambda_distill=0.5,
        lambda_mhcac_contrastive=0.1,
        lambda_orthogonality=0.05,
        lambda_sparsity=0.01,
        distill_temperature=2.0,
        mhcac_text_dropout=0.2,
        class_weights=None,
        cls_label_smoothing=0.05,
        itc_queue_size=1024,
    ):
        super().__init__()

        self.tokenizer = self.init_tokenizer()

        self.use_biovil = bool(use_biovil)
        self.use_pubmedclip = bool(use_pubmedclip)
        self.use_swin = bool(use_swin)
        if not any([self.use_biovil, self.use_pubmedclip, self.use_swin]):
            raise ValueError("At least one encoder must be enabled: biovil, pubmedclip, or swin")

        self.vit_model = vit_model
        vis_num_feat = 1408
        if self.use_biovil:
            self.visual_encoder, self.ln_vision, vis_num_feat = self.init_vision_encoder(
                "biovil", img_size, drop_path_rate, use_grad_checkpoint, vit_precision
            )
        else:
            self.visual_encoder = None
            self.ln_vision = nn.LayerNorm(vis_num_feat)

        if freeze_vit and self.visual_encoder is not None:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            logging.info("freeze vision encoder")
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, vis_num_feat, cross_attention_freq
        )
        self.Qformer.resize_token_embeddings(len(self.tokenizer))
        # transformers' meta fast-init breaks the BertLMPredictionHead bias:
        # resize_token_embeddings grows decoder.weight (tied to the word
        # embeddings) to the new vocab size but leaves cls.predictions.bias at
        # the old size and strands the aliased decoder.bias on the meta device.
        # Left alone this causes either "Cannot copy out of meta tensor" at
        # .to(device) or, once re-tied, an LM-head shape mismatch
        # ("expanded size 30523 must match 30522"). Rebuild the bias to match
        # decoder.weight (zeros for any newly added rows) and re-tie decoder.bias
        # to it, so both the meta crash and the size mismatch are fixed.
        _pred = getattr(getattr(self.Qformer, "cls", None), "predictions", None)
        if _pred is not None and getattr(_pred, "decoder", None) is not None:
            _vocab = _pred.decoder.weight.shape[0]
            _old = getattr(_pred, "bias", None)
            if _old is None or _old.is_meta or _old.shape[0] != _vocab:
                _new = torch.zeros(_vocab)
                if _old is not None and not _old.is_meta:
                    _n = min(_old.shape[0], _vocab)
                    _new[:_n] = _old.data[:_n]
                _new = nn.Parameter(_new)
                _pred.bias = _new
                _pred.decoder.bias = _new
        state_dict = self.Qformer.state_dict()
        for name, param in self.Qformer.named_parameters():
            if "_query" in name:
                key_orig = name.replace("_query", "")
                param.data.copy_(state_dict[key_orig])

        self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
        self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)

        self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        self.temp = nn.Parameter(0.07 * torch.ones([]))

        self.max_txt_len = max_txt_len
        self.lambda_itc = float(lambda_itc)
        self.lambda_itm = float(lambda_itm)
        self.lambda_lm = float(lambda_lm)
        self.lambda_cls = float(lambda_cls)
        self.lambda_teacher_cls = float(lambda_teacher_cls)
        self.lambda_distill = float(lambda_distill)
        self.lambda_mhcac_contrastive = float(lambda_mhcac_contrastive)
        self.lambda_orthogonality = float(lambda_orthogonality)
        self.lambda_sparsity = float(lambda_sparsity)
        self.distill_temperature = float(distill_temperature)
        self.itc_queue_size = int(itc_queue_size)
        if self.itc_queue_size < 0:
            raise ValueError("itc_queue_size must be non-negative")
        # A detached cross-batch queue makes ITC meaningful for the L4 recipe's
        # microbatch of two.  fp16 keeps a 1,024 x 32 x 256 image queue ~16 MB.
        self.register_buffer(
            "itc_image_queue",
            torch.zeros(
                self.itc_queue_size,
                num_query_token,
                embed_dim,
                dtype=torch.float16,
            ),
            persistent=False,
        )
        self.register_buffer(
            "itc_text_queue",
            torch.zeros(self.itc_queue_size, embed_dim, dtype=torch.float16),
            persistent=False,
        )
        self.register_buffer(
            "itc_queue_ptr", torch.zeros((), dtype=torch.long), persistent=False
        )
        self.register_buffer(
            "itc_queue_filled", torch.zeros((), dtype=torch.long), persistent=False
        )
        
        # Spatial/intensity augmentation is applied once in ReportDataset so
        # every encoder sees the same mildly transformed radiograph.
        self.pubmedclip = (
            # project=False: SharedVisualTokenProjector owns the 1408 projection.
            Pubmedclip(aug=None, project=False).eval() if self.use_pubmedclip else None
        )

        self.swin = (
            SwinEncoder(
                model_name=swin_model_name,
                pretrained=swin_pretrained,
                frozen=swin_frozen,
                backend=swin_backend,
                normalize=swin_normalize,
            ).eval()
            if self.use_swin
            else None
        )
        swin_dim = self.swin.embed_dim if self.use_swin else 768

        # self.medclip = Medclip().eval()

        # Multi-view fusion: one module per enabled encoder, operating on that
        # encoder's raw (pre-projection) output, so both the MHCAC branch and the
        # Q-Former concat are built from fused tokens. Zero-init makes this an
        # exact identity at step 0, hence regression-free against a single-view
        # checkpoint.
        self.multi_view = bool(multi_view)
        self.lambda_mpc = float(lambda_mpc)
        self.lambda_view_consistency = float(lambda_view_consistency)
        self.view_fusion = None
        self.mpc_loss_fn = None
        # Pre-fusion streams are only stashed when an auxiliary loss consumes
        # them; otherwise holding the references just wastes memory.
        self._keep_prefusion = bool(multi_view) and (
            self.lambda_mpc > 0 or self.lambda_view_consistency > 0
        )
        if self.multi_view:
            vf_cfg = dict(view_fusion_cfg or {})
            vf_cfg.pop("dim_source", None)
            stream_dims = {}
            if self.use_biovil:
                stream_dims["biovil"] = vis_num_feat
            if self.use_pubmedclip:
                stream_dims["pubmedclip"] = 768
            if self.use_swin:
                stream_dims["swin"] = swin_dim
            self.view_fusion = nn.ModuleDict({
                name: ViewFusionModule(dim=dim, **vf_cfg)
                for name, dim in stream_dims.items()
            })
            self.mpc_loss_fn = (
                MultiPositiveContrastiveLoss() if self.lambda_mpc > 0 else None
            )
            logging.info(f"multi-view fusion enabled for streams: {stream_dims}")

        # The single projection/merge point. Both MHCAC and the Q-Former read the
        # tokens it produces, so the two branches can no longer drift onto
        # different visual representations.
        shared_stream_dims = {}
        if self.use_biovil:
            # ln_vision already emits VISUAL_DIM, so this stream gets an Identity.
            shared_stream_dims["biovil"] = VISUAL_DIM
        if self.use_pubmedclip:
            shared_stream_dims["pubmedclip"] = 768
        if self.use_swin:
            shared_stream_dims["swin"] = swin_dim
        self.shared_visual_projector = SharedVisualTokenProjector(
            shared_stream_dims, visual_dim=VISUAL_DIM
        )

        self.mhcac = AbnormalityClassificationModel(
            embed_dim=768,
            num_abnormalities=14,
            num_classes=3,
            num_layers=6,
            num_commmon_tokens=14,
            initial_expert_tokens=None,
            visual_dim=VISUAL_DIM,
            text_dropout_rate=mhcac_text_dropout,
            use_cnn=self.use_biovil,
        )

        # sqrt(negative prevalence / class prevalence), capped at 10, computed
        # from the full study-level training cohort (negative, positive, uncertain).
        default_class_weights = [
            [1.0, 1.40, 0.0],
            [1.0, 5.56, 4.79],
            [1.0, 2.00, 5.47],
            [1.0, 1.85, 6.80],
            [1.0, 6.01, 10.0],
            [1.0, 2.67, 3.81],
            [1.0, 4.50, 7.17],
            [1.0, 3.44, 3.28],
            [1.0, 1.95, 4.14],
            [1.0, 4.61, 10.0],
            [1.0, 1.78, 5.45],
            [1.0, 10.0, 10.0],
            [1.0, 7.20, 10.0],
            [1.0, 1.57, 10.0],
        ]

        if class_weights is None:
            class_weights = default_class_weights
        elif len(class_weights) == 0:
            class_weights = None
        self.cls_loss_fn = ClassificationLoss(
            class_weights=class_weights,
            num_abnormalities=14,
            label_smoothing=cls_label_smoothing,
        )
        

    def _create_mask(self, embeddings, mask_ratio=0.1):
        num_patches = embeddings.size(1)
        num_masked = int(mask_ratio * num_patches)
        
        # Create a mask of ones, then set a subset to zero
        mask = torch.ones(num_patches, device=embeddings.device)
        mask[:num_masked] = 0
        mask = mask[torch.randperm(num_patches)]  # Shuffle to randomize masked positions
        
        # Expand mask to match embeddings' dimensions and apply
        mask = mask.unsqueeze(0).expand(embeddings.size(0), -1)
        mask = mask.unsqueeze(-1)  # Add dimension for broadcasting
        return embeddings * mask  # Apply mask by element-wise multiplication

    def _encode_aux_streams(self, aux_image, cached=None):
        """[B, N, 3, H, W] -> dict[name, [B, N, P, D]] of raw frozen-encoder output.

        Batched (one encoder call over B*N images, not a per-image loop) and
        under no_grad. That detaches the auxiliary *features* only -- the fusion
        module's W_K/W_V are applied outside this block and still get gradient.
        """
        cached = cached or {}
        B, N = aux_image.shape[:2] if aux_image is not None else (0, 0)
        streams = {}

        def unflatten(x):
            return x.reshape(B, N, *x.shape[1:])

        # Cached auxiliary features arrive already shaped [B, N, P, D].
        for name in ("biovil", "pubmedclip", "swin"):
            if name in cached:
                streams[name] = cached[name]

        need = [
            name for name, enabled in (
                ("biovil", self.use_biovil), ("pubmedclip", self.use_pubmedclip),
                ("swin", self.use_swin),
            ) if enabled and name not in streams
        ]
        if not need:
            return streams
        if aux_image is None:
            raise ValueError(
                f"aux_image is required to encode auxiliary streams {need}; "
                "the requested encoders are not covered by the feature cache."
            )

        flat = aux_image.flatten(0, 1)
        with torch.no_grad():
            if "biovil" in need:
                streams["biovil"] = unflatten(
                    self.visual_encoder(flat).projected_patch_embeddings.reshape(
                        flat.shape[0], -1, 1408
                    )
                )
            if "pubmedclip" in need:
                # Fuse the 768-dim ViT stream; the 1408 projection is recomputed
                # from the fused tokens, so the aux projection is discarded here.
                streams["pubmedclip"] = unflatten(
                    self.pubmedclip(flat, apply_aug=False)[0]
                )
            if "swin" in need:
                streams["swin"] = unflatten(self.swin(flat))
        return streams

    def _stash_prefusion(self, name, anchor, aux_streams):
        """Keep the pre-fusion tensors the auxiliary losses need, if any do."""
        if self._keep_prefusion:
            self._last_prefusion_streams[name] = (anchor, aux_streams.get(name))

    def _fuse(self, name, anchor, aux_streams, aux_mask, anchor_view_id, aux_view_ids):
        """Fuse one encoder stream with its auxiliary views, if multi-view is on."""
        if not self.multi_view or self.view_fusion is None:
            return anchor
        aux = aux_streams.get(name)
        if aux is None:
            # This micro-batch has NO auxiliary view on this rank, but a peer rank
            # may have one. The old `return anchor` skipped the fusion module here,
            # so its parameters were used on some ranks and skipped on others in
            # the SAME step. DDP then made those params grad-ready in a different
            # ORDER across ranks, and NCCL -- which matches collectives purely by
            # issue order (no tags) -- deadlocked at the first backward. Neither
            # find_unused_parameters=True nor the zero-weighted `ddp_param_touch`
            # in forward() fixes this: they equalise the *set* of used params, not
            # the reduction *order* (that is why gloo, which tags collectives, did
            # not hang while NCCL did). Instead, always run the module on a zero
            # auxiliary with an all-False mask, passing anchor_view_id through so
            # `view_emb` is exercised too. ViewFusionModule gates both residual
            # branches to zero for a study with no auxiliary view, so the output
            # equals `anchor` bit-for-bit (no scientific change) while every
            # view_fusion parameter enters the graph identically on every rank and
            # every step.
            batch, _, dim = anchor.shape
            aux = anchor.new_zeros(batch, 1, 1, dim)
            aux_mask = anchor.new_zeros(batch, 1, dtype=torch.bool)
            aux_view_ids = None
            if anchor_view_id is None:
                anchor_view_id = torch.zeros(
                    batch, dtype=torch.long, device=anchor.device
                )
        return self.view_fusion[name](
            anchor, aux.to(anchor.dtype), aux_mask, anchor_view_id, aux_view_ids
        )

    def _encode_image_streams(self, image, apply_aug=False, cached=None,
                              aux_image=None, aux_cached=None, aux_mask=None,
                              anchor_view_id=None, aux_view_ids=None):
        # ``cached`` holds raw frozen-encoder outputs (before ln_vision /
        # *_qformer_proj) precomputed by pretraining/precompute_features.py. When
        # present we skip the frozen encoder forward; the trainable projection
        # layers below still run so training is identical.
        cached = cached or {}
        # Raw (pre-merge) per-encoder outputs, keyed by stream name. The shared
        # projector is the only thing that turns these into visual tokens.
        raw_streams = {}
        self._last_prefusion_streams = {}

        # Multi-view: each stream is fused at its raw, pre-projection output, so
        # the trainable projections below run once on the fused [B, P, D] tensor
        # and both the MHCAC and Q-Former branches see fused tokens.
        # A batch where no study has an auxiliary view (N_max == 0) skips the
        # auxiliary encode entirely rather than running encoders on empty input.
        has_aux_input = (
            aux_image is not None and aux_image.shape[1] > 0
        ) or any(v is not None and v.shape[1] > 0 for v in (aux_cached or {}).values())
        fuse_on = self.multi_view and self.view_fusion is not None
        aux_streams = (
            self._encode_aux_streams(aux_image, cached=aux_cached)
            if fuse_on and has_aux_input
            else {}
        )

        if self.use_biovil:
            cnn_raw = cached["biovil"] if "biovil" in cached else (
                self.visual_encoder(image).projected_patch_embeddings.reshape(
                    image.shape[0], -1, VISUAL_DIM
                )
            )
            self._stash_prefusion("biovil", cnn_raw, aux_streams)
            cnn_raw = self._fuse("biovil", cnn_raw, aux_streams, aux_mask,
                                 anchor_view_id, aux_view_ids)
            # ln_vision normalises the frozen encoder output; it is not a
            # dimension projection, so it stays on the encoder side of the merge.
            raw_streams["biovil"] = self.ln_vision(cnn_raw)

        if self.use_pubmedclip:
            if "pubmedclip" in cached:
                vit_patches = cached["pubmedclip"]
            else:
                vit_patches, _ = self.pubmedclip(image, apply_aug=apply_aug)
            self._stash_prefusion("pubmedclip", vit_patches, aux_streams)
            vit_patches = self._fuse("pubmedclip", vit_patches, aux_streams,
                                     aux_mask, anchor_view_id, aux_view_ids)
            # The encoder's own ``mlp`` head is no longer used to reach VISUAL_DIM:
            # the shared projector owns that projection for every stream alike.
            raw_streams["pubmedclip"] = vit_patches

        if self.use_swin:
            swin_patches = cached["swin"] if "swin" in cached else self.swin(image)
            self._stash_prefusion("swin", swin_patches, aux_streams)
            raw_streams["swin"] = self._fuse(
                "swin", swin_patches, aux_streams, aux_mask, anchor_view_id, aux_view_ids
            )

        if not raw_streams:
            raise ValueError("No image encoder stream is enabled.")

        # One merge, one representation, consumed by both downstream branches.
        return self.shared_visual_projector(raw_streams)

    @staticmethod
    def _apply_encoder_keep(shared_visual, encoder_keep):
        """Zero disabled streams for post-training inference sensitivity only."""
        if encoder_keep is None:
            return shared_visual
        keep = {str(name).lower() for name in encoder_keep}
        known = set(shared_visual.spans)
        unknown = keep - known
        if unknown:
            raise ValueError(
                f"Unknown encoder_keep entries {sorted(unknown)}; expected {sorted(known)}"
            )
        if not keep:
            raise ValueError("encoder_keep must retain at least one encoder")
        return shared_visual.without(*(known - keep))
    
    def initialize_expert_tokens(self, chexpert_cols, embed_dim):
        # Initialize expert tokens based on text embeddings of abnormality names
        expert_embeddings = []
        for abnormality in chexpert_cols:
            # Get the text embedding (CLS token) for each abnormality
            text_tokens = self.tokenizer(
                abnormality,
                padding="max_length",
                truncation=True,
                max_length=20,  # Adjust max_length if necessary
                return_tensors="pt",
            ).to(next(self.parameters()).device)  # Move to device of the model
            
            text_output = self.Qformer.bert(
                text_tokens.input_ids,
                attention_mask=text_tokens.attention_mask,
                return_dict=True,
            )
            cls_embedding = text_output.last_hidden_state[:, 0, :]  # CLS token embedding
            expert_embeddings.append(cls_embedding)

        # Stack embeddings and return as initialized expert tokens
        embeddings = torch.cat(expert_embeddings, dim=0).reshape(len(chexpert_cols), embed_dim)
        torch.save(embeddings, "weights/expert_embeddings.pt")
        return embeddings
        
    
    @staticmethod
    def _batch_mask(samples, key, batch_size, device, fallback_key=None, default=True):
        value = samples.get(key)
        if value is None and fallback_key is not None:
            value = samples.get(fallback_key)
        if value is None:
            return torch.full((batch_size,), default, dtype=torch.bool, device=device)
        mask = torch.as_tensor(value, dtype=torch.bool, device=device)
        if mask.ndim == 0:
            mask = mask.expand(batch_size)
        mask = mask.reshape(-1)
        if mask.numel() != batch_size:
            raise ValueError(f"{key} must contain one boolean per batch item")
        return mask

    @staticmethod
    def _gather_with_local_grad(tensor):
        """Gather equal-sized DDP batches while preserving this rank's gradient.

        Remote features are constants on each rank; DDP later combines the
        parameter gradients produced where each feature was local.  This also
        avoids a backward collective deadlock when one rank has no valid report.
        """
        if not dist.is_available() or not dist.is_initialized():
            return tensor
        gathered = concat_all_gather(tensor.detach())
        batch_size = tensor.shape[0]
        start = dist.get_rank() * batch_size
        return torch.cat(
            [gathered[:start], tensor, gathered[start + batch_size :]], dim=0
        )

    @torch.no_grad()
    def _update_itc_queue(self, image_features, text_features, valid_mask):
        if self.itc_queue_size == 0 or not self.training:
            return
        images = concat_all_gather(image_features.detach())
        texts = concat_all_gather(text_features.detach())
        valid = concat_all_gather(valid_mask).bool()
        images = images[valid]
        texts = texts[valid]
        if images.shape[0] == 0:
            return
        if images.shape[0] >= self.itc_queue_size:
            images = images[-self.itc_queue_size :]
            texts = texts[-self.itc_queue_size :]

        count = images.shape[0]
        pointer = int(self.itc_queue_ptr.item())
        first = min(count, self.itc_queue_size - pointer)
        self.itc_image_queue[pointer : pointer + first].copy_(
            images[:first].to(self.itc_image_queue.dtype)
        )
        self.itc_text_queue[pointer : pointer + first].copy_(
            texts[:first].to(self.itc_text_queue.dtype)
        )
        remaining = count - first
        if remaining:
            self.itc_image_queue[:remaining].copy_(
                images[first:].to(self.itc_image_queue.dtype)
            )
            self.itc_text_queue[:remaining].copy_(
                texts[first:].to(self.itc_text_queue.dtype)
            )
        self.itc_queue_ptr.fill_((pointer + count) % self.itc_queue_size)
        self.itc_queue_filled.fill_(
            min(self.itc_queue_size, int(self.itc_queue_filled.item()) + count)
        )

    def _image_text_contrastive(self, image_features, text_features, valid_mask):
        image_features_all = self._gather_with_local_grad(image_features)
        text_features_all = self._gather_with_local_grad(text_features)
        valid_all = concat_all_gather(valid_mask)

        current_count = valid_all.numel()
        # The queue is a training-only source of negatives. Validation must not
        # depend on whichever train samples happened to fill the ring buffer at
        # the end of an epoch.
        queue_filled = int(self.itc_queue_filled.item()) if self.training else 0
        if queue_filled:
            # clone() decouples autograd's saved operands from the ring-buffer
            # update performed later in this same forward pass.
            image_features_all = torch.cat(
                [
                    image_features_all,
                    self.itc_image_queue[:queue_filled]
                    .to(image_features_all.dtype)
                    .clone(),
                ],
                dim=0,
            )
            text_features_all = torch.cat(
                [
                    text_features_all,
                    self.itc_text_queue[:queue_filled]
                    .to(text_features_all.dtype)
                    .clone(),
                ],
                dim=0,
            )
            candidate_valid = torch.cat(
                [
                    valid_all,
                    torch.ones(queue_filled, dtype=torch.bool, device=valid_all.device),
                ]
            )
        else:
            candidate_valid = valid_all

        # Floor is 0.01, CLIP's maximum logit scale of 100. The previous 1e-3
        # allowed a logit scale of 1000: ITC always pushes temp down to sharpen
        # the softmax, and by optimizer step ~1971 of the e123 run temp had
        # reached 5.19e-4, i.e. through the floor. Any batch holding a false
        # negative (MIMIC-CXR repeats reports, and the 1024-entry queue is very
        # likely to carry a duplicate of the positive) then produced an ITC loss
        # in the hundreds -- normalized features bound it at 2/T + ln(N), which
        # is ~36 at T=0.07 but ~2007 at T=1e-3. Bursts of 90-170 were observed.
        # clamp has zero gradient outside the range, so temp freezes on contact
        # with the floor rather than drifting further.
        temperature = self.temp.clamp(min=0.01, max=0.5)
        sim_i2t = torch.einsum(
            "bqd,nd->bnq", image_features, text_features_all
        ).amax(dim=-1) / temperature
        sim_t2i = torch.einsum(
            "bd,nqd->bnq", text_features, image_features_all
        ).amax(dim=-1) / temperature
        sim_i2t = sim_i2t.masked_fill(
            ~candidate_valid.unsqueeze(0), float("-inf")
        )
        sim_t2i = sim_t2i.masked_fill(
            ~candidate_valid.unsqueeze(0), float("-inf")
        )

        zero = graph_connected_zero(image_features)
        if not valid_mask.any():
            return (
                zero,
                sim_i2t[:, :current_count],
                sim_t2i[:, :current_count],
                valid_all,
            )
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        targets = rank * image_features.shape[0] + torch.arange(
            image_features.shape[0], device=image_features.device
        )
        loss_itc = 0.5 * (
            F.cross_entropy(sim_i2t[valid_mask], targets[valid_mask])
            + F.cross_entropy(sim_t2i[valid_mask], targets[valid_mask])
        )
        # ITM needs raw images/token ids, which the lightweight ITC queue does
        # not retain, so hard-negative mining uses the current global batch.
        return (
            loss_itc,
            sim_i2t[:, :current_count],
            sim_t2i[:, :current_count],
            valid_all,
        )

    def _image_text_matching(
        self,
        image_embeds,
        text_tokens,
        valid_mask,
        valid_all,
        sim_i2t,
        sim_t2i,
    ):
        zero = graph_connected_zero(image_embeds)
        # GLOBAL gate only. valid_all is gathered across ranks, so every rank
        # decides identically here. The old `not valid_mask.any()` term was
        # LOCAL: a rank whose own batch had no valid report returned here and
        # skipped the three all-gathers below while the peer ran them, leaving
        # the ranks one collective apart -> NCCL ALLGATHER timeout/deadlock.
        if valid_all.sum() < 2:
            return zero

        image_embeds_all = self._gather_with_local_grad(image_embeds)
        text_ids_all_ranks = concat_all_gather(text_tokens.input_ids)
        text_atts_all_ranks = concat_all_gather(text_tokens.attention_mask)
        # A rank whose local batch has no valid report still had to participate in
        # the gathers above. It must ALSO run itm_head + the Qformer cross-attention
        # below (on a dummy row) so those params are used on every rank -> identical
        # DDP grad-readiness order. The old `return zero` skipped them here while a
        # peer used them -> bucket-rebuild order diverged -> NCCL deadlock. The dummy
        # row's loss is zeroed out at the end.
        local_valid = bool(valid_mask.any())
        batch_size = image_embeds.shape[0]
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        positive_indices = rank * batch_size + torch.arange(
            batch_size, device=image_embeds.device
        )

        with torch.no_grad():
            rows = torch.arange(batch_size, device=image_embeds.device)
            candidate_mask = valid_all.unsqueeze(0).expand(batch_size, -1).clone()
            candidate_mask[rows, positive_indices] = False
            # Mask before softmax. Masking the positive *after* softmax lets an
            # overwhelmingly large self-logit underflow every negative to zero,
            # which makes torch.multinomial raise a CUDA device-side assert.
            weights_t2i = negative_sampling_weights(sim_t2i, candidate_mask)
            weights_i2t = negative_sampling_weights(sim_i2t, candidate_mask)

        # Fall back to a single dummy row when this rank has no valid report so the
        # ITM forward still runs (params used); its loss is masked to zero below.
        # valid_all.sum() >= 2 guarantees at least one eligible candidate; the
        # helper above also handles underflow and non-finite similarity scores.
        if local_valid:
            local_indices = valid_mask.nonzero(as_tuple=True)[0]
        else:
            local_indices = torch.zeros(
                1, dtype=torch.long, device=image_embeds.device
            )
        negative_image_indices = torch.stack(
            [torch.multinomial(weights_t2i[i], 1).squeeze(0) for i in local_indices]
        )
        negative_text_indices = torch.stack(
            [torch.multinomial(weights_i2t[i], 1).squeeze(0) for i in local_indices]
        )

        image_pos = image_embeds[local_indices]
        image_neg = image_embeds_all[negative_image_indices]
        text_ids_pos = text_tokens.input_ids[local_indices]
        text_atts_pos = text_tokens.attention_mask[local_indices]
        text_ids_neg = text_ids_all_ranks[negative_text_indices]
        text_atts_neg = text_atts_all_ranks[negative_text_indices]

        # positive, negative-image, negative-text triples
        text_ids = torch.cat([text_ids_pos, text_ids_pos, text_ids_neg], dim=0)
        text_atts = torch.cat([text_atts_pos, text_atts_pos, text_atts_neg], dim=0)
        image_inputs = torch.cat([image_pos, image_neg, image_pos], dim=0)
        query_tokens = self.query_tokens.expand(text_ids.shape[0], -1, -1)
        query_atts = torch.ones(
            query_tokens.shape[:-1], dtype=torch.long, device=image_embeds.device
        )
        output = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=torch.cat([query_atts, text_atts], dim=1),
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=torch.ones(
                image_inputs.shape[:-1], dtype=torch.long, device=image_embeds.device
            ),
            return_dict=True,
        )
        vl_embeddings = output.last_hidden_state[:, : query_tokens.shape[1]]
        logits = self.itm_head(vl_embeddings).mean(dim=1)
        num_positive = local_indices.numel()
        labels = torch.cat(
            [
                torch.ones(num_positive, dtype=torch.long, device=image_embeds.device),
                torch.zeros(2 * num_positive, dtype=torch.long, device=image_embeds.device),
            ]
        )
        loss_itm = F.cross_entropy(logits, labels)
        # Zero (but keep connected to itm_head/Qformer params) when this rank ran
        # only the dummy row, so it adds no gradient signal.
        if not local_valid:
            loss_itm = loss_itm * 0.0
        return loss_itm

    def _language_modeling(self, text_tokens, query_tokens, query_output, valid_mask):
        # Run the LM forward UNCONDITIONALLY so the Qformer LM head is used on every
        # rank, including one whose micro-batch has no valid report. A local
        # `if not valid_mask.any(): return zero` skipped these params on that rank
        # while a peer used them, so they became grad-ready at different points ->
        # DDP bucket-rebuild order diverged -> NCCL all-reduce deadlock. Invalid rows
        # are masked to -100 (no loss); the zero stays connected to output.loss (and
        # thus the LM-head params) so the branch enters the graph at the same
        # position on every rank.
        decoder_input_ids = text_tokens.input_ids.clone()
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
        labels = decoder_input_ids.masked_fill(
            decoder_input_ids == self.tokenizer.pad_token_id, -100
        )
        labels[~valid_mask] = -100
        query_atts = torch.ones(
            query_tokens.shape[:-1], dtype=torch.long, device=query_tokens.device
        )
        output = self.Qformer(
            decoder_input_ids,
            attention_mask=torch.cat([query_atts, text_tokens.attention_mask], dim=1),
            past_key_values=query_output.past_key_values,
            return_dict=True,
            labels=labels,
            reduction="none",
        )
        token_count = (labels[:, 1:] != -100).sum()
        # clamp_min(1) makes a no-valid-token rank return a graph-connected zero
        # (output.loss is all-zero for -100 targets) instead of skipping the head;
        # for token_count > 0 this is exactly the original mean.
        return output.loss.sum() / token_count.clamp_min(1)

    def forward(self, samples):
        image = samples.get("image")
        text = samples["text_output"]
        cached = {
            k: samples[f"{k}_feat"]
            for k in ("biovil", "pubmedclip", "swin")
            if f"{k}_feat" in samples
        }
        aux_cached = {
            k: samples[f"aux_{k}_feat"]
            for k in ("biovil", "pubmedclip", "swin")
            if f"aux_{k}_feat" in samples
        }
        shared_visual = self._encode_image_streams(
            image,
            apply_aug=False,
            cached=cached,
            aux_image=samples.get("aux_image"),
            aux_cached=aux_cached,
            aux_mask=samples.get("aux_mask"),
            anchor_view_id=samples.get("anchor_view_id"),
            aux_view_ids=samples.get("aux_view_ids"),
        )
        shared_visual = self._apply_encoder_keep(
            shared_visual, samples.get("encoder_keep")
        )
        image_embeds = shared_visual.tokens
        device = image_embeds.device
        batch_size = image_embeds.shape[0]
        classification_mask = self._batch_mask(
            samples,
            "classification_mask",
            batch_size,
            device,
            fallback_key="has_chexpert_label",
        )
        generation_mask = self._batch_mask(
            samples, "generation_mask", batch_size, device
        )

        image_atts = torch.ones(
            image_embeds.shape[:-1], dtype=torch.long, device=device
        )
        query_tokens = self.query_tokens.expand(batch_size, -1, -1)
        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_atts,
            use_cache=True,
            return_dict=True,
        )
        image_features = F.normalize(
            self.vision_proj(query_output.last_hidden_state), dim=-1
        )

        text_tokens = self.tokenizer(
            text,
            padding="max_length",
            truncation=True,
            max_length=self.max_txt_len,
            return_tensors="pt",
        ).to(device)
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        text_features = F.normalize(
            self.text_proj(text_output.last_hidden_state[:, 0]), dim=-1
        )

        loss_itc, sim_i2t, sim_t2i, generation_mask_all = (
            self._image_text_contrastive(
                image_features, text_features, generation_mask
            )
        )
        loss_itm = self._image_text_matching(
            image_embeds,
            text_tokens,
            generation_mask,
            generation_mask_all,
            sim_i2t,
            sim_t2i,
        )
        loss_lm = self._language_modeling(
            text_tokens, query_tokens, query_output, generation_mask
        )
        self._update_itc_queue(image_features, text_features, generation_mask)

        cls_labels = samples["classification_labels"]
        student_logits, _, contrastive_loss, orth_loss, sparsity_loss = self.mhcac(
            shared_visual,
            text_embeddings=None,
            labels=cls_labels,
            sample_mask=classification_mask,
        )
        cls_loss = self.cls_loss_fn(
            student_logits, cls_labels, sample_mask=classification_mask
        )

        # The teacher may read only a valid FINDINGS target.  The image-only
        # student remains the sole classification path exported at inference.
        teacher_mask = classification_mask & generation_mask
        loss_teacher_cls = student_logits.sum() * 0.0
        loss_distill = student_logits.sum() * 0.0
        # GLOBAL gate. `teacher_mask.any()` is a per-rank observation, but the
        # teacher MHCAC below uses the teacher-only text-attention params. A LOCAL
        # gate let a rank whose micro-batch had a valid FINDINGS run that forward
        # while a peer rank (none valid this step) skipped it, so those params
        # became grad-ready at different points across ranks -> DDP launches its
        # gradient all-reduce buckets in different orders -> NCCL deadlock. Same
        # class of bug already fixed for ITM via the global `valid_all` gate;
        # find_unused_parameters=True does not prevent it. All-reduce the flag so
        # every rank runs-or-skips the teacher branch together. The lambda check is
        # a static per-run constant (identical on every rank), so gating the
        # collective on it adds no desync and skips the all-reduce when the teacher
        # is disabled. On a rank that runs only because a peer had a valid target,
        # teacher_mask is locally all-False and ClassificationLoss returns a
        # graph-connected `logits.sum()*0.0`, so the teacher params still enter the
        # graph at the same position -> identical bucket order.
        teacher_active = False
        if self.lambda_teacher_cls > 0 or self.lambda_distill > 0:
            local_has_teacher = torch.as_tensor(
                float(teacher_mask.any()), device=device
            )
            if dist.is_available() and dist.is_initialized():
                dist.all_reduce(local_has_teacher, op=dist.ReduceOp.MAX)
            teacher_active = local_has_teacher.item() > 0
        if teacher_active:
            teacher_logits, _, _, _, _ = self.mhcac(
                shared_visual,
                text_embeddings=text_output.last_hidden_state,
                text_attention_mask=text_tokens.attention_mask,
                # Teacher supervision is applied by ClassificationLoss below;
                # do not duplicate the O(B^2) token contrastive calculation.
                labels=None,
            )
            loss_teacher_cls = self.cls_loss_fn(
                teacher_logits, cls_labels, sample_mask=teacher_mask
            )
            loss_distill = soft_target_kl_loss(
                student_logits,
                teacher_logits,
                sample_mask=teacher_mask,
                temperature=self.distill_temperature,
            )

        loss_mpc = cls_loss.new_zeros(())
        loss_view_consistency = cls_loss.new_zeros(())
        aux_mask = samples.get("aux_mask")
        if self.multi_view and aux_mask is not None and aux_mask.any():
            if self.mpc_loss_fn is not None:
                terms = [
                    self.mpc_loss_fn(anchor_raw, aux_raw, aux_mask)
                    for anchor_raw, aux_raw in self._last_prefusion_streams.values()
                    if aux_raw is not None
                ]
                if terms:
                    loss_mpc = torch.stack(terms).mean()
            if self.lambda_view_consistency > 0:
                pre = self._last_prefusion_streams
                anchor_raw_streams = {
                    name: (
                        self.ln_vision(pre[name][0]) if name == "biovil" else pre[name][0]
                    )
                    for name in self.shared_visual_projector.stream_names
                    if name in pre
                }
                anchor_shared = self.shared_visual_projector(anchor_raw_streams)
                anchor_logits, _, _, _, _ = self.mhcac(
                    anchor_shared,
                    text_embeddings=None,
                    labels=None,
                )
                loss_view_consistency = view_consistency_loss(
                    student_logits, anchor_logits, aux_mask.any(dim=1)
                )

        total_loss = (
            self.lambda_itc * loss_itc
            + self.lambda_itm * loss_itm
            + self.lambda_lm * loss_lm
            + self.lambda_cls * cls_loss
            + self.lambda_teacher_cls * loss_teacher_cls
            + self.lambda_distill * loss_distill
            + self.lambda_mhcac_contrastive * contrastive_loss
            + self.lambda_orthogonality * orth_loss
            + self.lambda_sparsity * sparsity_loss
            + self.lambda_mpc * loss_mpc
            + self.lambda_view_consistency * loss_view_consistency
        )
        # DDP determinism guard. ITM (`itm_head`) and the teacher MHCAC
        # text-attention run only on ranks whose local batch has a valid report,
        # so those params are used on some ranks and skipped on others in the
        # same step. That per-rank divergence in the used-parameter set makes DDP
        # launch its gradient all-reduce buckets in different orders across ranks
        # -> NCCL deadlock (find_unused_parameters=True does not prevent it).
        # A zero-weighted touch pulls EVERY trainable parameter into the graph on
        # every rank, so the used-parameter set — and thus the bucket order — is
        # identical everywhere. It adds no gradient signal (multiplied by 0).
        ddp_param_touch = sum(p.sum() for p in self.parameters() if p.requires_grad)
        total_loss = total_loss + 0.0 * ddp_param_touch
        return BlipOutput(
            loss=total_loss,
            loss_itc=loss_itc,
            loss_itm=loss_itm,
            loss_lm=loss_lm,
            loss_cls=cls_loss,
            loss_teacher_cls=loss_teacher_cls,
            loss_distill=loss_distill,
            loss_contrastive=contrastive_loss,
            loss_orthagonal=orth_loss,
            loss_sparsity=sparsity_loss,
            loss_mpc=loss_mpc,
            loss_view_consistency=loss_view_consistency,
            classification_logits=student_logits,
            classification_mask=classification_mask,
        )

    @torch.no_grad()
    def generate(
        self,
        samples,
        use_nucleus_sampling=False,
        num_beams=1,
        max_length=30,
        min_length=10,
        top_p=0.9,
        repetition_penalty=1.0,
    ):
        """
        Args:
            samples (dict): A dictionary containing the following keys:
                - image (torch.Tensor): A tensor of shape (batch_size, 3, H, W)
            use_nucleus_sampling (bool): Whether to use nucleus sampling. If False, use top-k sampling.
            num_beams (int): Number of beams for beam search. 1 means no beam search.
            max_length (int): The maximum length of the sequence to be generated.
            min_length (int): The minimum length of the sequence to be generated.
            top_p (float): The cumulative probability for nucleus sampling.
            repetition_penalty (float): The parameter for repetition penalty. 1.0 means no penalty.
            num_captions (int): Number of captions to be generated for each image.
        Returns:
            captions (list): A list of strings of length batch_size * num_captions.
        """
        image = samples.get("image")
        cached = {
            k: samples[f"{k}_feat"]
            for k in ("biovil", "pubmedclip", "swin")
            if f"{k}_feat" in samples
        }
        aux_cached = {
            k: samples[f"aux_{k}_feat"]
            for k in ("biovil", "pubmedclip", "swin")
            if f"aux_{k}_feat" in samples
        }
        shared_visual = self._encode_image_streams(
            image,
            apply_aug=False,
            cached=cached,
            aux_image=samples.get("aux_image"),
            aux_cached=aux_cached,
            aux_mask=samples.get("aux_mask"),
            anchor_view_id=samples.get("anchor_view_id"),
            aux_view_ids=samples.get("aux_view_ids"),
        )
        shared_visual = self._apply_encoder_keep(
            shared_visual, samples.get("encoder_keep")
        )
        image_embeds = shared_visual.tokens
        batch_size = image_embeds.shape[0]
        device = image_embeds.device

        if not use_nucleus_sampling:
            image_embeds = image_embeds.repeat_interleave(num_beams, dim=0)
        else:
            num_beams = 1
        image_atts = torch.ones(
            image_embeds.shape[:-1], dtype=torch.long, device=device
        )

        model_kwargs = {
            "encoder_hidden_states": image_embeds,
            "encoder_attention_mask": image_atts,
        }

        input_ids = (
            torch.LongTensor(batch_size, 1)
            .fill_(self.tokenizer.bos_token_id)
            .to(device)
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)

        outputs = self.Qformer.generate(
            input_ids=input_ids,
            query_embeds=query_tokens,
            max_length=max_length,
            min_length=min_length,
            num_beams=num_beams,
            do_sample=use_nucleus_sampling,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            eos_token_id=self.tokenizer.sep_token_id,
            pad_token_id=self.tokenizer.pad_token_id,
            **model_kwargs
        )
        captions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        return captions

    def forward_image(self, image, aux_image=None, aux_mask=None,
                      anchor_view_id=None, aux_view_ids=None):
        """Return image-only classification logits and learned Q-Former tokens.

        ``image`` may be a tensor (legacy API) or a complete samples dict.  The
        dict form supports the frozen-feature cache and study auxiliary view.
        Report text is intentionally ignored: this is the student/inference path.
        """
        cached = {}
        aux_cached = {}
        encoder_keep = None
        if isinstance(image, dict):
            samples = image
            encoder_keep = samples.get("encoder_keep")
            image = samples.get("image")
            aux_image = samples.get("aux_image", aux_image)
            aux_mask = samples.get("aux_mask", aux_mask)
            anchor_view_id = samples.get("anchor_view_id", anchor_view_id)
            aux_view_ids = samples.get("aux_view_ids", aux_view_ids)
            cached = {
                k: samples[f"{k}_feat"]
                for k in ("biovil", "pubmedclip", "swin")
                if f"{k}_feat" in samples
            }
            aux_cached = {
                k: samples[f"aux_{k}_feat"]
                for k in ("biovil", "pubmedclip", "swin")
                if f"aux_{k}_feat" in samples
            }
        shared_visual = self._encode_image_streams(
            image, apply_aug=False, cached=cached,
            aux_image=aux_image, aux_mask=aux_mask,
            aux_cached=aux_cached,
            anchor_view_id=anchor_view_id, aux_view_ids=aux_view_ids,
        )
        shared_visual = self._apply_encoder_keep(shared_visual, encoder_keep)

        concat_image_embeds = shared_visual.tokens

        classification_logits, attention, contrastive_loss, orth_loss, sparsity_loss = self.mhcac(
            shared_visual,
            text_embeddings=None,
            labels=None,
        )

        image_atts = torch.ones(
            concat_image_embeds.shape[:-1],
            dtype=torch.long,
            device=concat_image_embeds.device,
        )

        query_tokens = self.query_tokens.expand(concat_image_embeds.shape[0], -1, -1)

        query_output = self.Qformer.bert(
            query_embeds=query_tokens,
            encoder_hidden_states=concat_image_embeds,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
    
        # print("CNN patches shape:", image_embeds.shape)
        # print("VIT patches shape:", image_embeds_2.shape)
        # image_patches = self.image_embed_proj_norm(self.image_embed_proj(image_embeds))
        # txt_cls_token = text_output.last_hidden_state[:, 0, :]
        # print(f"query_output.last_hidden_state shape: {query_output.last_hidden_state.shape}")

        return classification_logits, query_output.last_hidden_state

    @torch.no_grad()
    def encode_visual_once(self, samples):
        """Encode and fuse all E123 streams once for sensitivity evaluation."""
        cached = {
            name: samples[f"{name}_feat"]
            for name in ("biovil", "pubmedclip", "swin")
            if f"{name}_feat" in samples
        }
        aux_cached = {
            name: samples[f"aux_{name}_feat"]
            for name in ("biovil", "pubmedclip", "swin")
            if f"aux_{name}_feat" in samples
        }
        return self._encode_image_streams(
            samples.get("image"),
            apply_aug=False,
            cached=cached,
            aux_image=samples.get("aux_image"),
            aux_cached=aux_cached,
            aux_mask=samples.get("aux_mask"),
            anchor_view_id=samples.get("anchor_view_id"),
            aux_view_ids=samples.get("aux_view_ids"),
        )

    @torch.no_grad()
    def classify_shared_visual(self, shared_visual, encoder_keep):
        """Classify a pre-encoded batch after a declared stream mask."""
        masked = self._apply_encoder_keep(shared_visual, encoder_keep)
        logits, _, _, _, _ = self.mhcac(
            masked, text_embeddings=None, labels=None
        )
        return logits


    def forward_text(self, text_tokens):
        text_output = self.Qformer.bert(
            text_tokens.input_ids,
            attention_mask=text_tokens.attention_mask,
            return_dict=True,
        )
        return text_output.last_hidden_state[:, 0, :]

    def compute_itm(self, image_inputs, text_ids, text_atts):
        image_atts = torch.ones(image_inputs.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        query_tokens = self.query_tokens.expand(image_inputs.shape[0], -1, -1)
        query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
            image_inputs.device
        )
        attention_mask = torch.cat([query_atts, text_atts], dim=1)
        output_itm = self.Qformer.bert(
            text_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=image_inputs,
            encoder_attention_mask=image_atts,
            return_dict=True,
        )
        vl_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
        itm_logit = self.itm_head(vl_embeddings)
        itm_logit = itm_logit[:, :, 1].mean(dim=1)
        return itm_logit

    @torch.no_grad()
    def extract_features(self, samples, mode="multimodal"):
        """
        Extract features for multimodal or unimodal samples.
        Args:
            samples (dict): A dictionary of samples, containing the following keys:
                - image (torch.Tensor): A tensor of shape (B, C, H, W) containing the image.
                    Raw images should be preprocessed before being passed to feature extractor.
                - text_input (list): A list of strings containing the text, length B.
            mode (str): The mode of feature extraction. Can be either "multimodal", "text" or "image".
                If "multimodal", return image features and multimodal features;
                if "text", return text features;
                if "image", return image features.
                Default: "multimodal".
        Returns:
            BlipOutputFeatures: A BlipOutputFeatures object containing the features.
                See lavis/models/blip_models/blip_outputs.py for more details.
        """
        image = samples.get("image")
        caption = samples.get("text_output")

        # assert mode is one of "image", "text", "multimodal"
        assert mode in [
            "image",
            "text",
            "multimodal",
        ], "mode must be one of 'image', 'text', 'multimodal'"

        # initalize output
        image_embeds, text_embeds, multimodal_embeds = None, None, None
        image_features, text_features = None, None

        if mode == "image":
            assert (
                image is not None
            ), "Image is not provided for mode 'image' or 'multimodal'"
            # return query features
            with self.maybe_autocast():
                image_embeds_frozen = self._encode_image_streams(image, apply_aug=False).tokens
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )

            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            image_embeds = query_output.last_hidden_state
            image_features = F.normalize(self.vision_proj(image_embeds), dim=-1)

        elif mode == "text":
            assert (
                caption is not None
            ), "text input is None for mode 'text' or 'multimodal'"

            # return text features
            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )

            text_output = self.Qformer.bert(
                text.input_ids,
                attention_mask=text.attention_mask,
                return_dict=True,
            )
            text_embeds = text_output.last_hidden_state
            text_features = self.text_proj(text_embeds)
            text_features = F.normalize(text_features, dim=-1)

        elif mode == "multimodal":
            # return multimodel query features
            with self.maybe_autocast():
                image_embeds_frozen = self._encode_image_streams(image, apply_aug=False).tokens
            image_embeds_frozen = image_embeds_frozen.float()
            image_atts = torch.ones(
                image_embeds_frozen.size()[:-1], dtype=torch.long
            ).to(self.device)
            query_tokens = self.query_tokens.expand(
                image_embeds_frozen.shape[0], -1, -1
            )
            query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(
                self.device
            )

            text = self.tokenizer(caption, return_tensors="pt", padding=True).to(
                self.device
            )
            attention_mask = torch.cat([query_atts, text.attention_mask], dim=1)

            output = self.Qformer.bert(
                text.input_ids,
                query_embeds=query_tokens,
                attention_mask=attention_mask,
                encoder_hidden_states=image_embeds_frozen,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            multimodal_embeds = output.last_hidden_state[:, : query_tokens.size(1), :]

        return BlipOutputFeatures(
            image_embeds=image_embeds,
            image_embeds_proj=image_features,
            text_embeds=text_embeds,
            text_embeds_proj=text_features,
            multimodal_embeds=multimodal_embeds,
        )

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        vit_model_cls = cfg.get("vit_model_cls", [])
        if isinstance(vit_model_cls, str):
            vit_model_cls = [vit_model_cls]
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        cross_attention_freq = cfg.get("cross_attention_freq", 2)

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)

        max_txt_len = cfg.get("max_txt_len", 32)
        encoders = cfg.get("encoders", {}) or {}

        def cfg_bool(value, default=False):
            if value is None:
                return default
            if isinstance(value, str):
                return value.strip().lower() in {"1", "true", "yes", "y", "on"}
            return bool(value)

        use_biovil = cfg_bool(
            encoders.get("biovil", cfg.get("use_biovil", vit_model == "biovil"))
        )
        use_pubmedclip = cfg_bool(
            encoders.get(
                "pubmedclip",
                cfg.get("use_pubmedclip", "pubmedclip" in list(vit_model_cls)),
            )
        )
        use_swin = cfg_bool(encoders.get("swin", cfg.get("use_swin", False)))
        swin_cfg = cfg.get("swin", {}) or {}
        swin_model_name = cfg.get(
            "swin_model_name",
            swin_cfg.get("model_name", "ChayanM/SwinV2-GPT2_Mimic"),
        )
        swin_backend = cfg.get("swin_backend", swin_cfg.get("backend", "hf"))
        swin_pretrained = cfg_bool(
            cfg.get("swin_pretrained", swin_cfg.get("pretrained", True)),
            default=True,
        )
        swin_frozen = cfg_bool(
            cfg.get("swin_frozen", swin_cfg.get("frozen", True)),
            default=True,
        )
        swin_normalize_raw = cfg.get("swin_normalize", swin_cfg.get("normalize", None))
        swin_normalize = (
            None
            if swin_normalize_raw is None
            else cfg_bool(swin_normalize_raw, default=True)
        )

        # Multi-view. from_config silently drops unknown config blocks, so these
        # keys must be read explicitly to take effect.
        multi_view = cfg_bool(cfg.get("multi_view", False))
        view_fusion_cfg_raw = cfg.get("view_fusion", {}) or {}
        view_fusion_cfg = {
            "num_heads": int(view_fusion_cfg_raw.get("num_heads", 8)),
            "ffn_ratio": int(view_fusion_cfg_raw.get("ffn_ratio", 4)),
            "num_blocks": int(view_fusion_cfg_raw.get("num_blocks", 1)),
            "num_view_types": int(view_fusion_cfg_raw.get("num_view_types", 4)),
            "dropout": float(view_fusion_cfg_raw.get("dropout", 0.1)),
            "p_view_drop": float(view_fusion_cfg_raw.get("p_view_drop", 0.2)),
        }
        loss_cfg = cfg.get("loss", {}) or {}
        lambda_mpc = float(loss_cfg.get("lambda_mpc", 0.0))
        lambda_view_consistency = float(loss_cfg.get("lambda_view_consistency", 0.0))
        lambda_itc = float(loss_cfg.get("lambda_itc", 1.0))
        lambda_itm = float(loss_cfg.get("lambda_itm", 1.0))
        lambda_lm = float(loss_cfg.get("lambda_lm", 1.0))
        lambda_cls = float(loss_cfg.get("lambda_cls", 1.0))
        lambda_teacher_cls = float(loss_cfg.get("lambda_teacher_cls", 0.5))
        lambda_distill = float(loss_cfg.get("lambda_distill", 0.5))
        lambda_mhcac_contrastive = float(
            loss_cfg.get("lambda_mhcac_contrastive", 0.1)
        )
        lambda_orthogonality = float(loss_cfg.get("lambda_orthogonality", 0.05))
        lambda_sparsity = float(loss_cfg.get("lambda_sparsity", 0.01))
        itc_queue_size = int(loss_cfg.get("itc_queue_size", 1024))

        mhcac_cfg = cfg.get("mhcac", {}) or {}
        distill_temperature = float(mhcac_cfg.get("distill_temperature", 2.0))
        mhcac_text_dropout = float(mhcac_cfg.get("text_dropout", 0.2))
        class_weights = mhcac_cfg.get("class_weights", None)
        if class_weights is not None:
            class_weights = [list(weights) for weights in class_weights]
        cls_label_smoothing = float(mhcac_cfg.get("label_smoothing", 0.05))

        model = cls(
            vit_model=vit_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            num_query_token=num_query_token,
            cross_attention_freq=cross_attention_freq,
            max_txt_len=max_txt_len,
            use_biovil=use_biovil,
            use_pubmedclip=use_pubmedclip,
            use_swin=use_swin,
            swin_model_name=swin_model_name,
            swin_backend=swin_backend,
            swin_pretrained=swin_pretrained,
            swin_frozen=swin_frozen,
            swin_normalize=swin_normalize,
            multi_view=multi_view,
            view_fusion_cfg=view_fusion_cfg,
            lambda_mpc=lambda_mpc,
            lambda_view_consistency=lambda_view_consistency,
            lambda_itc=lambda_itc,
            lambda_itm=lambda_itm,
            lambda_lm=lambda_lm,
            lambda_cls=lambda_cls,
            lambda_teacher_cls=lambda_teacher_cls,
            lambda_distill=lambda_distill,
            lambda_mhcac_contrastive=lambda_mhcac_contrastive,
            lambda_orthogonality=lambda_orthogonality,
            lambda_sparsity=lambda_sparsity,
            distill_temperature=distill_temperature,
            mhcac_text_dropout=mhcac_text_dropout,
            class_weights=class_weights,
            cls_label_smoothing=cls_label_smoothing,
            itc_queue_size=itc_queue_size,
        )
        model.load_checkpoint_from_config(cfg)

        return model

    def compute_sim_matrix(self, data_loader, task_cfg):
        """
        Compute similarity i2t, t2i matrix for the given data loader.
        """
        k_test = task_cfg.k_test

        return compute_sim_matrix(model=self, data_loader=data_loader, k_test=k_test)
