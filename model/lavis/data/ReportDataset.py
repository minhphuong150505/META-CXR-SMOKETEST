import dataclasses
import json
import os
import re
from time import time
from enum import Enum, auto
from pathlib import Path
from typing import List, Any

from local_config import (
    CHEXPERT_CSV,
    PROCESSED_TRAIN_CSV,
    PROCESSED_VAL_CSV,
    PROCESSED_TEST_CSV,
)

os.environ['GRADIO_TEMP_DIR'] = os.path.join(os.getcwd(), "gradio_tmp")

import numpy as np
import pandas as pd
import torch
from PIL import Image
from omegaconf import OmegaConf
from skimage import io
from torchvision import transforms
from torchvision.transforms import Compose, Resize, ToTensor, CenterCrop

from model.lavis.processors import BaseProcessor
from model.lavis.common.registry import registry
from model.lavis.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from model.lavis.datasets.datasets.base_dataset import BaseDataset
from model.lavis.datasets.datasets.caption_datasets import __DisplMixin
from model.lavis.data.mimic_cxr_utils import (
    UNKNOWN_VIEW_ID,
    VIEW_ID_MAP,
    build_study_index,
    view_id,
)

@registry.register_processor("my_blip_caption")
class MyBlipCaptionProcessor(BaseProcessor):
    def __init__(self, prompt="", max_words=50):
        self.prompt = prompt
        self.max_words = max_words

    def __call__(self, caption):
        caption = self.prompt + self.pre_caption(caption)

        return caption

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)

    def pre_caption(self, caption):
        caption = re.sub(
            r"([!\"()*#;~])",
            " ",
            caption,
        )
        caption = re.sub(
            r"\s{2,}",
            " ",
            caption,
        )
        caption = caption.rstrip("\n")
        caption = caption.strip(" ")

        # truncate caption
        caption_words = caption.split(" ")
        if len(caption_words) > self.max_words:
            caption = " ".join(caption_words[: self.max_words])

        return caption


class ExpandChannels:
    """
    Transforms an image with one channel to an image with three channels by copying
    pixel intensities of the image along the 1st dimension.
    """

    def __call__(self, data: torch.Tensor) -> torch.Tensor:
        """
        :param data: Tensor of shape [1, H, W].
        :return: Tensor with channel copied three times, shape [3, H, W].
        """
        if data.shape[0] != 1:
            raise ValueError(f"Expected input of shape [1, H, W], found {data.shape}")
        return torch.repeat_interleave(data, 3, dim=0)


def create_chest_xray_transform_for_inference(resize: int, center_crop_size: int) -> Compose:
    """
    Defines the image transformation pipeline for Chest-Xray datasets.

    :param resize: The size to resize the image to. Linear resampling is used.
                   Resizing is applied on the axis with smaller shape.
    :param center_crop_size: The size to center crop the image to. Square crop is applied.
    """

    transforms = [Resize(resize), CenterCrop(center_crop_size), ToTensor(), ExpandChannels()]
    return Compose(transforms)

class SeparatorStyle(Enum):
    """Different separator style."""
    SINGLE = auto()
    TWO = auto()


@dataclasses.dataclass
class Conversation:
    """A class that keeps all conversation history."""
    system: str
    roles: List[str]
    messages: List[List[str]]
    offset: int
    sep_style: SeparatorStyle = SeparatorStyle.SINGLE
    sep: str = "###"
    sep2: str = None

    # Used for gradio server
    skip_next: bool = False
    conv_id: Any = None

    def get_prompt(self):
        if self.sep_style == SeparatorStyle.SINGLE:
            ret = self.system
            for role, message in self.messages:
                if message:
                    ret += self.sep + " " + role + ": " + message
                else:
                    ret += self.sep + " " + role + ":"
            return ret
        elif self.sep_style == SeparatorStyle.TWO:
            seps = [self.sep, self.sep2]
            ret = self.system + seps[0]
            for i, (role, message) in enumerate(self.messages):
                if message:
                    ret += role + ": " + message + seps[i % 2]
                else:
                    ret += role + ":"
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def append_message(self, role, message):
        self.messages.append([role, message])

    def to_gradio_chatbot(self):
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset:]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def copy(self):
        return Conversation(
            system=self.system,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)

    def dict(self):
        return {
            "system": self.system,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
            "sep": self.sep,
            "sep2": self.sep2,
            "conv_id": self.conv_id,
        }

class MyReportProcessor():
    def __init__(self, prompt="", max_words=50, prompt_neg=""):
        self.prompt = prompt
        self.max_words = max_words
        self.prompt_neg = prompt_neg

    def __call__(self, findings, no_labels=False):
        prompt = self.prompt

        if no_labels:
            findings = "no common findings"  # cannot write which findings as we don't no them
        prompt = prompt.format(findings=findings)

        return prompt

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        prompt = cfg.get("prompt", "")
        max_words = cfg.get("max_words", 50)

        return cls(prompt=prompt, max_words=max_words)


class MIMIC_CXR_Dataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, split, cfg, ann_paths=[], truncate=None):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Preprocessed CSVs contain one image row with a FINDINGS-only target,
        # its provenance/validity, and a CheXpert-label validity flag.  Empty
        # targets are intentionally retained for classification/distillation
        # masking rather than being silently dropped here.
        # Produced by preporcessing/preprocess_mimic_cxr.py over the full p10-p19 set.
        # NOTE: as of 2026-07-20 the splits are NOT view-filtered — they keep every
        # ViewPosition (AP/PA/LATERAL/LL/UNKNOWN) so multi-view fusion can use them.
        # Filter on ViewPosition here if a run needs frontal-only.
        self.cur_split = split
        csv_map = {
            "train": PROCESSED_TRAIN_CSV,
            "val":   PROCESSED_VAL_CSV,
            "test":  PROCESSED_TEST_CSV,
        }
        if split not in csv_map:
            raise ValueError(f"Unknown split '{split}' (expected train/val/test)")
        self.reports = pd.read_csv(csv_map[split])
        required_report_cols = {
            "subject_id", "study_id", "dicom_id", "image_path", "findings_clean",
            "extraction_method", "target_valid", "has_chexpert_label",
        }
        missing_report_cols = sorted(required_report_cols - set(self.reports.columns))
        if missing_report_cols:
            raise ValueError(
                f"Processed {split} CSV is missing columns: {missing_report_cols}. "
                "Rebuild it with preporcessing/preprocess_mimic_cxr.py."
            )
        self.reports["subject_id"] = self.reports["subject_id"].astype(int)
        self.reports["study_id"] = self.reports["study_id"].astype(int)
        if self.reports["dicom_id"].isna().any():
            raise ValueError(f"Processed {split} CSV has empty dicom_id values.")
        self.reports["dicom_id"] = self.reports["dicom_id"].astype(str)
        invalid_path = (
            self.reports["image_path"].isna()
            | self.reports["image_path"].astype(str).str.strip().eq("")
        )
        if invalid_path.any():
            raise ValueError(f"Processed {split} CSV has {int(invalid_path.sum())} empty image paths.")
        self.reports["image_path"] = self.reports["image_path"].astype(str)
        if self.reports["dicom_id"].duplicated().any():
            raise ValueError(f"Processed {split} CSV contains duplicate dicom_id rows.")
        self.reports["findings_clean"] = self.reports["findings_clean"].fillna("")
        self.reports = self.reports.rename(columns={"findings_clean": "findings"})
        self.reports["target_valid"] = self._coerce_bool(
            self.reports["target_valid"], "target_valid"
        ) & self.reports["findings"].str.strip().ne("")
        study_key = ["subject_id", "study_id"]
        inconsistent_targets = (
            self.reports.groupby(study_key, sort=False)["findings"].nunique(dropna=False) > 1
        )
        if inconsistent_targets.any():
            raise ValueError(
                f"Processed {split} CSV has {int(inconsistent_targets.sum())} studies "
                "with inconsistent FINDINGS targets across views."
            )

        self.use_pred_labels = True

        self.chexpert = pd.read_csv(CHEXPERT_CSV)

        self.chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum",
                              "Cardiomegaly", "Lung Opacity",
                              "Lung Lesion", "Edema",
                              "Consolidation", "Pneumonia",
                              "Atelectasis", "Pneumothorax",
                              "Pleural Effusion", "Pleural Other",
                              "Fracture", "Support Devices"]

        required_chexpert_cols = {"subject_id", "study_id", *self.chexpert_cols}
        missing_chexpert_cols = sorted(required_chexpert_cols - set(self.chexpert.columns))
        if missing_chexpert_cols:
            raise ValueError(f"CheXpert CSV is missing columns: {missing_chexpert_cols}")
        self.chexpert["subject_id"] = self.chexpert["subject_id"].astype(int)
        self.chexpert["study_id"] = self.chexpert["study_id"].astype(int)
        label_key = study_key
        if self.chexpert.duplicated(label_key).any():
            raise ValueError(
                "CheXpert CSV is not unique by (subject_id, study_id); refusing "
                "a many-to-many label merge."
            )
        self.chexpert["_has_chexpert_label_raw"] = (
            self.chexpert[self.chexpert_cols].notna().any(axis=1)
        )
        # CE uses 0=negative, 1=positive, 2=uncertain. Zeros in an entirely
        # unlabelled row are placeholders and are excluded by classification_mask.
        self.chexpert[self.chexpert_cols] = (
            self.chexpert[self.chexpert_cols].replace(-1, 2).fillna(0).astype("int8")
        )

        print(f"Number of chexpert records: {len(self.chexpert)}")

        # A runner epoch now means exactly one pass over studies. The historical
        # two custom half-epochs evaluated twice and made scheduler semantics
        # depend on the dataset implementation.
        self.custom_epochs_per_epoch = 1
        self.current_custom_epoch = 0
        self.vit_model = cfg.model_cfg['vit_model']
        self.vit_model_cls = cfg.model_cfg['vit_model_cls']
        processor_key = "train" if split == "train" else "eval"
        processor_cfg = cfg.datasets_cfg.mimic_cxr.vis_processor[processor_key]
        self.img_size = int(processor_cfg.get("image_size", 448))
        resize_size = int(processor_cfg.get("resize_size", round(self.img_size * 512 / 448)))
        aug_cfg = processor_cfg.get("augmentation", {}) or {}
        augmentation_enabled = split == "train" and bool(aug_cfg.get("enabled", True))
        if augmentation_enabled and cfg.run_cfg.get("feature_cache_dir", None):
            raise ValueError(
                "feature_cache_dir contains deterministic frozen-encoder features, "
                "so it cannot be combined with train image augmentation. Disable "
                "datasets.mimic_cxr.vis_processor.train.augmentation.enabled or "
                "train without the cache."
            )
        image_ops = [Resize(resize_size), CenterCrop(self.img_size)]
        if augmentation_enabled:
            degrees = float(aug_cfg.get("degrees", 5.0))
            translate = float(aug_cfg.get("translate", 0.02))
            scale_delta = float(aug_cfg.get("scale_delta", 0.05))
            affine_p = float(aug_cfg.get("affine_p", 0.5))
            jitter_p = float(aug_cfg.get("jitter_p", 0.5))
            image_ops.extend([
                transforms.RandomApply([
                    transforms.RandomAffine(
                        degrees=degrees,
                        translate=(translate, translate),
                        scale=(1.0 - scale_delta, 1.0 + scale_delta),
                    )
                ], p=affine_p),
                transforms.RandomApply([
                    transforms.ColorJitter(
                        brightness=float(aug_cfg.get("brightness", 0.1)),
                        contrast=float(aug_cfg.get("contrast", 0.1)),
                    )
                ], p=jitter_p),
            ])
        image_ops.extend([ToTensor(), ExpandChannels()])
        self.general_trans = transforms.Compose(image_ops)

        if self.vit_model == 'biovil':
            self.vis_transforms = create_chest_xray_transform_for_inference(512, center_crop_size=self.img_size)

        self.annotation = self.reports.copy()
        self.annotation["findings"] = self.annotation["findings"].str.replace(
            "\n", " ", regex=False
        )
        has_processed_label_flag = "has_chexpert_label" in self.annotation
        if has_processed_label_flag:
            self.annotation["_has_chexpert_label_processed"] = self._coerce_bool(
                self.annotation["has_chexpert_label"], "has_chexpert_label"
            )
        labels = self.chexpert[
            label_key + self.chexpert_cols + ["_has_chexpert_label_raw"]
        ]
        self.annotation = self.annotation.merge(
            labels,
            how="left",
            on=label_key,
            validate="many_to_one",
            indicator="_chexpert_merge",
        )
        raw_has_label = self.annotation["_has_chexpert_label_raw"].fillna(False).astype(bool)
        if has_processed_label_flag:
            preprocessed_has_label = self.annotation["_has_chexpert_label_processed"]
            inconsistent = preprocessed_has_label & ~raw_has_label
            if inconsistent.any():
                raise ValueError(
                    f"{int(inconsistent.sum())} processed rows claim CheXpert labels "
                    "but no non-null source labels were found."
                )
            self.annotation["classification_valid"] = preprocessed_has_label & raw_has_label
        else:
            self.annotation["classification_valid"] = raw_has_label
        self.annotation[self.chexpert_cols] = (
            self.annotation[self.chexpert_cols].fillna(0).astype("int8")
        )
        self.annotation = self.annotation.drop(
            columns=[
                "_has_chexpert_label_raw",
                "_chexpert_merge",
                *(["_has_chexpert_label_processed"] if has_processed_label_flag else []),
            ]
        ).reset_index(drop=True)

        self.img_ids = {
            dicom_id: index for index, dicom_id in enumerate(self.annotation["dicom_id"])
        }
        self.id_to_dicom = {value: key for key, value in self.img_ids.items()}
        print(f"Number of image annotation records: {len(self.annotation)}")
        
        add_findings_in_prompt = cfg.run_cfg.get("add_findings_in_prompt", False)
        self.prompt = cfg.datasets_cfg.mimic_cxr.text_processor.train.prompt if split == 'train' \
            else cfg.datasets_cfg.mimic_cxr.text_processor.eval.prompt

        self.text_processor = MyReportProcessor(
            prompt=self.prompt, max_words=1000)

        # Optional precomputed frozen-encoder feature cache.
        self._init_feature_cache(cfg)

        # One training/evaluation sample per study. ``multi_view`` controls
        # whether its one complementary view is returned, not whether image rows
        # are allowed to duplicate the report target.
        self._init_study_index(cfg, truncate=truncate)
        self.evaluator = MIMICEvalCap(self.annotation, self.img_ids)

    VIEW_ID_MAP = VIEW_ID_MAP
    UNKNOWN_VIEW_ID = UNKNOWN_VIEW_ID

    @staticmethod
    def _coerce_bool(series, name):
        """Read bool columns safely (``astype(bool)`` treats "False" as true)."""
        if pd.api.types.is_bool_dtype(series):
            return series.fillna(False).astype(bool)
        mapping = {
            True: True, False: False, 1: True, 0: False,
            "true": True, "false": False, "1": True, "0": False,
        }
        normalised = series.map(
            lambda value: value.strip().lower() if isinstance(value, str) else value
        )
        result = normalised.map(mapping)
        invalid = result.isna() & series.notna()
        if invalid.any():
            values = sorted({str(value) for value in series[invalid].head(5)})
            raise ValueError(f"Invalid boolean values in {name}: {values}")
        return result.fillna(False).astype(bool)

    def _view_id(self, view_position):
        return view_id(view_position)

    def _init_study_index(self, cfg, truncate=None):
        """Group rows by ``(subject_id, study_id)`` into anchor + auxiliary views.

        Every study keeps exactly one anchor row, and all sample keys that
        existed before (``text_output``, ``image_id``, ``classification_labels``,
        ``dicom_id``, ``image_path``, ``image``) continue to come from that
        anchor row, so ``MIMICEvalCap`` keeps working untouched.

        Degrades gracefully when the CSV has no ``ViewPosition`` column: every
        view maps to ``unknown`` and the anchor falls back to row order.
        """
        self.multi_view = bool(cfg.model_cfg.get("multi_view", False))
        data_cfg = cfg.model_cfg.get("data", {}) or {}
        self.study_sampling = bool(data_cfg.get("study_sampling", True))
        self.max_aux_views = int(data_cfg.get("max_aux_views", 1)) if self.multi_view else 0
        anchor_priority = list(data_cfg.get("anchor_priority", ["PA", "AP", "lateral"]))
        rows = self.annotation[
            ["subject_id", "study_id"]
            + (["ViewPosition"] if "ViewPosition" in self.annotation else [])
        ].to_dict("records")
        if self.study_sampling:
            self.studies = build_study_index(
                rows,
                anchor_priority=anchor_priority,
                max_aux_views=self.max_aux_views,
            )
        else:
            # Explicit escape hatch for building a feature cache over every
            # DICOM. Full-data train/eval configs should keep study_sampling=true.
            self.studies = [
                {
                    "study_key": (row["subject_id"], row["study_id"]),
                    "anchor": position,
                    "aux": [],
                    "anchor_view_id": view_id(row.get("ViewPosition")),
                    "aux_view_ids": [],
                }
                for position, row in enumerate(rows)
            ]
        if truncate is not None:
            self.studies = self.studies[:int(truncate)]

        n_multi = sum(1 for s in self.studies if s["aux"])
        sample_unit = "studies" if self.study_sampling else "image rows"
        print(f"[{self.cur_split}] {len(self.studies)} {sample_unit} from "
              f"{len(self.annotation)} image rows, {n_multi} with a complementary "
              f"view (multi_view={self.multi_view}, max_aux_views={self.max_aux_views})")

    def _init_feature_cache(self, cfg):
        """Open per-encoder feature memmaps if ``run.feature_cache_dir`` is set.

        Cache layout (see pretraining/precompute_features.py):
            <dir>/<encoder>/<split>_feats.npy   (memmap (N, P, D) float16)
            <dir>/<encoder>/<split>_ids.json    (list[dicom_id] in row order)
        When active, ``__getitem__`` returns the cached raw features instead of
        decoding the JPG, and the model skips the frozen encoder forward.
        """
        self.feature_cache = None
        cache_dir = cfg.run_cfg.get("feature_cache_dir", None)
        if not cache_dir:
            return
        encoders = cfg.model_cfg.get("encoders", {})
        # PubMedCLIP raw patch tokens are [P, 768].  Keeping them in the same
        # cache protocol avoids decoding/running its frozen ViT on every step.
        wanted = [
            encoder for encoder in ("biovil", "pubmedclip", "swin")
            if encoders.get(encoder, False)
        ]
        cache = {}
        for enc in wanted:
            feats_path = os.path.join(cache_dir, enc, f"{self.cur_split}_feats.npy")
            ids_path = os.path.join(cache_dir, enc, f"{self.cur_split}_ids.json")
            if not (os.path.exists(feats_path) and os.path.exists(ids_path)):
                raise FileNotFoundError(
                    f"feature_cache_dir set but cache missing for encoder '{enc}', "
                    f"split '{self.cur_split}': {feats_path}"
                )
            with open(ids_path) as f:
                row_ids = [str(dicom_id) for dicom_id in json.load(f)]
            if len(row_ids) != len(set(row_ids)):
                raise ValueError(f"Duplicate dicom_id values in feature cache: {ids_path}")
            feats = np.load(feats_path, mmap_mode="r")
            if len(feats) != len(row_ids):
                raise ValueError(
                    f"Feature/id length mismatch for {enc} {self.cur_split}: "
                    f"{len(feats)} != {len(row_ids)}"
                )
            cache[enc] = {
                "feats": feats,
                "row": {dicom: i for i, dicom in enumerate(row_ids)},
            }
        self.feature_cache = cache
        print(f"[{self.cur_split}] feature cache active for {wanted} from {cache_dir}")

    def set_custom_epoch(self, custom_epoch):
        self.current_custom_epoch = custom_epoch

    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        value_range = array.max()
        if value_range == 0:
            return np.zeros_like(array, dtype=np.uint8)
        array /= value_range
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")

        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")


    def _row_visual(self, ann):
        """Visual input for one CSV row: decoded image, or cached raw features."""
        # Canonical full-data CSVs store a relative ``files/p1X/...`` path. Keep
        # support for the old Kaggle marker, but reject arbitrary absolute/path-
        # traversal inputs instead of silently escaping ``vis_root``.
        raw = ann["image_path"].replace("\\", "/")
        marker = "/mimic-cxr-jpg-lite/"
        if marker in raw:
            rel = raw.split(marker, 1)[1]
        else:
            if raw.startswith("/") or re.match(r"^[A-Za-z]:/", raw):
                raise ValueError(
                    f"image_path must be relative to vis_root, got: {raw}"
                )
            rel = raw
        rel = os.path.normpath(rel)
        if rel == ".." or rel.startswith(f"..{os.sep}"):
            raise ValueError(f"image_path escapes vis_root: {raw}")
        image_path = os.path.join(self.vis_root, rel)

        out = {"image_path": str(image_path)}
        if self.feature_cache is None:
            out["image"] = self.general_trans(self.load_image(Path(image_path)))
        else:
            for enc, store in self.feature_cache.items():
                dicom_id = str(ann["dicom_id"])
                if dicom_id not in store["row"]:
                    raise KeyError(
                        f"DICOM {dicom_id} is absent from the {enc} feature cache "
                        f"for split {self.cur_split}. Build caches with "
                        "model.data.study_sampling=false so auxiliary views are included."
                    )
                row = store["row"][dicom_id]
                out[f"{enc}_feat"] = torch.from_numpy(
                    np.ascontiguousarray(store["feats"][row])
                ).float()
        return out

    def __getitem__(self, index):
        study = self.studies[index]
        ann = self.annotation.iloc[study["anchor"]]

        anchor_visual = self._row_visual(ann)
        image_path = anchor_visual.pop("image_path")

        # if self.cur_split == 'train':
        #     if self.vit_model == "biovil":  # old version worked with smaller img and without biovil img processing
        #         image_biovil = self.vis_transforms(image)
        #         if self.vis_augs is not None:
        #             image_biovil = self.vis_augs(image_biovil)

        #         # print(f"image_biovil shape is {image_biovil.shape}")
                
        #     for model in self.vit_model_cls:
        #         if model == 'pubmedclip':
        #             image_pubmed = self.general_trans(image)
        #             image_pubmed = self.pubmed_processor(images=image_pubmed, return_tensors="pt")
        #             image_pubmed = image_pubmed['pixel_values'].squeeze(0)
        #             if self.vis_augs is not None:
        #                 image_pubmed = self.vis_augs(image_pubmed)
                
                
                # print(f"image pubmed shape is {image_pubmed.shape}")
                 
                 
        caption = ann["findings"].strip()
        # input_text = self.text_processor(findings=None)

        # conv = Conversation(
        #     system="A chat between a curious user and an artificial intelligence assistant acting as an experienced radiologist. "
        #            "The assistant gives professional, detailed, and polite answers to the user's questions.",
        #     roles=["USER", "ASSISTANT"],
        #     messages=[],
        #     offset=0,
        #     sep_style=SeparatorStyle.TWO,
        #     sep=" ",
        #     sep2="</s>",
        # )
        # conv.append_message(conv.roles[0], input_text)
        # conv.append_message(conv.roles[1], None)
        # prompt = conv.get_prompt()

        # if "<IMG>" in prompt:
        #     before_img, after_img = prompt.split("<IMG>")
        #     prompt = (before_img, after_img)


        # Get the CheXpert classification labels (e.g., 0 for negative, 1 for positive, -1 for uncertain)
        chexpert_labels = ann[self.chexpert_cols].values.astype(float)
        # print(torch.tensor(chexpert_labels, dtype=torch.long))
        
        end_time = time()
        # print(f"__getitem__ took {end_time - start_time:.4f} seconds")
        
        sample = {
            # "text_input": prompt,
            "text_output": caption,
            "image_id": self.img_ids[ann["dicom_id"]],
            "classification_labels": torch.tensor(chexpert_labels, dtype=torch.long),  # Convert to tensor
            "classification_mask": torch.tensor(
                bool(ann["classification_valid"]), dtype=torch.bool
            ),
            "generation_mask": torch.tensor(bool(ann["target_valid"]), dtype=torch.bool),
            "dicom_id": ann["dicom_id"],
            "image_path": str(image_path)
        }
        sample.update(anchor_visual)  # "image", or the cached "<enc>_feat" tensors

        if self.multi_view:
            aux_visuals = [
                self._row_visual(self.annotation.iloc[p]) for p in study["aux"]
            ]
            sample["anchor_view_id"] = study["anchor_view_id"]
            sample["aux_view_ids"] = list(study["aux_view_ids"])
            if self.feature_cache is None:
                sample["aux_image"] = [a["image"] for a in aux_visuals]
            else:
                for enc in self.feature_cache:
                    sample[f"aux_{enc}_feat"] = [a[f"{enc}_feat"] for a in aux_visuals]
        return sample

    def __len__(self):
        return len(self.studies)

    def collater(self, samples):
        """Pad ragged auxiliary-view counts to the batch's N_max.

        Pre-existing keys are delegated to the default collate untouched, so a
        ``multi_view=False`` batch is byte-identical to the original.
        """
        if not self.multi_view:
            return super().collater(samples)

        aux_keys = [
            k for k in samples[0]
            if k == "aux_image" or (k.startswith("aux_") and k.endswith("_feat"))
        ]
        skip = set(aux_keys) | {"aux_view_ids"}
        batch = super().collater(
            [{k: v for k, v in s.items() if k not in skip} for s in samples]
        )

        B = len(samples)
        n_max = max(len(s["aux_view_ids"]) for s in samples)
        aux_mask = torch.zeros(B, n_max, dtype=torch.bool)
        aux_view_ids = torch.full(
            (B, n_max), self.UNKNOWN_VIEW_ID, dtype=torch.long
        )
        for i, s in enumerate(samples):
            n = len(s["aux_view_ids"])
            if n:
                aux_mask[i, :n] = True
                aux_view_ids[i, :n] = torch.tensor(s["aux_view_ids"], dtype=torch.long)
        batch["aux_mask"] = aux_mask
        batch["aux_view_ids"] = aux_view_ids

        for key in aux_keys:
            anchor_key = "image" if key == "aux_image" else key[len("aux_"):]
            template = samples[0][anchor_key]
            if n_max == 0:
                batch[key] = torch.zeros(
                    (B, 0) + tuple(template.shape), dtype=template.dtype
                )
                continue
            rows = []
            for s in samples:
                items = list(s[key])
                items += [torch.zeros_like(s[anchor_key])] * (n_max - len(items))
                rows.append(torch.stack(items, dim=0))
            batch[key] = torch.stack(rows, dim=0)
        return batch


@registry.register_builder("mimic_cxr")
class MIMIC_CXR_Builder(BaseDatasetBuilder):
    train_dataset_cls = MIMIC_CXR_Dataset
    eval_dataset_cls = MIMIC_CXR_Dataset

    DATASET_CONFIG_DICT = {
        "default": "defaults_report.yaml"
    }


class MIMICEvalCap:
    def __init__(self, gts, img_id_map):

        self.gts = gts

        # invert img_id_map
        self.dicom_to_id = img_id_map
        self.id_to_dicom = {v: k for k, v in img_id_map.items()}

        # METEOR starts a Java process. Construct scorers lazily so every train
        # dataset/rank does not pay that cost when no caption evaluation occurs.
        self.scorers = None


    def preprocess(self, s):
        s = s.replace('\n', '')
        s = s.replace('<s>', '')
        s = s.replace('</s>', '')
        return s

    def evaluate(self, res):

        # Caption metrics are optional and are not part of the Stage-1
        # classification smoke path. Import them only if this method is called;
        # this avoids installing NLTK, Java/METEOR and pycocoevalcap on Kaggle.
        try:
            from nltk import word_tokenize
            from pycocoevalcap.bleu.bleu import Bleu
            from pycocoevalcap.meteor.meteor import Meteor
            from pycocoevalcap.rouge.rouge import Rouge
        except ImportError as exc:
            raise RuntimeError(
                "Caption evaluation requires the optional 'eval-generation' "
                "dependencies; Stage-1 classification does not."
            ) from exc

        res = {self.id_to_dicom[elem["image_id"]]: elem["caption"] for elem in res}
        valid_dicom_ids = set(
            self.gts.loc[self.gts["target_valid"].astype(bool), "dicom_id"].astype(str)
        ) if "target_valid" in self.gts else set(self.gts["dicom_id"].astype(str))
        # Invalid/absent FINDINGS targets are kept for classification but must
        # not enter language metrics.
        res = {str(dicom_id): caption for dicom_id, caption in res.items()
               if str(dicom_id) in valid_dicom_ids}
        res_keys_set = set(res)
        gts = {}
        gts_img_id = {}
        for _, elem in self.gts.iterrows():
            dicom_id = str(elem["dicom_id"])
            if dicom_id in res_keys_set:
                gts[dicom_id] = [elem["findings"]]
                gts_img_id[self.dicom_to_id[dicom_id]] = [elem["findings"]]

        # gts = {elem["dicom_id"]: [elem["findings"]] for _, elem in self.gts.iterrows() if elem["dicom_id"] in res.keys()}
        # gts_img_id = {self.dicom_to_id[elem["findings"]]: [elem["Note"]] for _, elem in self.gts.iterrows() if elem["dicom_id"] in res.keys()}
        assert res.keys() == gts.keys()
        # =================================================
        # Pre-process sentences
        # =================================================
        print('tokenization...')
        for dicom in res.keys():
            pred_text = ' '.join(word_tokenize(self.preprocess(res[dicom]))).lower()
            true_text = ' '.join(word_tokenize(self.preprocess(gts[dicom][0]))).lower()

            res[dicom] = [pred_text]
            gts[dicom] = [true_text]

        # =================================================
        # Compute scores
        # =================================================
        if self.scorers is None:
            print('setting up scorers...')
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L")
            ]

        final_scores = {}
        for scorer, method in self.scorers:
            print('computing %s score...' % (scorer.method()))
            score, scores = scorer.compute_score(gts, res)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    final_scores[m] = sc
                    #final_scores["elem_wise_" + str(m)] = scs
                    print("%s: %0.3f" % (m, sc))
            else:
                print("%s: %0.3f" % (method, score))
                #final_scores["elem_wise_" + str(method)] = scores
                final_scores[method] = score

        final_scores['agg_metrics'] = np.mean(list({k: v for k, v in final_scores.items() if "elem_wise" not in k}.values()))

        return final_scores, gts_img_id

class CheXpertDataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, split, cfg, ann_paths = [], truncate=None):
        """
        vis_root (string): Root directory of images
        ann_path (string): Path to the CheXpert annotation file
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

        # Load annotation file
        self.cur_split = split
        self.annotations = pd.read_csv("/workspace/CheXpert-v1.0-small/valid.csv")
        self.custom_epochs_per_epoch = 1
        self.current_custom_epoch = 0
        
        # Filter only frontal images with AP/PA views
        self.annotations = self.annotations[(self.annotations['Frontal/Lateral'] == 'Frontal')]

        # Define the CheXpert label columns
        self.chexpert_cols = ["No Finding", "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", 
                              "Lung Lesion", "Edema", "Consolidation", "Pneumonia", "Atelectasis", 
                              "Pneumothorax", "Pleural Effusion", "Pleural Other", "Fracture", "Support Devices"]

        # Handle uncertain labels (-1) by replacing with 2 (for cross-entropy loss compatibility)
        # self.annotations[self.chexpert_cols] = self.annotations[self.chexpert_cols].replace(-1, 2)
        
        for column in self.chexpert_cols:
            self.annotations[column].fillna(0.0, inplace=True)

        # Define transformations
        self.img_size = cfg.datasets_cfg.mimic_cxr.vis_processor.train.image_size
        self.general_trans = transforms.Compose([Resize((512, 512)), CenterCrop(448), ToTensor(), ExpandChannels()])
        self.img_ids = {path: i for i, path in enumerate(self.annotations['Path'])}
        if truncate is not None:
            self.annotations = self.annotations[:truncate]

        print(f"Number of annotation records: {len(self.annotations)}")
        
    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        array /= array.max()
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")

        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")

    def set_custom_epoch(self, custom_epoch):
        self.current_custom_epoch = custom_epoch

    def __getitem__(self, index):
        start_time = time()
        subset_size = len(self.annotations) // self.custom_epochs_per_epoch
        start_index = self.current_custom_epoch * subset_size
        actual_index = start_index + index

        ann = self.annotations.iloc[actual_index]

        image_path = os.path.join(self.vis_root, ann["Path"])
        image = self.load_image(Path(image_path))
        image = self.general_trans(image)
        
        
        chexpert_labels = ann[self.chexpert_cols].values.astype(float)
        
        end_time = time()
        return {
            "image": image,
            "image_id": self.img_ids[ann["Path"]],
            "classification_labels": torch.tensor(chexpert_labels, dtype=torch.long),
            "image_path": str(image_path)
        }
    
    def __len__(self):
        return len(self.annotations)
    

class IU_Xray_Dataset(BaseDataset, __DisplMixin):
    def __init__(self, vis_processor, text_processor, vis_root, split, cfg, ann_paths=[], truncate=None):
        """
        Args:
            vis_processor: Vision processor
            text_processor: Text processor  
            vis_root (string): Root directory with all the images and reports
            split (string): 'train', 'val', or 'test'
            cfg: Configuration
            ann_paths: Annotation paths
            truncate: Optional truncation
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)
        
        # Load reports
        self.reports_df = pd.read_csv(os.path.join(vis_root, 'indiana_reports.csv'))
        
        # Load projections
        self.projections_df = pd.read_csv(os.path.join(vis_root, 'indiana_projections.csv'))
        
        # Filter to only include frontal images
        self.frontal_images = self.projections_df[self.projections_df['projection'] == 'Frontal']
        
        # Create train/val/test split
        total_cases = len(self.frontal_images)
        train_size = int(0.7 * total_cases)
        val_size = int(0.2 * total_cases)
        
        if split == 'train':
            self.annotations = self.frontal_images[:train_size]
        elif split == 'val':
            self.annotations = self.frontal_images[train_size:train_size + val_size]
        elif split == 'test':
            self.annotations = self.frontal_images[train_size + val_size:]
            
        # Create mapping from uid to report text
        self.uid_to_report = dict(zip(self.reports_df['uid'], self.reports_df['findings']))
        
        self.img_size = cfg.datasets_cfg.mimic_cxr.vis_processor.train.image_size
        self.general_trans = transforms.Compose([Resize((512, 512)), CenterCrop(448), ToTensor(), ExpandChannels()])
        # Create image id mapping
        self.img_ids = {row['filename']: idx for idx, row in self.annotations.iterrows()}
        self.id_to_img = {v: k for k, v in self.img_ids.items()}
        
        # Store split
        self.split = split

        if truncate is not None:
            self.annotations = self.annotations[:truncate]
    
    def remap_to_uint8(self, array: np.ndarray, percentiles=None) -> np.ndarray:
        """Remap values in input so the output range is :math:`[0, 255]`.

        Percentiles can be used to specify the range of values to remap.
        This is useful to discard outliers in the input data.

        :param array: Input array.
        :param percentiles: Percentiles of the input values that will be mapped to ``0`` and ``255``.
            Passing ``None`` is equivalent to using percentiles ``(0, 100)`` (but faster).
        :returns: Array with ``0`` and ``255`` as minimum and maximum values.
        """
        array = array.astype(float)
        if percentiles is not None:
            len_percentiles = len(percentiles)
            if len_percentiles != 2:
                message = (
                    'The value for percentiles should be a sequence of length 2,'
                    f' but has length {len_percentiles}'
                )
                raise ValueError(message)
            a, b = percentiles
            if a >= b:
                raise ValueError(f'Percentiles must be in ascending order, but a sequence "{percentiles}" was passed')
            if a < 0 or b > 100:
                raise ValueError(f'Percentiles must be in the range [0, 100], but a sequence "{percentiles}" was passed')
            cutoff: np.ndarray = np.percentile(array, percentiles)
            array = np.clip(array, *cutoff)
        array -= array.min()
        array /= array.max()
        array *= 255
        return array.astype(np.uint8)

    def load_image(self, path) -> Image.Image:
        """Load an image from disk.

        The image values are remapped to :math:`[0, 255]` and cast to 8-bit unsigned integers.

        :param path: Path to image.
        :returns: Image as ``Pillow`` ``Image``.
        """
        # Although ITK supports JPEG and PNG, we use Pillow for consistency with older trained models
        if path.suffix in [".jpg", ".jpeg", ".png"]:
            image = io.imread(path)
        else:
            raise ValueError(f"Image type not supported, filename was: {path}")

        image = self.remap_to_uint8(image)
        return Image.fromarray(image).convert("L")
    
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        ann = self.annotations.iloc[idx]
        
        # Load image
        image_path = os.path.join(self.vis_root, 'images', 'images_normalized', ann['filename'])
        image = self.load_image(Path(image_path))
        image = self.general_trans(image)
            
        # Get report text
        report = self.uid_to_report.get(ann['uid'], '')
        
        # return {
        #     'image': image,
        #     'image_id': self.img_ids[ann['filename']],
        #     'text_output': report,
        #     'image_path': str(image_path)
        # }
        return {
            'image': image,
            'image_id': self.img_ids[ann['filename']]
        }
