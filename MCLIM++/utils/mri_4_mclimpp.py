import os
import torch
import open_clip
import numpy as np
import json
from scipy import ndimage
from monai.data import CacheDataset, DataLoader, Dataset, DistributedSampler, PersistentDataset, load_decathlon_datalist
from monai.transforms import (
    EnsureChannelFirstd,
    Compose,
    CropForegroundd,
    LoadImage,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    RandCropByPosNegLabeld,
    ScaleIntensityRangePercentilesd,
    CopyItemsd,
    SpatialPadd,
    DeleteItemsd,
    RandCropByLabelClassesd,
    ToTensord,
)
from monai.transforms.transform import MapTransform
import open_clip

with open('MCLIM++/standard/JHU5Levels/brain_structures_level4_GPT5.json', 'r') as file:
    LEVEL4JSON = json.load(file)

JHU4LABEL = list(LEVEL4JSON.keys())

with open('MCLIM++/standard/JHU5Levels/brain_structures_level3_GPT5.json', 'r') as file:
    LEVEL3JSON = json.load(file)

JHU3LABEL = list(LEVEL3JSON.keys())

with open('MCLIM++/standard/JHU5Levels/brain_structures_level2_GPT5.json', 'r') as file:
    LEVEL2JSON = json.load(file)

JHU2LABEL = list(LEVEL2JSON.keys())


TOKENIZER = open_clip.get_tokenizer('hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224')

# HIREA45 = [[1, 3, 5], [2, 4, 6], [9, 7], [8, 10], [11, 13, 15], [16, 12, 14], [17, 19], [18, 20], [21], [22], [23], [24], [25], [26], [27], 
#            [28], [29], [30], [31], [32], [33], [34], [35, 37], [36, 38], [41, 39], [40, 42], [43], [44], [45], [46], [49], [50], [51,53,55,57,59], 
#            [52,54,56,58,60], [65, 67, 69, 61, 63], [64, 66, 68, 70, 62], [71], [72], [73], [74], [275, 75], 
#            [76, 277], [77, 274], [78, 276], [79], [80], [81], [82], [83], [84], [161, 163, 165, 168, 85, 87, 157, 159], [89, 160, 162, 164, 166, 167, 86, 88, 158], 
#            [99, 97, 91, 93], [100, 98, 92, 94], [95], [96], [101, 103, 105, 107, 111, 113, 254], [102, 104, 106, 108, 112, 114, 255], 
#            [115, 109], [116, 110], [153, 117, 119], [120, 154, 118], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], 
#            [132], [133], [134], [135, 137, 147, 149, 151, 281], [136, 138, 148, 150, 152, 282], [139], [140], [141], [142], [143], [144], [145], [146], 
#            [206, 210, 212, 214, 216, 155], [207, 211, 213, 215, 217, 156], [169, 170, 278], [171, 172], [173], [279, 174, 175], [176, 177], [178], [179], 
#            [180], [181], [182], [183], [192, 194, 196, 198, 200, 202, 204, 208, 184, 186, 188, 190], [193, 195, 197, 199, 201, 203, 205, 209, 185, 187, 189, 191], 
#            [224, 226, 228, 218, 220, 222], [225, 227, 229, 219, 221, 223], [230, 232, 234, 236, 238], [231, 233, 235, 237, 239], [240, 242, 244, 246, 248], 
#            [241, 243, 245, 247, 249], [250, 253], [251, 252], [280], [47], [48]]

HIREA45 = [[1, 3, 5], [2, 4, 6], [9, 7], [8, 10], [11, 13, 15], [16, 12, 14], [17, 19], [18, 20], [21], [22], [23], [24], [25], [26], [27], 
           [28], [29], [30], [31], [32], [33], [34], [35, 37], [36, 38], [41, 39], [40, 42], [43], [44], [45], [46], [49], [50], [51,53,55,57,59], 
           [52,54,56,58,60], [65, 67, 69, 61, 63], [64, 66, 68, 70, 62], [71], [72], [73], [74], [75], 
           [76], [77, 274], [78, 276], [79], [80], [81], [82], [83], [84], [161, 163, 165, 168, 85, 87, 157, 159], [89, 160, 162, 164, 166, 167, 86, 88, 158], 
           [99, 97, 91, 93], [100, 98, 92, 94], [95], [96], [101, 103, 105, 107, 111, 113, 254], [102, 104, 106, 108, 112, 114, 255], 
           [115, 109], [116, 110], [153, 117, 119], [120, 154, 118], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], 
           [132], [133], [134], [135, 137, 147, 149, 151, 281], [136, 138, 148, 150, 152, 282], [139], [140], [141], [142], [143], [144], [145], [146], 
           [206, 210, 212, 214, 216, 155], [207, 211, 213, 215, 217, 156], [169, 170, 278], [171, 172], [173], [279, 174, 175], [176, 177], [178], [179], 
           [180], [181], [182], [183], [192, 194, 196, 198, 200, 202, 204, 208, 184, 186, 188, 190], [193, 195, 197, 199, 201, 203, 205, 209, 185, 187, 189, 191], 
           [224, 226, 228, 218, 220, 222], [225, 227, 229, 219, 221, 223], [230, 232, 234, 236, 238], [231, 233, 235, 237, 239], [242, 244, 246, 248], 
           [243, 245, 247, 249], [250, 253], [251, 252], [280], [47], [48]]#remove 240, 241 275 277

HIREA34 = [[1, 3, 5, 7, 9, 13], [2, 4, 6, 8, 10, 14], [11, 15, 17, 19, 21], [12, 16, 18, 20, 22], [23, 25, 27, 31], [24, 26, 32, 28], [29, 35, 39, 41, 111], [30, 36, 40, 42, 112], [33], [34],
           [37], [38], [43,45,47], [44,46,48], [49], [50], [51], [52], [53], [54], [55, 108], [56, 109], [57], [58], [59], [60], [61, 96, 100], [62, 97, 101],
           [63, 87, 104], [64, 88, 105], [65, 67, 69], [66, 68, 70], [71, 73, 75, 77, 98, 102], [72, 74, 76, 78, 99, 103], [79, 81, 83, 85, 106], [80, 82, 84, 86, 107], 
           [89, 90, 91], [92, 93, 94], [95],  [110]]

HIREA23 = [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12], [13], [14], [15], [16], [17], [18], [19], [20], [22, 23], [21, 24], [25],  [26], 
           [27, 29, 31, 33, 35], [28, 30, 32, 34, 36], [37, 38], [39, 40]]

HIREA12 = [[1, 3, 15], [2, 4, 16], [5, 7], [6, 8], [9], [10], [11], [12], [13], [14], [17, 18]]



class UniqueLabeld(MapTransform):
    """
    Convert the data-format to the format that only has unique label value.
    """

    def __init__(self, keys, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)

    def __call__(self, data):
        d = dict(data)
        onehots_l5 = []

        for i in range(d["jhu5"].shape[0]):
            onehot_l5 = np.zeros((282))

            sub_label = np.unique(d["jhu5"][i]).astype(np.uint8)
            if sub_label[0] == 0:
                sub_label = sub_label[1:]

            label_l5 = sub_label-1 #从0开始

            onehot_l5 = np.zeros((282)) #jhu5 with 282 labels
            onehot_l5[label_l5] = 1
            onehots_l5.append(onehot_l5)

        onehots_l5 = np.array(onehots_l5)
        d['label'] = onehots_l5
        return d



class UniqueLabelled_Plus(MapTransform):
    """
    For jhu level5 atlas patch, compute:
    1) level5 binary presence vector: label
    2) level5 patch ratio: patch_ratio
    3) level5 coverage ratio: coverage_ratio (requires uncropped full atlas)

    Optionally aggregate level5 labels to level4/3/2/1 using:
    HIREA45, HIREA34, HIREA23, HIREA12

    Outputs
    -------
    Always:
        d["label"]          : [B, num_classes_l5]
        d["patch_ratio"]    : [B, num_classes_l5]
        d["coverage_ratio"] : [B, num_classes_l5]

    If aggregate_hierarchy=True:
        d["label_l4"]       : [B, num_classes_l4]
        d["label_l3"]       : [B, num_classes_l3]
        d["label_l2"]       : [B, num_classes_l2]
        d["label_l1"]       : [B, num_classes_l1]
    """

    def __init__(
        self,
        keys,
        patch_key="jhu5",
        full_key="jhu5_full",
        num_classes_l5=282,
        num_classes_l4=None,
        num_classes_l3=None,
        num_classes_l2=None,
        num_classes_l1=None,
        allow_missing_keys=False,
        return_patch_ratio=True,
        return_coverage_ratio=True,
        aggregate_hierarchy=False,
    ):
        super().__init__(keys, allow_missing_keys)

        self.patch_key = patch_key
        self.full_key = full_key

        self.num_classes_l5 = num_classes_l5
        self.num_classes_l4 = num_classes_l4
        self.num_classes_l3 = num_classes_l3
        self.num_classes_l2 = num_classes_l2
        self.num_classes_l1 = num_classes_l1

        self.return_patch_ratio = return_patch_ratio
        self.return_coverage_ratio = return_coverage_ratio
        self.aggregate_hierarchy = aggregate_hierarchy

        if self.aggregate_hierarchy:
            if any(x is None for x in [HIREA45, HIREA34, HIREA23, HIREA12]):
                raise ValueError(
                    "When aggregate_hierarchy=True, HIREA45/HIREA34/HIREA23/HIREA12 must all be provided."
                )
            if any(x is None for x in [num_classes_l4, num_classes_l3, num_classes_l2, num_classes_l1]):
                raise ValueError(
                    "When aggregate_hierarchy=True, num_classes_l4/l3/l2/l1 must all be provided."
                )

            self.l5_to_l4 = self._build_child_to_parent_map(HIREA45)
            self.l4_to_l3 = self._build_child_to_parent_map(HIREA34)
            self.l3_to_l2 = self._build_child_to_parent_map(HIREA23)
            self.l2_to_l1 = self._build_child_to_parent_map(HIREA12)

    def _to_numpy(self, x):
        if hasattr(x, "detach"):  # torch.Tensor
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        return x

    def _ensure_batch_dim(self, x):
        """
        Support:
        - [H, W, D]
        - [B, H, W, D]
        - [B, 1, H, W, D]
        """
        x = self._to_numpy(x)

        if x.ndim == 3:
            x = x[None, ...]  # [1, H, W, D]
        elif x.ndim == 5 and x.shape[1] == 1:
            x = x[:, 0]       # [B, H, W, D]
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported atlas shape: {x.shape}")

        return x

    def _build_child_to_parent_map(self, hierarchy_list):
        """
        Example:
            HIREA45 = [
                [1,3,5],   # level4 label 1 contains level5 labels 1,3,5
                [2,4,6],   # level4 label 2 contains level5 labels 2,4,6
                ...
            ]

        Returns:
            child_to_parent = {
                1:1, 3:1, 5:1,
                2:2, 4:2, 6:2,
                ...
            }
        """
        mapping = {}
        for parent_idx, child_list in enumerate(hierarchy_list, start=1):
            for child in child_list:
                if child in mapping:
                    raise ValueError(f"Child label {child} appears multiple times in hierarchy mapping.")
                mapping[child] = parent_idx
        return mapping

    def _aggregate_to_parent_level(self, child_label_vec, child_to_parent_map, num_parent_classes):
        """
        child_label_vec: shape [num_child_classes], binary vector
        return parent_label_vec: shape [num_parent_classes], binary vector
        """
        parent_label_vec = np.zeros((num_parent_classes,), dtype=np.float32)

        child_indices = np.where(child_label_vec > 0)[0] + 1  # back to 1-based label
        for child_label in child_indices:
            parent_label = child_to_parent_map.get(int(child_label), None)
            if parent_label is not None:
                parent_label_vec[parent_label - 1] = 1.0

        return parent_label_vec

    def __call__(self, data):
        d = dict(data)

        patch_atlas = self._ensure_batch_dim(d[self.patch_key])

        if self.return_coverage_ratio:
            if self.full_key not in d:
                raise KeyError(
                    f"'{self.full_key}' not found in data, but coverage_ratio requires uncropped full atlas."
                )
            full_atlas = self._ensure_batch_dim(d[self.full_key])

            if full_atlas.shape[0] != patch_atlas.shape[0]:
                raise ValueError(
                    f"Batch size mismatch: {self.patch_key}.shape[0]={patch_atlas.shape[0]} "
                    f"but {self.full_key}.shape[0]={full_atlas.shape[0]}"
                )

        labels_l5_all = []
        patch_ratios_l5_all = []
        coverage_ratios_l5_all = []

        labels_l4_all = []
        labels_l3_all = []
        labels_l2_all = []
        labels_l1_all = []

        for i in range(patch_atlas.shape[0]):
            patch_i = patch_atlas[i]

            patch_labels, patch_counts = np.unique(patch_i, return_counts=True)
            patch_count_dict = dict(zip(patch_labels.astype(np.int32), patch_counts.astype(np.int64)))

            nonzero_patch_voxels = int(np.sum(patch_i > 0))

            label_l5_vec = np.zeros((self.num_classes_l5,), dtype=np.float32)
            patch_ratio_l5_vec = np.zeros((self.num_classes_l5,), dtype=np.float32)
            coverage_ratio_l5_vec = np.zeros((self.num_classes_l5,), dtype=np.float32)

            valid_patch_labels = [lab for lab in patch_count_dict.keys() if lab > 0]

            for lab in valid_patch_labels:
                cls_idx = lab - 1
                if cls_idx < 0 or cls_idx >= self.num_classes_l5:
                    continue

                patch_voxels = patch_count_dict[lab]

                label_l5_vec[cls_idx] = 1.0

                if self.return_patch_ratio and nonzero_patch_voxels > 0:
                    patch_ratio_l5_vec[cls_idx] = patch_voxels / nonzero_patch_voxels

                if self.return_coverage_ratio:
                    full_i = full_atlas[i]
                    full_region_voxels = np.sum(full_i == lab)
                    if full_region_voxels > 0:
                        coverage_ratio_l5_vec[cls_idx] = patch_voxels / full_region_voxels

            labels_l5_all.append(label_l5_vec)

            if self.return_patch_ratio:
                patch_ratios_l5_all.append(patch_ratio_l5_vec)

            if self.return_coverage_ratio:
                coverage_ratios_l5_all.append(coverage_ratio_l5_vec)

            if self.aggregate_hierarchy:
                label_l4_vec = self._aggregate_to_parent_level(
                    label_l5_vec, self.l5_to_l4, self.num_classes_l4
                )
                label_l3_vec = self._aggregate_to_parent_level(
                    label_l4_vec, self.l4_to_l3, self.num_classes_l3
                )
                label_l2_vec = self._aggregate_to_parent_level(
                    label_l3_vec, self.l3_to_l2, self.num_classes_l2
                )
                label_l1_vec = self._aggregate_to_parent_level(
                    label_l2_vec, self.l2_to_l1, self.num_classes_l1
                )

                labels_l4_all.append(label_l4_vec)
                labels_l3_all.append(label_l3_vec)
                labels_l2_all.append(label_l2_vec)
                labels_l1_all.append(label_l1_vec)

        d["label"] = np.stack(labels_l5_all, axis=0)

        if self.return_patch_ratio:
            d["patch_ratio"] = np.stack(patch_ratios_l5_all, axis=0)

        if self.return_coverage_ratio:
            d["coverage_ratio"] = np.stack(coverage_ratios_l5_all, axis=0)

        if self.aggregate_hierarchy:
            d["label_l4"] = np.stack(labels_l4_all, axis=0)
            d["label_l3"] = np.stack(labels_l3_all, axis=0)
            d["label_l2"] = np.stack(labels_l2_all, axis=0)
            d["label_l1"] = np.stack(labels_l1_all, axis=0)

        return d


class UniqueLabelled_Plus_Tumor(MapTransform):
    def __init__(
        self,
        keys=["jhu5", "jhu5_full", "tumor_mask"],
        patch_key="jhu5",
        full_count_key="jhu5_counts",
        tumor_key="tumor_mask",
        num_classes_l5=282,
        num_classes_l4=len(JHU4LABEL),
        num_classes_l3=len(JHU3LABEL),
        num_classes_l2=len(JHU2LABEL),
        num_classes_l1=len(HIREA12),
        allow_missing_keys=False,
        return_patch_ratio=True,
        return_coverage_ratio=True,
        tumor_threshold=0.3,
        tumor_morph_iter=10,
        args=None,
    ):
        super().__init__(keys, allow_missing_keys)

        self.patch_key = patch_key
        self.full_count_key = full_count_key
        self.tumor_key = tumor_key

        self.num_classes_l5 = num_classes_l5
        self.num_classes_l4 = num_classes_l4
        self.num_classes_l3 = num_classes_l3
        self.num_classes_l2 = num_classes_l2
        self.num_classes_l1 = num_classes_l1

        self.return_patch_ratio = return_patch_ratio
        self.return_coverage_ratio = return_coverage_ratio
        self.aggregate_hierarchy = args.hire

        self.tumor_threshold = tumor_threshold
        self.tumor_morph_iter = tumor_morph_iter
        self.use_tumor_label = args.tumor_aware

        # atlas label values to ignore (1-based)
        self.ignore_label_ids_l5 = [
            240, 241, 256, 257, 258, 259, 260, 261, 262, 263,
            264, 265, 266, 267, 268, 269, 270, 271, 273, 275, 277
        ]

        # convert to 0-based indices
        self.ignore_label_indices_l5 = [lab - 1 for lab in self.ignore_label_ids_l5]

        # 1 = keep, 0 = ignore
        self.valid_label_mask_l5 = np.ones((self.num_classes_l5,), dtype=np.float32)
        self.valid_label_mask_l5[self.ignore_label_indices_l5] = 0.0

        if self.aggregate_hierarchy:
            if any(x is None for x in [HIREA45, HIREA34, HIREA23, HIREA12]):
                raise ValueError(
                    "When aggregate_hierarchy=True, HIREA45/HIREA34/HIREA23/HIREA12 must all be provided."
                )
            if any(x is None for x in [num_classes_l4, num_classes_l3, num_classes_l2, num_classes_l1]):
                raise ValueError(
                    "When aggregate_hierarchy=True, num_classes_l4/l3/l2/l1 must all be provided."
                )

            self.l5_to_l4 = self._build_child_to_parent_map(HIREA45)
            self.l4_to_l3 = self._build_child_to_parent_map(HIREA34)
            self.l3_to_l2 = self._build_child_to_parent_map(HIREA23)
            self.l2_to_l1 = self._build_child_to_parent_map(HIREA12)

    def _to_numpy(self, x):
        if hasattr(x, "detach"):
            x = x.detach().cpu().numpy()
        else:
            x = np.asarray(x)
        return x

    def _ensure_batch_dim(self, x):
        x = self._to_numpy(x)

        if x.ndim == 3:
            x = x[None, ...]
        elif x.ndim == 5 and x.shape[1] == 1:
            x = x[:, 0]
        elif x.ndim == 4:
            pass
        else:
            raise ValueError(f"Unsupported atlas/mask shape: {x.shape}")

        return x

    def _match_batch_size(self, x, target_bs, key_name="tensor"):
        """
        Match x.shape[0] to target_bs.

        Allowed:
        - x.shape[0] == target_bs: keep as is
        - x.shape[0] == 1 and target_bs > 1: repeat along batch dim

        Otherwise raise error.
        """
        bs = x.shape[0]
        if bs == target_bs:
            return x
        if bs == 1 and target_bs > 1:
            return np.repeat(x, repeats=target_bs, axis=0)
        raise ValueError(
            f"Batch size mismatch for {key_name}: got {bs}, expected {target_bs}, "
            f"and cannot safely broadcast."
        )

    def _build_child_to_parent_map(self, hierarchy_list):
        mapping = {}
        for parent_idx, child_list in enumerate(hierarchy_list, start=1):
            for child in child_list:
                if child in mapping:
                    raise ValueError(f"Child label {child} appears multiple times in hierarchy mapping.")
                mapping[child] = parent_idx
        return mapping

    def _aggregate_to_parent_level(self, child_label_vec, child_to_parent_map, num_parent_classes):
        parent_label_vec = np.zeros((num_parent_classes,), dtype=np.float32)

        child_indices = np.where(child_label_vec > 0)[0] + 1
        for child_label in child_indices:
            parent_label = child_to_parent_map.get(int(child_label), None)
            if parent_label is not None:
                parent_label_vec[parent_label - 1] = 1.0

        return parent_label_vec

    def _clean_tumor_mask(self, tumor_mask):
        tumor_mask = tumor_mask > 0
        if self.tumor_morph_iter > 0:
            tumor_mask = ndimage.binary_erosion(tumor_mask, iterations=self.tumor_morph_iter)
            tumor_mask = ndimage.binary_dilation(tumor_mask, iterations=self.tumor_morph_iter)
        return tumor_mask.astype(np.uint8)
    
    def _ensure_count_batch_dim(self, x):
        """
        Support:
        - [282]
        - [1, 282]
        - [B, 282]
        """
        x = self._to_numpy(x).astype(np.float32)

        if x.ndim == 1:
            x = x[None, ...]   # [1, 282]
        elif x.ndim == 2:
            pass
        else:
            raise ValueError(f"Unsupported full-region-count shape: {x.shape}")

        return x

    def __call__(self, data):
        d = dict(data)

        patch_atlas = self._ensure_batch_dim(d[self.patch_key])
        patch_bs = patch_atlas.shape[0]

        if self.return_coverage_ratio:
            if self.full_count_key not in d:
                raise KeyError(
                    f"'{self.full_count_key}' not found in data, but coverage_ratio requires "
                    f"precomputed 282-dim full-region voxel counts."
                )

            full_region_counts = self._ensure_count_batch_dim(d[self.full_count_key])
            full_region_counts = self._match_batch_size(
                full_region_counts, patch_bs, key_name=self.full_count_key
            )

            if full_region_counts.shape[1] != self.num_classes_l5:
                raise ValueError(
                    f"{self.full_count_key} second dim must be {self.num_classes_l5}, "
                    f"but got {full_region_counts.shape}"
                )

        has_tumor_mask = self.use_tumor_label and (self.tumor_key in d)
        if has_tumor_mask:
            patch_tumor = self._ensure_batch_dim(d[self.tumor_key])
            patch_tumor = self._match_batch_size(patch_tumor, patch_bs, key_name=self.tumor_key)

        labels_l5_all = []
        patch_ratios_l5_all = []
        coverage_ratios_l5_all = []

        tumor_class_all = []
        tumor_onehot_all = []

        labels_l4_all = []
        labels_l3_all = []
        labels_l2_all = []
        labels_l1_all = []

        ignore_mask_all = []

        for i in range(patch_bs):
            patch_i = patch_atlas[i]

            patch_labels, patch_counts = np.unique(patch_i, return_counts=True)
            patch_count_dict = dict(zip(patch_labels.astype(np.int32), patch_counts.astype(np.int64)))

            nonzero_patch_voxels = int(np.sum(patch_i > 0))

            label_l5_vec = np.zeros((self.num_classes_l5,), dtype=np.float32)
            patch_ratio_l5_vec = np.zeros((self.num_classes_l5,), dtype=np.float32)
            coverage_ratio_l5_vec = np.zeros((self.num_classes_l5,), dtype=np.float32)

            tumor_class_vec = np.full((self.num_classes_l5,), -1, dtype=np.int64)
            tumor_onehot_vec = np.zeros((self.num_classes_l5, 3), dtype=np.float32)

            if has_tumor_mask:
                tumor_i = self._clean_tumor_mask(patch_tumor[i])
            else:
                tumor_i = None

            valid_patch_labels = [lab for lab in patch_count_dict.keys() if lab > 0]

            for lab in valid_patch_labels:
                cls_idx = lab - 1
                if cls_idx < 0 or cls_idx >= self.num_classes_l5:
                    continue

                region_mask = (patch_i == lab)
                region_voxels = patch_count_dict[lab]

                label_l5_vec[cls_idx] = 1.0

                if self.return_patch_ratio and nonzero_patch_voxels > 0:
                    patch_ratio_l5_vec[cls_idx] = region_voxels / nonzero_patch_voxels

                if self.return_coverage_ratio:
                    full_region_voxels = float(full_region_counts[i, cls_idx])
                    if full_region_voxels > 0:
                        coverage_ratio_l5_vec[cls_idx] = region_voxels / full_region_voxels

                if has_tumor_mask:
                    overlap_voxels = int((region_mask & (tumor_i > 0)).sum())
                    overlap_ratio = overlap_voxels / max(region_voxels, 1)

                    if overlap_voxels == 0:
                        tumor_class = 0
                    elif overlap_ratio < self.tumor_threshold:
                        tumor_class = 1
                    else:
                        tumor_class = 2

                    tumor_class_vec[cls_idx] = tumor_class
                    tumor_onehot_vec[cls_idx, tumor_class] = 1.0
                else:
                    tumor_class_vec[cls_idx] = 0
                    tumor_onehot_vec[cls_idx, 0] = 1.0

            # force ignored labels to not participate in training
            patch_ratio_l5_vec[self.ignore_label_indices_l5] = 0.0
            coverage_ratio_l5_vec[self.ignore_label_indices_l5] = 0.0
            tumor_onehot_vec[self.ignore_label_indices_l5, :] = 0.0
            ignore_mask_vec = self.valid_label_mask_l5.copy()

            labels_l5_all.append(label_l5_vec)

            if self.return_patch_ratio:
                patch_ratios_l5_all.append(patch_ratio_l5_vec)

            if self.return_coverage_ratio:
                coverage_ratios_l5_all.append(coverage_ratio_l5_vec)

            ignore_mask_all.append(ignore_mask_vec)

            if self.use_tumor_label:
                tumor_class_all.append(tumor_class_vec)
                tumor_onehot_all.append(tumor_onehot_vec)

            if self.aggregate_hierarchy:
                label_l4_vec = self._aggregate_to_parent_level(
                    label_l5_vec, self.l5_to_l4, self.num_classes_l4
                )
                label_l3_vec = self._aggregate_to_parent_level(
                    label_l4_vec, self.l4_to_l3, self.num_classes_l3
                )
                label_l2_vec = self._aggregate_to_parent_level(
                    label_l3_vec, self.l3_to_l2, self.num_classes_l2
                )
                label_l1_vec = self._aggregate_to_parent_level(
                    label_l2_vec, self.l2_to_l1, self.num_classes_l1
                )

                labels_l4_all.append(label_l4_vec)
                labels_l3_all.append(label_l3_vec)
                labels_l2_all.append(label_l2_vec)
                labels_l1_all.append(label_l1_vec)

        d["label"] = np.stack(labels_l5_all, axis=0)
        d["ignore_mask"] = np.stack(ignore_mask_all, axis=0)

        if self.return_patch_ratio:
            d["patch_ratio"] = np.stack(patch_ratios_l5_all, axis=0)

        if self.return_coverage_ratio:
            d["coverage_ratio"] = np.stack(coverage_ratios_l5_all, axis=0)

        if self.use_tumor_label:
            d["label_tumor_class"] = np.stack(tumor_class_all, axis=0)
            d["label_tumor_onehot"] = np.stack(tumor_onehot_all, axis=0)

        if self.aggregate_hierarchy:
            d["label_l4"] = np.stack(labels_l4_all, axis=0)
            d["label_l3"] = np.stack(labels_l3_all, axis=0)
            d["label_l2"] = np.stack(labels_l2_all, axis=0)
            d["label_l1"] = np.stack(labels_l1_all, axis=0)

        return d


class Mask_Origin_Img(MapTransform):
    """
    Mask the input image for MIM.
    """

    def __init__(self, keys, img_size, mask_ratio, patch_size, allow_missing_keys=False):
        super().__init__(keys, allow_missing_keys)
        self.mask_ratio = mask_ratio
        self.img_size = img_size 
        self.patch_size = patch_size
        self.patch_num_per_dim = int(img_size//patch_size)
        self.len_keep = round(self.patch_num_per_dim * self.patch_num_per_dim * self.patch_num_per_dim * (1 - self.mask_ratio))

    def __call__(self, data):
        d = dict(data)
        if self.mask_ratio>0:
            f: int = self.patch_num_per_dim
            idx = np.random.rand(f * f * f).argsort()
            idx = idx[:self.len_keep]
            msk = np.array(list(range(f * f * f))) 
            msk = np.where(np.isin(msk,idx),1,0)
            img = d['image']
            mask = np.zeros_like(img)
            
            for i in range(self.patch_num_per_dim):
                for j in range(self.patch_num_per_dim):
                    for k in range(self.patch_num_per_dim):
                            mask[:,i*self.patch_size:(i+1)*self.patch_size,j*self.patch_size:(j+1)*self.patch_size,k*self.patch_size:(k+1)*self.patch_size] = msk[(i*self.patch_num_per_dim*self.patch_num_per_dim)+(j*self.patch_num_per_dim)+k]

            d['mask_image'] = img * mask
            d['mask'] = mask
        return d



def build_dataset_to_pretrain(
    dataset_path_adni,
    dataset_path_multi,
    input_size,
    mim_ratio,
    patch_size,
    args,
    inference=False
) -> Dataset:
    ratios = [0] + [1] * 282

    tr_transforms = Compose(
    [
        LoadImaged(
            keys=["image", "jhu5", "tumor_mask", "jacobian"],
            image_only=True,
            allow_missing_keys=True,
        ),

        EnsureChannelFirstd(
            keys=["image", "jhu5", "tumor_mask", "jacobian"],
            allow_missing_keys=True,
        ),
        Orientationd(
            keys=["image", "jhu5", "tumor_mask", "jacobian"],
            axcodes="RAS",
            allow_missing_keys=True,
        ),

        CropForegroundd(
            keys=["image", "jhu5", "tumor_mask", "jacobian"],
            source_key="jhu5",
            allow_missing_keys=True,
        ),

        ScaleIntensityRangePercentilesd(
            keys=["image"],
            lower=1,
            upper=99,
            b_min=0,
            b_max=1,
            allow_missing_keys=False,
        ),
        NormalizeIntensityd(
            keys=["image"],
            allow_missing_keys=False,
        ),
        SpatialPadd(
            keys=["image", "jhu5", "tumor_mask", "jacobian"],
            spatial_size=[input_size, input_size, input_size],
            allow_missing_keys=True,
        ),
        
        RandCropByLabelClassesd(
            keys=["image", "jhu5", "tumor_mask", "jacobian"],
            label_key="jhu5",
            spatial_size=[input_size, input_size, input_size],
            ratios=ratios,
            num_classes=283,
            num_samples=2,
            allow_missing_keys=True,
        ),

        UniqueLabelled_Plus_Tumor(
            keys=["jhu5", "jhu5_counts", "tumor_mask"],
            patch_key="jhu5",
            full_count_key="jhu5_counts",
            tumor_key="tumor_mask",
            args=args,
        ),

        Mask_Origin_Img(
            keys=["image"],
            img_size=input_size,
            mask_ratio=mim_ratio,
            patch_size=patch_size,
        ),

        ToTensord(
            keys=[
                "image",
                "modality",
                "mask_image",
                "mask",
                "label",
                "patch_ratio",
                "coverage_ratio",
                "label_tumor_class",
                "label_tumor_onehot",
                "ignore_mask",
                "jacobian",
            ],
            track_meta=False,
            allow_missing_keys=True,
        ),

        # 最后删掉中间字段
        DeleteItemsd(
            keys=["jhu5", "jhu5_counts", "tumor_mask"],
        ),
    ]
)
    datalist = []

    scan_list_adni = sorted(os.listdir(dataset_path_adni))
    scan_list_multi = sorted(os.listdir(dataset_path_multi))

    # -------------------------
    # ADNI
    # -------------------------
    for scan in scan_list_adni:
        scan_dir = os.path.join(dataset_path_adni, scan)
        if not os.path.isdir(scan_dir):
            continue

        image_path = os.path.join(scan_dir, "Rigid_Warped.nii.gz")
        jhu5_path = os.path.join(scan_dir, "Rigid_Warped_jhu5.nii.gz")
        jhu5_count_path = jhu5_path.replace(".nii.gz", "_voxel_count.npy")
        jacobian_path = os.path.join(scan_dir, "Rigid_Warped_jhu5_Jacobian.nii.gz")

        if not os.path.isfile(image_path):
            print(f"[SKIP ADNI] missing image: {image_path}")
            continue
        if not os.path.isfile(jhu5_path):
            print(f"[SKIP ADNI] missing jhu5: {jhu5_path}")
            continue
        if not os.path.isfile(jhu5_count_path):
            print(f"[SKIP ADNI] missing jhu5 voxel count: {jhu5_count_path}")
            continue

        item = {
            "image": image_path,
            "jhu5": jhu5_path,
            "jhu5_counts": np.load(jhu5_count_path).astype(np.float32),
            "modality": 0,   
        }

        if args.deform:
            item["jacobian"] = jacobian_path

        datalist.append(item)

    # -------------------------
    # MULTI
    # -------------------------
    for scan in scan_list_multi:
        scan_dir = os.path.join(dataset_path_multi, scan)
        if not os.path.isdir(scan_dir):
            continue

        jhu5_path = os.path.join(scan_dir, "Rigid_Warped_jhu5.nii.gz")
        jhu5_count_path = jhu5_path.replace(".nii.gz", "_voxel_count.npy")
        jacobian_path = os.path.join(scan_dir, "Rigid_Warped_jhu5_Jacobian.nii.gz")

        if not os.path.isfile(jhu5_path):
            print(f"[SKIP MULTI] missing jhu5: {jhu5_path}")
            continue
        if not os.path.isfile(jhu5_count_path):
            print(f"[SKIP MULTI] missing jhu5 voxel count: {jhu5_count_path}")
            continue

        tumor_mask_path = os.path.join(scan_dir, f"{scan}_sudomask.nii.gz")
        has_tumor_mask = os.path.isfile(tumor_mask_path)

        modals = []
        if "HCP" in scan:
            modals = ["t1", "t2"]
        elif "IXI" in scan:
            modals = ["t1", "t2", "pd"]
        elif "TU" in scan:
            modals = ["t1", "t2", "t1ce", "flair"]
        else:
            continue

        for modal in modals:
            
            image_path = os.path.join(scan_dir, f"{scan}_{modal}.nii.gz")
            if not os.path.isfile(image_path):
                print(f"[SKIP {scan}] missing image: {image_path}")
                continue

            item = {
                "image": image_path,
                "jhu5": jhu5_path,
                "jhu5_counts": np.load(jhu5_count_path).astype(np.float32),
            }

            if has_tumor_mask:
                item["tumor_mask"] = tumor_mask_path

            if modal=='t1':
                item["modality"] = 0
            elif modal=='t1ce':
                item["modality"] = 1
            elif modal=='t2':
                item["modality"] = 2
            elif modal=='flair':
                item["modality"] = 3
            elif modal=='pd':
                item["modality"] = 4

            if args.deform:
                item["jacobian"] = jacobian_path

            datalist.append(item)

    print("Dataset all training: number of data: {}".format(len(datalist)))

    dataset_train = Dataset(data=datalist, transform=tr_transforms)
    return dataset_train


if __name__ == '__main__':
    dataset = build_dataset_to_pretrain('MCLIM/adni_affine_ss_full_list_clean.txt', 64)
    data_loader_train = DataLoader(
        dataset=dataset, batch_size=2, num_workers=4, pin_memory=False, persistent_workers=True, collate_fn=custom_list_data_collate
    )
    for data in data_loader_train:
        print(data['data'].shape)
        print(data['label'].shape)
        break