import numpy as np
import torch
import torch.nn as nn
import open_clip
from torch.nn import functional as F

from models.unet import UNetEncoder, UNetDecoder

from models.swinunetr import SwinUNETR

from transformers.models.bert.modeling_bert import *

try:
    import torch.distributed.nn
    from torch import distributed as dist

    has_distributed = True
except ImportError:
    has_distributed = False

try:
    import horovod.torch as hvd
except ImportError:
    hvd = None


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

def gather_features(
    image_features,
    text_features,
    local_loss=False,
    gather_with_grad=False,
    rank=0,
    world_size=1,
    use_horovod=False  # included for signature continuity, though not used below
):
    """
    Gathers image and text features across distributed processes.

    Args:
        image_features: Tensor of shape (batch_size, feature_dim)
        text_features: Tensor of shape (batch_size, feature_dim)
        local_loss: bool, whether to apply local-only loss (exclude other ranks' features in grad)
        gather_with_grad: bool, whether to allow gradient flow through the gather operation
        rank: int, current process rank
        world_size: int, total number of processes
        use_horovod: bool, placeholder (not actively used here)

    Returns:
        all_image_features, all_text_features: concatenated tensors from all processes
    """
    if local_loss and not gather_with_grad:
        raise ValueError("local_loss=True with gather_with_grad=False is not supported safely in this implementation.")
    
    if gather_with_grad:
        # Allows gradients to flow through the gathering operation
        gathered_image = torch.distributed.nn.all_gather(image_features)
        gathered_text = torch.distributed.nn.all_gather(text_features)
        all_image_features = torch.cat(gathered_image, dim=0)
        all_text_features = torch.cat(gathered_text, dim=0)
    else:
        # No gradient through gather
        gathered_image = [torch.zeros_like(image_features) for _ in range(world_size)]
        gathered_text = [torch.zeros_like(text_features) for _ in range(world_size)]
        dist.all_gather(gathered_image, image_features)
        dist.all_gather(gathered_text, text_features)
        
        if not local_loss:
            # Ensure local rank's originals are included even if grads aren't flowing
            gathered_image[rank] = image_features
            gathered_text[rank] = text_features

        all_image_features = torch.cat(gathered_image, dim=0)
        all_text_features = torch.cat(gathered_text, dim=0)
        
    return all_image_features, all_text_features


class ClipLoss(nn.Module):
    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)
            # print('BS: '+str(all_image_features.shape[0]))
            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale=20):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)
        labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = (
            F.cross_entropy(logits_per_image, labels, ignore_index=-100) +
            F.cross_entropy(logits_per_text, labels, ignore_index=-100)
        ) / 2

        return total_loss
    

class MatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, image_features, text_features, labels, logit_scale, pos_weight=None):
        logits_per_image = logit_scale * (image_features @ text_features.T)

        if pos_weight is not None:
            pos_weight = pos_weight.to(logits_per_image.device, dtype=logits_per_image.dtype)

        total_loss = F.binary_cross_entropy_with_logits(
            logits_per_image,
            labels.float(),
            pos_weight=pos_weight
        )
        return total_loss


def masked_recon_loss(input, recon, mask, norm='l2', eps=1e-8, invert_mask=True):
    if invert_mask:
        mask = 1 - mask

    mask = mask.to(input.dtype)

    if norm == 'l2':
        diff = (input - recon).pow(2)
    elif norm == 'l1':
        diff = (input - recon).abs()
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    diff = diff * mask

    # per-sample loss
    B = input.shape[0]
    diff = diff.view(B, -1)
    mask = mask.view(B, -1)

    per_sample_loss = diff.sum(dim=1) / mask.sum(dim=1).clamp_min(eps)
    return per_sample_loss.mean()


class ReconstructLoss(nn.Module):
    def __init__(self, invert_mask=True, eps=1e-8):
        super().__init__()
        self.invert_mask = invert_mask
        self.eps = eps

    def forward(self, origin_image, reconstruct_image, mim_mask, norm='l2'):
        return masked_recon_loss(
            origin_image,
            reconstruct_image,
            mim_mask,
            norm=norm,
            eps=self.eps,
            invert_mask=self.invert_mask,
        )
   

class HierarchicalMarginalizationLossStable(nn.Module):


    def __init__(
        self,
        HIREA45,
        HIREA34,
        HIREA23,
        HIREA12,
        level_weights=(1.0, 0.75, 0.75, 0.5, 0.25),   # (w5, w4, w3, w2, w1)
        recompute_parent_labels=True,
        aggregation="sqrt_mean",   # "sum" | "mean" | "sqrt_mean"
        clamp_parent_logits=8.0,  # 例如 8.0；None 表示不截断
    ):
        super().__init__()

        self.level_weights = level_weights
        self.recompute_parent_labels = recompute_parent_labels
        self.aggregation = aggregation
        self.clamp_parent_logits = clamp_parent_logits

        self.register_buffer("A45", self._build_child_to_parent_matrix(HIREA45))
        self.register_buffer("A34", self._build_child_to_parent_matrix(HIREA34))
        self.register_buffer("A23", self._build_child_to_parent_matrix(HIREA23))
        self.register_buffer("A12", self._build_child_to_parent_matrix(HIREA12))

    @staticmethod
    def _build_child_to_parent_matrix(mapping):
        n_parent = len(mapping)
        n_child = max(max(children) for children in mapping if len(children) > 0)

        A = torch.zeros(n_child, n_parent, dtype=torch.float32)
        for p_idx, children in enumerate(mapping):
            for c in children:
                A[c - 1, p_idx] = 1.0
        return A

    @staticmethod
    def _masked_bce_with_logits(logits, targets, valid_mask):
        loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        loss = loss * valid_mask
        denom = valid_mask.sum().clamp(min=1.0)
        return loss.sum() / denom

    @staticmethod
    def _prepare_mask(ignore_mask, ref_tensor):
        if ignore_mask is None:
            return torch.ones_like(ref_tensor, dtype=torch.float32)
        if ignore_mask.dim() == 3 and ignore_mask.size(1) == 1:
            ignore_mask = ignore_mask.squeeze(1)
        return ignore_mask.float()

    def _aggregate_logits_up(self, child_logits, child_valid_mask, A):
        """
        child_logits:     [B, n_child]
        child_valid_mask: [B, n_child], 1=保留, 0=忽略
        A:                [n_child, n_parent]
        """
        masked_child_logits = child_logits * child_valid_mask                  # [B, n_child]
        parent_sum_logits = masked_child_logits @ A                           # [B, n_parent]
        valid_child_count = child_valid_mask @ A                              # [B, n_parent]
        parent_valid_mask = (valid_child_count > 0).float()

        if self.aggregation == "sum":
            parent_logits = parent_sum_logits
        elif self.aggregation == "mean":
            parent_logits = parent_sum_logits / valid_child_count.clamp(min=1.0)
        elif self.aggregation == "sqrt_mean":
            parent_logits = parent_sum_logits / torch.sqrt(valid_child_count.clamp(min=1.0))
        else:
            raise ValueError(f"Unknown aggregation mode: {self.aggregation}")

        # 对于完全无有效 child 的父类，logit 直接置0（反正该位置也会被 mask 掉）
        parent_logits = parent_logits * parent_valid_mask

        if self.clamp_parent_logits is not None:
            parent_logits = parent_logits.clamp(
                min=-self.clamp_parent_logits,
                max=self.clamp_parent_logits
            )

        return parent_logits, parent_valid_mask

    @staticmethod
    def _aggregate_targets_up(child_targets, child_valid_mask, A):
        effective_child_targets = child_targets * child_valid_mask
        parent_targets = ((effective_child_targets @ A) > 0).float()
        return parent_targets

    def forward(
        self,
        logits_l5,
        labels_l5_all,
        labels_l4_all=None,
        labels_l3_all=None,
        labels_l2_all=None,
        labels_l1_all=None,
        ignore_mask=None,
    ):
        labels_l5 = labels_l5_all.float()

        keep_l5 = self._prepare_mask(ignore_mask, labels_l5)

        assert logits_l5.shape == labels_l5.shape, (
            f"logits_l5 {logits_l5.shape} vs labels_l5 {labels_l5.shape} mismatch"
        )
        assert logits_l5.size(1) == self.A45.size(0), (
            f"level5 dim={logits_l5.size(1)} but A45 child dim={self.A45.size(0)}"
        )

        logits_l4, keep_l4 = self._aggregate_logits_up(logits_l5, keep_l5, self.A45)
        logits_l3, keep_l3 = self._aggregate_logits_up(logits_l4, keep_l4, self.A34)
        logits_l2, keep_l2 = self._aggregate_logits_up(logits_l3, keep_l3, self.A23)
        logits_l1, keep_l1 = self._aggregate_logits_up(logits_l2, keep_l2, self.A12)

        if self.recompute_parent_labels:
            labels_l4 = self._aggregate_targets_up(labels_l5, keep_l5, self.A45)
            labels_l3 = self._aggregate_targets_up(labels_l4, keep_l4, self.A34)
            labels_l2 = self._aggregate_targets_up(labels_l3, keep_l3, self.A23)
            labels_l1 = self._aggregate_targets_up(labels_l2, keep_l2, self.A12)
        else:
            assert labels_l4_all is not None
            assert labels_l3_all is not None
            assert labels_l2_all is not None
            assert labels_l1_all is not None

            labels_l4 = labels_l4_all.float()
            labels_l3 = labels_l3_all.float()
            labels_l2 = labels_l2_all.float()
            labels_l1 = labels_l1_all.float()

        loss_l5 = self._masked_bce_with_logits(logits_l5, labels_l5, keep_l5)
        loss_l4 = self._masked_bce_with_logits(logits_l4, labels_l4, keep_l4)
        loss_l3 = self._masked_bce_with_logits(logits_l3, labels_l3, keep_l3)
        loss_l2 = self._masked_bce_with_logits(logits_l2, labels_l2, keep_l2)
        loss_l1 = self._masked_bce_with_logits(logits_l1, labels_l1, keep_l1)

        w5, w4, w3, w2, w1 = self.level_weights
        loss_total = (
            w5 * loss_l5 +
            w4 * loss_l4 +
            w3 * loss_l3 +
            w2 * loss_l2 +
            w1 * loss_l1
        )

        return {
            "loss_total": loss_total,
            "loss_l5": loss_l5,
            "loss_l4": loss_l4,
            "loss_l3": loss_l3,
            "loss_l2": loss_l2,
            "loss_l1": loss_l1,
        }
    

def safe_normalize(x, dim=1, eps=1e-6):
    norm = x.norm(dim=dim, keepdim=True)
    return x / norm.clamp_min(eps)

class ProjectionHead(nn.Module):
    def __init__(self, in_dim, out_dim=512, hidden_dim=None, dropout=0.1):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return self.net(x)


class MCLIM(nn.Module):
    def __init__(self, rank=0, world_size=1, local_batch_size=4, input_size=64, backbone='unet', feature_size=32):
        super().__init__()
        self.local_bs = local_batch_size

        self.backbone = backbone

        if backbone=='unet':
            self.image_encoder = UNetEncoder(dims=[32, 64, 128, 256, 512])
            self.image_decoder = UNetDecoder(dims=[32, 64, 128, 256, 512])

        if backbone=='swinunetr':
            self.image_encoder_decoder = SwinUNETR(
                img_size=input_size,
                in_channels=1,
                out_channels=1,
                feature_size=feature_size,
                use_checkpoint=False,
            )
            self.image_proj = ProjectionHead(feature_size*16, out_dim=512)
            
            

        self.text_encoder = open_clip.create_model_and_transforms(
            'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'
        )[0].text
        
        self.clip_loss = ClipLoss(rank=rank, world_size=world_size)
        self.reconstruct_loss = ReconstructLoss()
        self.warp_loss_fun = ReconstructLoss(invert_mask=False)

        # [M, T, R, L]
        self.register_buffer('text_tokens_all', torch.randn(5, 3, 282, 64).long())
        text_tokens_all = torch.load(
            'Text_Token_of_brain_region_discription.pth', #Use Different LLMs dim:(Modality, TumorState, Region, TokenLength)
            map_location='cpu'
        ).detach().long()
        self.text_tokens_all.copy_(text_tokens_all)

        self.scale_factor = nn.Parameter(torch.ones([]) * np.log(1 / 1))
        self.proj_intensity = nn.Conv3d(feature_size, 1, kernel_size=3, stride=1, padding=1, bias=True)
        self.proj_warp = nn.Conv3d(feature_size, 1, kernel_size=3, stride=1, padding=1, bias=True)

        self.text_encoder.eval()
        for p in self.text_encoder.parameters():
            p.requires_grad = False

        with torch.no_grad():
            M, T, R, L = self.text_tokens_all.shape
            flat_tokens = self.text_tokens_all.view(M * T * R, L)
            flat_prototypes, _ = self.text_encoder(flat_tokens)
            text_prototypes_all = flat_prototypes.view(M, T, R, -1)

        self.register_buffer("text_prototypes_all", text_prototypes_all.detach())

        self.region_head_l5 = nn.Linear(16*feature_size, 282)

        self.hier_region_loss = HierarchicalMarginalizationLossStable(
            HIREA45=HIREA45,
            HIREA34=HIREA34,
            HIREA23=HIREA23,
            HIREA12=HIREA12,
            level_weights=(1.0, 0.75, 0.75, 0.5, 0.25),
            recompute_parent_labels=True,
            aggregation="sqrt_mean",
            clamp_parent_logits=8.0,
        )

    def aggregate_text_by_modality_and_tumor(self, modality, weights_l5, label_tumor_onehot=None):
        selected_proto = self.text_prototypes_all[modality]   # [B, T, R, D]

        if label_tumor_onehot is None:
            healthy_proto = selected_proto[:, 0]              # [B, R, D]
            text_global = torch.einsum('br,brd->bd', weights_l5, healthy_proto)
        else:
            proto = selected_proto.permute(0, 2, 1, 3)       # [B, R, T, D]
            region_state_weights = weights_l5.unsqueeze(-1) * label_tumor_onehot.float()
            text_global = torch.einsum('brt,brtd->bd', region_state_weights, proto)

        return text_global


    def modality_conditioned_matching_loss(
        self,
        image_global_features,
        modality,
        label,
        logit_scale,
        label_tumor_onehot=None,
        ignore_mask=None,   # [B, R], 1=keep, 0=ignore
        ):
        selected_proto = self.text_prototypes_all[modality]   # [B, T, R, D]

        if label_tumor_onehot is None:
            # healthy-only matching
            proto = selected_proto[:, 0]                      # [B, R, D]
            logits = logit_scale * torch.einsum('bd,brd->br', image_global_features, proto)
            target = label.float()                            # [B, R]

            if ignore_mask is None:
                elem_mask = torch.ones_like(target, dtype=logits.dtype, device=logits.device)
            else:
                elem_mask = ignore_mask.to(logits.device, dtype=logits.dtype)   # [B, R]

        else:
            # tumor-aware matching
            proto = selected_proto.permute(0, 2, 1, 3)       # [B, R, T, D]
            logits = logit_scale * torch.einsum('bd,brtd->brt', image_global_features, proto)
            target = label.unsqueeze(-1).float() * label_tumor_onehot.float()   # [B, R, T]

            if ignore_mask is None:
                elem_mask = torch.ones_like(target, dtype=logits.dtype, device=logits.device)
            else:
                elem_mask = ignore_mask.unsqueeze(-1).to(logits.device, dtype=logits.dtype)  # [B, R, 1]
                elem_mask = elem_mask.expand_as(target)                                         # [B, R, T]

        # 只在未忽略元素上统计正负样本
        valid_target = target[elem_mask > 0]

        if valid_target.numel() == 0:
            return logits.sum() * 0.0

        num_pos = valid_target.sum()
        num_neg = valid_target.numel() - num_pos
        pos_weight = torch.clamp(num_neg / (num_pos + 1e-8), min=1.0, max=10.0)

        loss_raw = F.binary_cross_entropy_with_logits(
            logits,
            target,
            reduction="none",
            pos_weight=pos_weight
        )

        match_loss = (loss_raw * elem_mask).sum() / (elem_mask.sum() + 1e-8)
        return match_loss

    def forward(self,image,modality,mask,mask_image,label,ignore_mask,patch_ratio,coverage_ratio,weight_recon,label_tumor_class=None,label_tumor_onehot=None,jacobian=None,labels_l4_all=None,labels_l3_all=None,labels_l2_all=None,labels_l1_all=None):
        
        label = label.float().squeeze(1)
        patch_ratio = patch_ratio.float().squeeze(1)
        coverage_ratio = coverage_ratio.float().squeeze(1)
        modality = modality.long()

        if label_tumor_class is not None:
            label_tumor_class = label_tumor_class.long().squeeze(1)
        if label_tumor_onehot is not None:
            label_tumor_onehot = label_tumor_onehot.float().squeeze(1)

        if labels_l4_all is not None:
            labels_l4_all = labels_l4_all.float()
            labels_l3_all = labels_l3_all.float()
            labels_l2_all = labels_l2_all.float()
            labels_l1_all = labels_l1_all.float()

        if mask_image is not None:
            if self.backbone == 'unet':
                image_features_pyramid = self.image_encoder(mask_image)
                
            if self.backbone == 'swinunetr':
                image_features_pyramid, decode_feature = self.image_encoder_decoder(mask_image)
        else:
            if self.backbone == 'unet':
                image_features_pyramid = self.image_encoder(image)
            if self.backbone == 'swinunetr':
                image_features_pyramid, decode_feature = self.image_encoder_decoder(image)

        image_features_pyramid.reverse()
        image_local_features = image_features_pyramid[0]
        image_global_features = image_local_features.mean(dim=(2, 3, 4))

        if self.backbone == 'unet':
            decode_feature = self.image_decoder(image_features_pyramid)

        if (mask_image is not None) and (weight_recon > 0):
            reconstruct_img = self.proj_intensity(decode_feature)
            recon_loss = self.reconstruct_loss(image, reconstruct_img, mask, 'l2')
        else:
            recon_loss = 0.0

        semantic_loss = 0.0
        if jacobian is not None:
            deformation_field = self.proj_warp(decode_feature)
            jacobian_mask = jacobian != 0
            jacobian_inv = torch.zeros_like(jacobian)
            jacobian_inv[jacobian_mask] = 1.0 / (jacobian[jacobian_mask] + 1e-6)
            semantic_loss = self.warp_loss_fun(jacobian_inv, deformation_field, jacobian_mask, 'l1')

        # region weights
        weights_l5 = label * torch.sqrt(
            torch.clamp(patch_ratio, min=1e-8) *
            torch.clamp(coverage_ratio, min=1e-8)
        )
        weights_l5 = weights_l5 / (weights_l5.sum(dim=1, keepdim=True) + 1e-8)

        # tumor-state-aware text aggregation
        text_global_features_pyto_l5 = self.aggregate_text_by_modality_and_tumor(
            modality=modality,
            weights_l5=weights_l5,
            label_tumor_onehot=label_tumor_onehot
        )
        image_global_features_unnorm = image_global_features

        if self.backbone == 'swinunetr':
            image_global_features = self.image_proj(image_global_features)

        image_global_features = safe_normalize(image_global_features, dim=1)
        text_global_features_pyto_l5 = safe_normalize(text_global_features_pyto_l5, dim=1)
        logit_scale = self.scale_factor.exp()

        clip_loss = self.clip_loss(image_global_features, text_global_features_pyto_l5)

        # Matching
        match_loss = self.modality_conditioned_matching_loss(
            image_global_features=image_global_features,
            modality=modality,
            label=label,
            logit_scale=logit_scale,
            label_tumor_onehot=label_tumor_onehot,
            ignore_mask=ignore_mask.squeeze(1),
        )

        if (jacobian is not None) and (labels_l4_all is None):
            total_loss = clip_loss + match_loss + weight_recon * recon_loss + semantic_loss
            return total_loss, clip_loss, match_loss, recon_loss, semantic_loss, 0
        
        if labels_l4_all is not None:
            logits_l5 = self.region_head_l5(image_global_features_unnorm)
            hier_out = self.hier_region_loss(
            logits_l5=logits_l5,
            labels_l5_all=label,
            labels_l4_all=labels_l4_all,
            labels_l3_all=labels_l3_all,
            labels_l2_all=labels_l2_all,
            labels_l1_all=labels_l1_all,
            ignore_mask=ignore_mask.squeeze(1),
            )
            loss_hier = hier_out["loss_total"]
            loss_l5 = hier_out["loss_l5"]
            loss_l4 = hier_out["loss_l4"]
            loss_l3 = hier_out["loss_l3"]
            loss_l2 = hier_out["loss_l2"]
            loss_l1 = hier_out["loss_l1"]
            total_loss = clip_loss + match_loss + weight_recon * recon_loss + semantic_loss + loss_hier
            return total_loss, clip_loss, match_loss, recon_loss, semantic_loss, loss_hier, loss_l5, loss_l4, loss_l3, loss_l2, loss_l1

        total_loss = clip_loss + match_loss + weight_recon * recon_loss
        return total_loss, clip_loss, match_loss, recon_loss, 0, 0
        
        


if __name__ == '__main__':
    img = torch.randn(2, 1, 64, 64, 64)
