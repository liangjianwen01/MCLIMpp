import json
import torch
import open_clip

# =========================
# paths
# =========================
atlas_order_json = "/data/junyan/lab/MCLIM++/standard/JHU5Levels/brain_structures_level5_GPT5.json"

health_json = "/data/junyan/lab/MCLIM++/standard/Gemini35_brain_structures_level5_health_biomedbert.json"
mild_json = "/data/junyan/lab/MCLIM++/standard/Gemini35_brain_structures_level5_mild_tumor_involvement_biomedbert.json"
severe_json = "/data/junyan/lab/MCLIM++/standard/Gemini35_brain_structures_level5_severe_tumor_involvement_biomedbert.json"

save_path = "/data/junyan/lab/MCLIM++/standard/Gemini_atlas_jhu5_modality_tumor_token_biomedclip_ctx64_aligned_to_atlas_order.pth"

# =========================
# tokenizer
# =========================
tokenizer = open_clip.get_tokenizer(
    "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224"
)

# =========================
# load json
# =========================
with open(atlas_order_json, "r", encoding="utf-8") as f:
    atlas_order_data = json.load(f)

with open(health_json, "r", encoding="utf-8") as f:
    health_data = json.load(f)

with open(mild_json, "r", encoding="utf-8") as f:
    mild_data = json.load(f)

with open(severe_json, "r", encoding="utf-8") as f:
    severe_data = json.load(f)

# =========================
# use atlas json key order as the ONLY region order
# =========================
region_names = list(atlas_order_data.keys())

print(f"num atlas regions = {len(region_names)}")

# 严格检查三个描述 json 是否都覆盖 atlas 的所有脑区
atlas_key_set = set(region_names)
health_key_set = set(health_data.keys())
mild_key_set = set(mild_data.keys())
severe_key_set = set(severe_data.keys())

missing_in_health = atlas_key_set - health_key_set
missing_in_mild = atlas_key_set - mild_key_set
missing_in_severe = atlas_key_set - severe_key_set

extra_in_health = health_key_set - atlas_key_set
extra_in_mild = mild_key_set - atlas_key_set
extra_in_severe = severe_key_set - atlas_key_set

assert len(missing_in_health) == 0, f"health json 缺少 atlas 脑区: {sorted(list(missing_in_health))[:10]}"
assert len(missing_in_mild) == 0, f"mild json 缺少 atlas 脑区: {sorted(list(missing_in_mild))[:10]}"
assert len(missing_in_severe) == 0, f"severe json 缺少 atlas 脑区: {sorted(list(missing_in_severe))[:10]}"

if len(extra_in_health) > 0:
    print(f"[Warning] health json 有 atlas 中不存在的额外脑区，数量={len(extra_in_health)}")
if len(extra_in_mild) > 0:
    print(f"[Warning] mild json 有 atlas 中不存在的额外脑区，数量={len(extra_in_mild)}")
if len(extra_in_severe) > 0:
    print(f"[Warning] severe json 有 atlas 中不存在的额外脑区，数量={len(extra_in_severe)}")

# 固定模态顺序
modalities = ["T1", "T1ce", "T2", "FLAIR", "PD"]

# 固定状态顺序
states = ["health", "mild", "severe"]

def build_text(region_name, region_info, modality, health=False):
    """
    目标格式：
    脑区名字; Morphology; Location; Signal_模态; Tumor_Involvement(健康脑区不需要)
    """
    morphology = str(region_info.get("Morphology", "")).strip()
    location = str(region_info.get("Location", "")).strip()
    signal = str(region_info.get(f"Signal_{modality}", "")).strip()

    parts = [region_name, morphology, location, signal]

    if not health:
        tumor_involvement = str(region_info.get("Tumor_Involvement", "")).strip()
        if tumor_involvement:
            parts.append(tumor_involvement)

    parts = [p for p in parts if p]
    return "; ".join(parts)

# texts_by_state[state_idx][modality_idx] = list of 282 strings
texts_by_state = [[[] for _ in modalities] for _ in states]

# =========================
# build texts in STRICT atlas order
# =========================
for region_name in region_names:
    info_h = health_data[region_name]
    info_m = mild_data[region_name]
    info_s = severe_data[region_name]

    for m_idx, modality in enumerate(modalities):
        texts_by_state[0][m_idx].append(
            build_text(region_name, info_h, modality, health=True)
        )
        texts_by_state[1][m_idx].append(
            build_text(region_name, info_m, modality, health=False)
        )
        texts_by_state[2][m_idx].append(
            build_text(region_name, info_s, modality, health=False)
        )

# 严格检查长度
for s_idx, state in enumerate(states):
    for m_idx, modality in enumerate(modalities):
        assert len(texts_by_state[s_idx][m_idx]) == len(region_names), \
            f"{state}-{modality} 文本数量不对"

# 展示前几个脑区，确认顺序
print("\nFirst 5 atlas-ordered regions:")
for i in range(min(5, len(region_names))):
    print(f"{i}: {region_names[i]}")

print("\nExample text from first atlas-aligned region:")
for state_idx, state in enumerate(states):
    print(f"\n[{state}]")
    for modality_idx, modality in enumerate(modalities):
        print(f"{modality}: {texts_by_state[state_idx][modality_idx][0]}")

# =========================
# tokenize
# final shape = [5, 3, 282, 64]
# dim meaning: [modality, state, region, token_len]
# =========================
all_tokens = []

for modality_idx, modality in enumerate(modalities):
    tokens_this_modality = []
    for state_idx, state in enumerate(states):
        descriptions = texts_by_state[state_idx][modality_idx]

        texts, _ = tokenizer(
            descriptions,
            context_length=64,
            return_offsets_mapping=False
        )
        # [282, 64]
        tokens_this_modality.append(texts)

    # [3, 282, 64]
    tokens_this_modality = torch.stack(tokens_this_modality, dim=0)
    all_tokens.append(tokens_this_modality)

# [5, 3, 282, 64]
all_tokens = torch.stack(all_tokens, dim=0)

print("\nfinal token tensor shape:", all_tokens.shape)
print("dim meaning: [modality, state, region, token_len]")
print("modality order:", modalities)
print("state order:", states)
print("region order: strictly aligned to atlas_order_json keys")

torch.save(all_tokens, save_path)
print(f"\nsaved to: {save_path}")