import numpy as np  
import os
import SimpleITK as sitk  
import pandas as pd 

level5_labels = {"1": "SFG_L", "10": "MFG_DPFC_R", "100": "Midbrain_R", "101": "CST_L", "102": "CST_R", "103": "SCP_L", 
         "104": "SCP_R", "105": "MCP_L", "106": "MCP_R", "107": "PCT_L", "108": "PCT_R", "109": "ICP_L", 
         "11": "IFG_opercularis_L", "110": "ICP_R", "111": "ML_L", "112": "ML_R", "113": "Pons_L", "114": "Pons_R", 
         "115": "Medulla_L", "116": "Medulla_R", "117": "ACR_L", "118": "ACR_R", "119": "SCR_L", "12": "IFG_opercularis_R", 
         "120": "SCR_R", "121": "PCR_L", "122": "PCR_R", "123": "GCC_L", "124": "GCC_R", "125": "BCC_L", "126": "BCC_R", 
         "127": "SCC_L", "128": "SCC_R", "129": "PVWl_L", "13": "IFG_orbitalis_L", "130": "PVWl_R", "131": "ALIC_L", 
         "132": "ALIC_R", "133": "PLIC_L", "134": "PLIC_R", "135": "RLIC_L", "136": "RLIC_R", "137": "EC_L", "138": "EC_R", 
         "139": "CGC_L", "14": "IFG_orbitalis_R", "140": "CGC_R", "141": "CGH_L", "142": "CGH_R", "143": "Fx/ST_L", 
         "144": "Fx/ST_R", "145": "Fx_L", "146": "Fx_R", "147": "IFO_L", "148": "IFO_R", "149": "PTR_L", "15": "IFG_triangularis_L", 
         "150": "PTR_R", "151": "SS_L", "152": "SS_R", "153": "SFO_L", "154": "SFO_R", "155": "SLF_L", "156": "SLF_R", "157": "Cl_L", 
         "158": "Cl_R", "159": "BasalForebrain1_L", "16": "IFG_triangularis_R", "160": "BasalForebrain1_R", "161": "BasalForebrain2_L", 
         "162": "BasalForebrain2_R", "163": "BasalForebrain3_L", "164": "BasalForebrain3_R", "165": "BasalForebrain4_L", 
         "166": "BasalForebrain4_R", "167": "Mammillary_R", "168": "Mammillary_L", "169": "LV_Frontal_L", "17": "LFOG_L", "170": "LV_body_L", 
         "171": "LV_atrium_L", "172": "LV_Occipital_L", "173": "LV_Inferior_L", "174": "LV_Frontal_R", "175": "LV_body_R", "176": "LV_atrium_R", 
         "177": "LV_Occipital_R", "178": "LV_Inferior_R", "179": "III_ventricle", "18": "LFOG_R", "180": "PVWa_L", "181": "PVWa_R", "182": "PVWp_L", 
         "183": "PVWp_R", "184": "SFWM_L", "185": "SFWM_R", "186": "SFWM_PFC_L", "187": "SFWM_PFC_R", "188": "SFWM_pole_L", "189": "SFWM_pole_R", 
         "19": "MFOG_L", "190": "MFWM_L", "191": "MFWM_R", "192": "MFWM_DPFC_L", "193": "MFWM_DPFC_R", "194": "IFWM_opercularis_L", "195": "IFWM_opercularis_R", 
         "196": "IFWM_orbitalis_L", "197": "IFWM_orbitalis_R", "198": "IFWM_triangularis_L", "199": "IFWM_triangularis_R", "2": "SFG_R", "20": "MFOG_R", 
         "200": "LFOWM_L", "201": "LFOWM_R", "202": "MFOWM_L", "203": "MFOWM_R", "204": "RGWM_L", "205": "RGWM_R", "206": "PoCWM_L", "207": "PoCWM_R", 
         "208": "PrCWM_L", "209": "PrCWM_R", "21": "RG_L", "210": "SPWM_L", "211": "SPWM_R", "212": "SMWM_L", "213": "SMWM_R", "214": "AGWM_L", 
         "215": "AGWM_R", "216": "PrCuWM_L", "217": "PrCuWM_R", "218": "STWM_L", "219": "STWM_R", "22": "RG_R", "220": "STWM_L_pole", "221": "STWM_R_pole", 
         "222": "MTWM_L", "223": "MTWM_R", "224": "MTWM_L_pole", "225": "MTWM_R_pole", "226": "ITWM_L", "227": "ITWM_R", "228": "FuWM_L", "229": "FuWM_R", 
         "23": "PoCG_L", "230": "SOWM_L", "231": "SOWM_R", "232": "MOWM_L", "233": "MOWM_R", "234": "IOWM_L", "235": "IOWM_R", "236": "CuWM_L", 
         "237": "CuWM_R", "238": "LWM_L", "239": "LWM_R", "24": "PoCG_R", "240": "rostralWM_ACC_L", "241": "rostralWM_ACC_R", "242": "subcallosalWM_ACC_L", 
         "243": "subcallosalWM_ACC_R", "244": "subgenualWM_ACC_L", "245": "subgenualWM_ACC_R", "246": "dorsalWM_ACC_L", "247": "dorsalWM_ACC_R", "248": "PCCWM_L", 
         "249": "PCCWM_R", "25": "PrCG_L", "250": "CerebellumWM_R", "251": "CerebellumWM_L", "252": "MCP_cb_L", "253": "MCP_cb_R", "254": "ICP_pons_L", 
         "255": "ICP_pons_R", "256": "FrontSul_L", "257": "FrontSul_R", "258": "CentralSul_L", "259": "CentralSul_R", "26": "PrCG_R", "260": "SylFrontSul_L", 
         "261": "SylFrontSul_R", "262": "SylTempSul_L", "263": "SylTempSul_R", "264": "SylParieSul_L", "265": "SylParieSul_R", "266": "ParietSul_L", "267": "ParietSul_R", 
         "268": "CinguSul_L", "269": "CinguSul_R", "27": "SPG_L", "270": "OcciptSul_L", "271": "OcciptSul_R", "272": "TempSul_L", "273": "TempSul_R", "274": "Caudate_tail_L", 
         "275": "Fimbria_L", "276": "Caudate_tail_R", "277": "Fimbria_R", "278": "Chroid_LVetc_L", "279": "Chroid_LVetc_R", "28": "SPG_R", "280": "IV_ventricle",
         "281": "ECCL_L", "282": "ECCL_R", "29": "SMG_L", "3": "SFG_PFC_L", "30": "SMG_R", "31": "AG_L", "32": "AG_R", "33": "PrCu_L", "34": "PrCu_R", "35": "STG_L", "36": "STG_R", 
         "37": "STG_L_pole", "38": "STG_R_pole", "39": "MTG_L", "4": "SFG_PFC_R", "40": "MTG_R", "41": "MTG_L_pole", "42": "MTG_R_pole", "43": "ITG_L", "44": "ITG_R", "45": "PHG_L", 
         "46": "PHG_R", "47": "ENT_L", "48": "ENT_R", "49": "FuG_L", "5": "SFG_pole_L", "50": "FuG_R", "51": "SOG_L", "52": "SOG_R", "53": "MOG_L", "54": "MOG_R", "55": "IOG_L", 
         "56": "IOG_R", "57": "Cu_L", "58": "Cu_R", "59": "LG_L", "6": "SFG_pole_R", "60": "LG_R", "61": "rostral_ACC_L", "62": "rostral_ACC_R", "63": "subcallosal_ACC_L", 
         "64": "subcallosal_ACC_R", "65": "subgenual_ACC_L", "66": "subgenual_ACC_R", "67": "dorsal_ACC_L", "68": "dorsal_ACC_R", "69": "PCC_L", "7": "MFG_L", "70": "PCC_R", 
         "71": "Insula_L", "72": "Insula_R", "73": "Amyg_L", "74": "Amyg_R", "75": "Hippo_L", "76": "Hippo_R", "77": "Caud_L", "78": "Caud_R", "79": "Put_L", "8": "MFG_R", 
         "80": "Put_R", "81": "GP_L", "82": "GP_R", "83": "Thalamus_L", "84": "Thalamus_R", "85": "HypoThalamus_L", "86": "HypoThalamus_R", "87": "BasalForebrain_L", "88": "BasalForebrain_R", 
         "89": "NucAccumbens_L", "9": "MFG_DPFC_L", "90": "NucAccumbens_R", "91": "RedNc_L", "92": "RedNc_R", "93": "Snigra_L", "94": "Snigra_R", "95": "CerebellumGM_R", "96": "CerebellumGM_L", 
         "97": "CP_L", "98": "CP_R", "99": "Midbrain_L"}

hiera45 = [[1, 3, 5], [2, 4, 6], [9, 7], [8, 10], [11, 13, 15], [16, 12, 14], [17, 19], [18, 20], [21], [22], [23], [24], [25], [26], [27], 
           [28], [29], [30], [31], [32], [33], [34], [35, 37], [36, 38], [41, 39], [40, 42], [43], [44], [45], [46], [49], [50], [51,53,55,57,59], 
           [52,54,56,58,60], [65, 67, 69, 61, 63], [64, 66, 68, 70, 62], [71], [72], [73], [74], [275, 75], 
           [76, 277], [77, 274], [78, 276], [79], [80], [81], [82], [83], [84], [161, 163, 165, 168, 85, 87, 157, 159], [89, 160, 162, 164, 166, 167, 86, 88, 158], 
           [99, 97, 91, 93], [100, 98, 92, 94], [95], [96], [101, 103, 105, 107, 111, 113, 254], [102, 104, 106, 108, 112, 114, 255], 
           [115, 109], [116, 110], [153, 117, 119], [120, 154, 118], [121], [122], [123], [124], [125], [126], [127], [128], [129], [130], [131], 
           [132], [133], [134], [135, 137, 147, 149, 151, 281], [136, 138, 148, 150, 152, 282], [139], [140], [141], [142], [143], [144], [145], [146], 
           [206, 210, 212, 214, 216, 155], [207, 211, 213, 215, 217, 156], [169, 170, 278], [171, 172], [173], [279, 174, 175], [176, 177], [178], [179], 
           [180], [181], [182], [183], [192, 194, 196, 198, 200, 202, 204, 208, 184, 186, 188, 190], [193, 195, 197, 199, 201, 203, 205, 209, 185, 187, 189, 191], 
           [224, 226, 228, 218, 220, 222], [225, 227, 229, 219, 221, 223], [230, 232, 234, 236, 238], [231, 233, 235, 237, 239], [240, 242, 244, 246, 248], 
           [241, 243, 245, 247, 249], [250, 253], [251, 252], [280], [47], [48]]



level4_labels = {"1": "SFG_L", "2": "SFG_R", "3": "MFG_L", "4": "MFG_R", "5": "IFG_L", "6": "IFG_R", "7": "OG_L", "8": "OG_R", "9": "RG_L", "10": "RG_R", "11": "PoCG_L", 
          "12": "PoCG_R", "13": "PrCG_L", "14": "PrCG_R", "15": "SPG_L", "16": "SPG_R", "17": "SMG_L", "18": "SMG_R", "19": "AG_L", "20": "AG_R", "21": "PrCu_L", 
          "22": "PrCu_R", "23": "STG_L", "24": "STG_R", "25": "MTG_L", "26": "MTG_R", "27": "ITG_L", "28": "ITG_R", "29": "PHG_L", "30": "PHG_R", "31": "FuG_L",
            "32": "FuG_R", "33": "Occipital_L", "34": "Occipital_R", "35": "Cingulate_L", "36": "Cingulate_R", "37": "Insula_L", "38": "Insula_R", "39": "Amyg_L", 
            "40": "Amyg_R", "41": "Hippo_L", "42": "Hippo_R", "43": "Caud_L", "44": "Caud_R", "45": "Put_L", "46": "Put_R", "47": "GP_L", "48": "GP_R", "49": "Thalamus_L", "50": "Thalamus_R", "51": "BasalForbrain_L", "52": "BasalForbrain_R", "53": "midbrain_L", "54": "midbrain_R", 
            "55": "CerebellumGM_R", "56": "CerebellumGM_L", "108": "CerebellumWM_R", "109": "CerebellumWM_L", "57": "Pons_L", "58": "Pons_R", 
            "59": "Medulla_L", "60": "Medulla_R", "61": "CoreFrontalWM_L", "62": "CoreFrontalWM_R", "96": "PV_HI_front_L", "97": "PV_HI_front_R", "100": "PeripheralFrontalWM_L", "101": "PeripheralFrontalWM_R",
            "63": "CorePosteriorWM_L", "64": "CorePosteriorWM_R", "87": "PeripheralParietalWM_L", "88": "PeripheralParietalWM_R", "98": "PV_HI_posterior_L", "99": "PV_HI_posterior_R", "104": "PeripheralOccipitalWM_L", "105": "PeripheralOccipitalWM_R",
            "65": "GCC_L", "66": "GCC_R", "67": "BCC_L", "68": "BCC_R", "69": "SCC_L", "70": "SCC_R", 
            "71": "PV_HI_inferior_L", "72": "PV_HI_inferior_R", "73": "ALIC_L", "74": "ALIC_R", "75": "PLIC_L", "76": "PLIC_R", "77": "CoreInferiorWM_L", "78": "CoreInferiorWM_R", "102": "PeripheralTemporalWM_L", "103": "PeripheralTemporalWM_R",
            "79": "CGC_L", "80": "CGC_R", "81": "CGH_L", "82": "CGH_R", "83": "Fx/ST_L", "84": "Fx/ST_R", "85": "Fx_L", "86": "Fx_R", "106": "PeripheralLimbicWM_L", "107": "PeripheralLimbicWM_R",
            "89": "FrontalLateralVentricle_L", "90": "PosteriorLateralVentricle_L", "91": "InferiorLateralVentricle_L", "92": "FrontalLateralVentricle_R", "93": "PosteriorLateralVentricle_R", "94": "InferiorLateralVentricle_R", 
            "95": "III_ventricle", "110": "IV_ventricle", "111":'ENT_L', '112':'ENT_R'}

hiera34 = [[1, 3, 5, 7, 9, 13], [2, 4, 6, 8, 10, 14], [11, 15, 17, 19, 21], [12, 16, 18, 20, 22], [23, 25, 27, 31], [24, 26, 32, 28], [29, 35, 39, 41, 111], [30, 36, 40, 42, 112], [33], [34],
           [37], [38], [43,45,47], [44,46,48], [49], [50], [51], [52], [53], [54], [55, 108], [56, 109], [57], [58], [59], [60], [61, 96, 100], [62, 97, 101],
           [63, 87, 104], [64, 88, 105], [65, 67, 69], [66, 68, 70], [71, 73, 75, 77, 98, 102], [72, 74, 76, 78, 99, 103], [79, 81, 83, 85, 106], [80, 82, 84, 86, 107], 
           [89, 90, 91], [92, 93, 94], [95],  [110]]

level3_labels = {"1": "Frontal_L", "2": "Frontal_R", "3": "Parietal_L", "4": "Parietal_R", "5": "Temporal_L", "6": "Temporal_R", "7": "Limbic_L", "8": "Limbic_R", "9": "Occipital_L", 
          "10": "Occipital_R", "11": "Insula_L", "12": "Insula_R", "13": "BasalGang_L", "14": "BasalGang_R", "15": "Thalamus_L", "16": "Thalamus_R", "17": "BasalForbrain_L", 
          "18": "BasalForbrain_R", "19": "midbrain_L", "20": "midbrain_R", "21": "Cerebellum_R", "22": "Cerebellum_L", "23": "Pons_L", "24": "Pons_R", "25": "Medulla_L", "26": "Medulla_R", 
          "27": "FrontalWM_L", "28": "FrontalWM_R", "29": "PosteriorWM_L", "30": "PosteriorWM_R", "31": "CorpusCallosum_L", "32": "CorpusCallosum_R", 
          "33": "InferiorWM_L", "34": "InferiorWM_R", "35": "LimbicWM_L", "36": "LimbicWM_R", "37": "LateralVentricle_L", "38": "LateralVentricle_R", "39": "III_ventricle", 
          "40": "IV_ventricle"}


level2_labels = {"1": "CerebralCortex_L", "2": "CerebralCortex_R", "3": "CerebralNucli_L", "4": "CerebralNucli_R", "5": "Thalamus_L", "6": "Thalamus_R", "7": "BasalForbrain_L", 
          "8": "BasalForbrain_R", "9": "Mesencephalon_L", "10": "Mesencephalon_R", "11": "Metencephalon_L", "12": "Metencephalon_R", "13":"Myelencephalon_L", "14":"Myelencephalon_R", "15": "WhiteMatter_L", "16": "WhiteMatter_R", 
          "15": "LateralVentricle", "16":"IIIandIV_Ventricle"}

hiera23 = [[1, 3, 5, 7, 9, 11], [2, 4, 6, 8, 10, 12], [13], [14], [15], [16], [17], [18], [19], [20], [22, 23], [21, 24], [25],  [26], 
           [27, 29, 31, 33, 35], [28, 30, 32, 34, 36], [37, 38], [39, 40]]


hiera12 = [[1, 3, 15], [2, 4, 16], [5, 7], [6, 8], [9], [10], [11], [12], [13], [14], [17, 18]]

level5_map = sitk.ReadImage('C:/Users/SUST/Desktop/Brain_CLIP/standard/JHU5Levels/Deform_MNI_AltasWarped.nii.gz')

img_ori = level5_map.GetOrigin()
img_spa = level5_map.GetSpacing()
img_dir = level5_map.GetDirection()
level5_arr = sitk.GetArrayFromImage(level5_map)

level1_arr = np.zeros_like(level5_arr)
level2_arr = np.zeros_like(level5_arr)
level3_arr = np.zeros_like(level5_arr)
level4_arr = np.zeros_like(level5_arr)

for idx,group in enumerate(hiera45):
    for num in group:
        level4_arr[np.where(level5_arr==num)] = idx+1


for idx,group in enumerate(hiera34):
    for num in group:
        level3_arr[np.where(level4_arr==num)] = idx+1
level3_arr[np.where(level5_arr==89)] = 13
level3_arr[np.where(level5_arr==90)] = 14

for idx,group in enumerate(hiera23):
    for num in group:
        level2_arr[np.where(level3_arr==num)] = idx+1
level2_arr[np.where(level5_arr==73)] = 3
level2_arr[np.where(level5_arr==74)] = 4

for idx,group in enumerate(hiera12):
    for num in group:
        level1_arr[np.where(level2_arr==num)] = idx+1

level4_map = sitk.GetImageFromArray(level4_arr)
level4_map.SetOrigin(img_ori)
level4_map.SetSpacing(img_spa)
level4_map.SetDirection(img_dir)
sitk.WriteImage(level4_map,'C:/Users/SUST/Desktop/Brain_CLIP/standard/JHU5Levels/Deform_MNI_AltasL4WarpedFuben.nii.gz')

level3_map = sitk.GetImageFromArray(level3_arr)
level3_map.SetOrigin(img_ori)
level3_map.SetSpacing(img_spa)
level3_map.SetDirection(img_dir)
sitk.WriteImage(level3_map,'C:/Users/SUST/Desktop/Brain_CLIP/standard/JHU5Levels/Deform_MNI_AltasL3WarpedFuben.nii.gz')

level2_map = sitk.GetImageFromArray(level2_arr)
level2_map.SetOrigin(img_ori)
level2_map.SetSpacing(img_spa)
level2_map.SetDirection(img_dir)
sitk.WriteImage(level2_map,'C:/Users/SUST/Desktop/Brain_CLIP/standard/JHU5Levels/Deform_MNI_AltasL2WarpedFuben.nii.gz')

level1_map = sitk.GetImageFromArray(level1_arr)
level1_map.SetOrigin(img_ori)
level1_map.SetSpacing(img_spa)
level1_map.SetDirection(img_dir)
sitk.WriteImage(level1_map,'C:/Users/SUST/Desktop/Brain_CLIP/standard/JHU5Levels/Deform_MNI_AltasL1WarpedFuben.nii.gz')

level1_labels = {
    "1": "Left Telencephalon",      # 端脑左侧
    "2": "Right Telencephalon",     # 端脑右侧
    "3": "Left Diencephalon",      # 间脑左侧
    "4": "Right Diencephalon",     # 间脑右侧
    "5": "Left Mesencephalon",               
    "6": "Right Mesencephalon",  
    "7": "Left Metencephalon",
    "8": "Right Metencephalon",
    "9": "Left Myelencephalon",
    "10": "Right Myelencephalon",      
    "11": "Cerebrospinal Fluid"      # 脑脊液
}

full_name_level2  = {
    "1": "Left Cerebral Cortex",            # 左侧大脑皮层
    "2": "Right Cerebral Cortex",           # 右侧大脑皮层
    "3": "Left Cerebral Nuclei",          # 左侧基底核（注："Nucli" 拼写应为 "Nuclei"）
    "4": "Right Cerebral Nuclei",         # 右侧基底核
    "5": "Left Thalamus",                 # 左侧丘脑
    "6": "Right Thalamus",                # 右侧丘脑
    "7": "Left Basal Forebrain",             # 左侧基底节（原 "BasalForbrain" 拼写调整）
    "8": "Right Basal Forebrain",            # 右侧基底节
    "9": "Left Mesencephalon",                      # 中脑（无左右之分，建议删除 "_L"）
    "10": "Right Mesencephalon",                     # 中脑（重复条目，建议合并）
    "11": "Left Metencephalon",               # 左侧小脑（注：小脑通常不分左右半球，此处可能指小脑前/后叶）
    "12": "Right Metencephalon",              # 右侧小脑
    "13": "Left Myelencephalon",
    "14": "Right Myelencephalon",
    "15": "Left White Matter",            # 左侧白质
    "16": "Right White Matter",           # 右侧白质
    "17": "Lateral Ventricle",                    # 脑室（双侧脑室系统）
    "18": "Third and Fourth Ventricle"
}

full_name_level3 = {
    "1": "Left Frontal Lobe",            # 额叶左侧
    "2": "Right Frontal Lobe",           # 额叶右侧
    "3": "Left Parietal Lobe",          # 顶叶左侧
    "4": "Right Parietal Lobe",         # 顶叶右侧
    "5": "Left Temporal Lobe",          # 颞叶左侧
    "6": "Right Temporal Lobe",         # 颞叶右侧
    "7": "Left Limbic System",          # 边缘系统左侧（包含杏仁核、海马等）
    "8": "Right Limbic System",         # 边缘系统右侧
    "9": "Left Occipital Lobe",           # 枕叶左侧
    "10": "Right Occipital Lobe",        # 枕叶右侧
    "11": "Left Insula",                 # 岛叶左侧
    "12": "Right Insula",                # 岛叶右侧
    "13": "Left Basal Ganglia",          # 左侧基底核（壳核、尾状核等）
    "14": "Right Basal Ganglia",         # 右侧基底核
    "15": "Left Thalamus",                # 左侧丘脑
    "16": "Right Thalamus",               # 右侧丘脑
    "17": "Left Basal Forebrain",         # 左侧基底前脑（如伏隔核、杏仁核）
    "18": "Right Basal Forebrain",        # 右侧基底前脑
    "19": "Left Midbrain",                    # 中脑（单侧结构，无左右标记，建议删除_L）
    "20": "Right Midbrain",                    # 中脑（重复条目，合并）
    "21": "Right Cerebellum",             # 小脑右侧（注：小脑通常不分左右半球，此处可能指小脑前/后叶）
    "22": "Left Cerebellum",              # 小脑左侧
    "23": "Left Pons",                   # 脑桥左侧
    "24": "Right Pons",                  # 脑桥右侧
    "25": "Left Medulla Oblongata",      # 延髓左侧
    "26": "Right Medulla Oblongata",     # 延髓右侧
    "27": "Left Frontal White Matter",   # 额叶白质左侧
    "28": "Right Frontal White Matter",  # 额叶白质右侧
    "29": "Left Posterior White Matter",  # 后部白质左侧（顶枕叶白质）
    "30": "Right Posterior White Matter", # 后部白质右侧
    "31": "Left Corpus Callosum",          # 左侧胼胝体
    "32": "Right Corpus Callosum",         # 右侧胼胝体
    "33": "Left Inferior White Matter",   # 下部白质左侧（可能包含脑干旁白质）
    "34": "Right Inferior White Matter",  # 下部白质右侧
    "35": "Left Limbic White Matter",    # 边缘系统白质左侧（如扣带束）
    "36": "Right Limbic White Matter",   # 边缘系统白质右侧
    "37": "Left Lateral Ventricle",       # 左侧侧脑室
    "38": "Right Lateral Ventricle",      # 右侧侧脑室
    "39": "Third Ventricle",             # 第三脑室
    "40": "Fourth Ventricle",            # 第四脑室
}

full_name_level4 = {

    "1": "Left Superior Frontal Gyrus (Posterior Segment)",  # SFG_L
    "2": "Right Superior Frontal Gyrus (Posterior Segment)", # SFG_R
    "3": "Left Middle Frontal Gyrus (Dorsal Prefrontal Cortex)", # MFG_L
    "4": "Right Middle Frontal Gyrus (Dorsal Prefrontal Cortex)", # MFG_R
    "5": "Left Inferior Frontal Gyrus",                        # IFG_L
    "6": "Right Inferior Frontal Gyrus",                       # IFG_R
    "7": "Left Orbital Frontal Gyrus",                         # OG_L
    "8": "Right Orbital Frontal Gyrus",                        # OG_R
    "9": "Left Rectus Gyrus",                                  # RG_L
    "10": "Right Rectus Gyrus",                                 # RG_R
    "11": "Left Postcentral Gyrus",                             # PoCG_L
    "12": "Right Postcentral Gyrus",                            # PoCG_R
    "13": "Left Precentral Gyrus",                             # PrCG_L
    "14": "Right Precentral Gyrus",                            # PrCG_R
    "15": "Left Superior Parietal Gyrus",                        # SPG_L
    "16": "Right Superior Parietal Gyrus",                       # SPG_R
    "17": "Left Supramarginal Gyrus",                            # SMG_L
    "18": "Right Supramarginal Gyrus",                           # SMG_R
    "19": "Left Angular Gyrus",                                 # AG_L
    "20": "Right Angular Gyrus",                                # AG_R
    "21": "Left Precuneus",                                      # PrCu_L
    "22": "Right Precuneus",                                     # PrCu_R
    "23": "Left Superior Temporal Gyrus",                         # STG_L
    "24": "Right Superior Temporal Gyrus",                       # STG_R
    "25": "Left Middle Temporal Gyrus",                            # MTG_L
    "26": "Right Middle Temporal Gyrus",                           # MTG_R
    "27": "Left Inferior Temporal Gyrus",                         # ITG_L
    "28": "Right Inferior Temporal Gyrus",                       # ITG_R
    "29": "Left Parahippocampal Gyrus",                                  
    "30": "Right Parahippocampal Gyrus",                                 
    "31": "Left Fusiform gyrus",                                 
    "32": "Right Fusiform gyrus",                                
    "33": "Left Occipital Lobe",                                 # Occipital_L
    "34": "Right Occipital Lobe",                                # Occipital_R
    "35": "Left Cingulate Gyrus",                                 # Cingulate_L
    "36": "Right Cingulate Gyrus",                                # Cingulate_R
    "37": "Left Insula",                                         # Insula_L
    "38": "Right Insula",                                        # Insula_R
    "39": "Left Amygdala",                                      # Amyg_L
    "40": "Right Amygdala",                                     # Amyg_R
    "41": "Left Hippocampus",                                    # Hippo_L
    "42": "Right Hippocampus",                                   # Hippo_R
    "43": "Left Caudate Nucleus",                                # Caud_L
    "44": "Right Caudate Nucleus",                               # Caud_R
    "45": "Left Putamen",                                       # Put_L
    "46": "Right Putamen",                                      # Put_R
    "47": "Left Globus Pallidus",                                # GP_L
    "48": "Right Globus Pallidus",                               # GP_R
    "49": "Left Thalamus",                                      # Thalamus_L
    "50": "Right Thalamus",                                     # Thalamus_R
    "51": "Left Basal Forebrain",                                 # BasalForbrain_L
    "52": "Right Basal Forebrain",                                # BasalForbrain_R
    "53": "Left Midbrain",                                           # midbrain_L (单侧结构，建议删除_L)
    "54": "Right Midbrain",                                           # midbrain_R (重复条目，合并)
    "55": "Right Cerebellum Gray Matter",                         # CerebellumGM_R
    "56": "Left Cerebellum Gray Matter",                         # CerebellumGM_L
    "57": "Left Pons",                                           # Pons_L
    "58": "Right Pons",                                          # Pons_R
    "59": "Left Medulla Oblongata",                              # Medulla_L
    "60": "Right Medulla Oblongata",                             # Medulla_R
    "61": "Left Core Frontal White Matter",                        # CoreFrontalWM_L
    "62": "Right Core Frontal White Matter",                       # CoreFrontalWM_R
    "63": "Left Core Posterior White Matter",                     # CorePosteriorWM_L
    "64": "Right Core Posterior White Matter",                    # CorePosteriorWM_R
    "65": "Left Genu of Corpus Callosum",                         # GCC_L
    "66": "Right Genu of Corpus Callosum",                        # GCC_R
    "67": "Left Body of Corpus Callosum",                         # BCC_L
    "68": "Right Body of Corpus Callosum",                        # BCC_R
    "69": "Left Splenium of Corpus Callosum",                       # SCC_L
    "70": "Right Splenium of Corpus Callosum",                      # SCC_R
    "71": "Left Retrolenticular Part of Internal Capsule",         # PV_HI_inferior_L
    "72": "Right Retrolenticular Part of Internal Capsule",        # PV_HI_inferior_R
    "73": "Left Anterior Limb of Internal Capsule",              # ALIC_L
    "74": "Right Anterior Limb of Internal Capsule",             # ALIC_R
    "75": "Left Posterior Limb of Internal Capsule",             # PLIC_L
    "76": "Right Posterior Limb of Internal Capsule",            # PLIC_R
    "77": "Left Core Inferior White Matter",                       # CoreInferiorWM_L
    "78": "Right Core Inferior White Matter",                      # CoreInferiorWM_R
    "79": "Left Cingulum (Cingulate Gyrus)",                     # CGC_L
    "80": "Right Cingulum (Cingulate Gyrus)",                    # CGC_R
    "81": "Left Cingulum (Hippocampus)",                         # CGH_L
    "82": "Right Cingulum (Hippocampus)",                        # CGH_R
    "83": "Left Fornix/Caudate Tail",                            # Fx/ST_L
    "84": "Right Fornix/Caudate Tail",                           # Fx/ST_R
    "85": "Left Fornix (Column and Body)",                        # Fx_L
    "86": "Right Fornix (Column and Body)",                       # Fx_R
    "87": "Left Peripheral Parietal White Matter",                  # PeripheralParietalWM_L
    "88": "Right Peripheral Parietal White Matter",                 # PeripheralParietalWM_R
    "89": "Left Frontal Lateral Ventricle",                       # FrontalLateralVentricle_L
    "90": "Left Posterior Lateral Ventricle",                     # PosteriorLateralVentricle_L
    "91": "Left Inferior Lateral Ventricle",                      # InferiorLateralVentricle_L
    "92": "Right Frontal Lateral Ventricle",                      # FrontalLateralVentricle_R
    "93": "Right Posterior Lateral Ventricle",                    # PosteriorLateralVentricle_R
    "94": "Right Inferior Lateral Ventricle",                     # InferiorLateralVentricle_R
    "95": "Third Ventricle",                                     # III_ventricle
    "96": "Left Periventricular Hypothalamic Frontal White Matter", # PV_HI_front_L
    "97": "Right Periventricular Hypothalamic Frontal White Matter", # PV_HI_front_R
    "98": "Left Periventricular Hypothalamic Posterior White Matter", # PV_HI_posterior_L
    "99": "Right Periventricular Hypothalamic Posterior White Matter", # PV_HI_posterior_R
    "100": "Left Peripheral Frontal White Matter",                  # PeripheralFrontalWM_L
    "101": "Right Peripheral Frontal White Matter",                 # PeripheralFrontalWM_R
    "102": "Left Peripheral Temporal White Matter",                 # PeripheralTemporalWM_L
    "103": "Right Peripheral Temporal White Matter",                # PeripheralTemporalWM_R
    "104": "Left Peripheral Occipital White Matter",                 # PeripheralOccipitalWM_L
    "105": "Right Peripheral Occipital White Matter",                # PeripheralOccipitalWM_R
    "106": "Left Peripheral Limbic White Matter",                  # PeripheralLimbicWM_L
    "107": "Right Peripheral Limbic White Matter",                 # PeripheralLimbicWM_R
    "108": "Right Cerebellum White Matter",                        # CerebellumWM_R
    "109": "Left Cerebellum White Matter",                        # CerebellumWM_L
    "110": "Fourth Ventricle",                                    # IV_ventricle
    "111": "Left Nucleus Accumbens",                              # NucAccumbens_L
    "112": "Right Nucleus Accumbens",                             # NucAccumbens_R
}

full_name_level5 = {
    # **皮层结构 (Cortical Structures)**
    "1": "Left Superior Frontal Gyrus (Posterior Segment)",  # SFG_L
    "10": "Right Middle Frontal Gyrus (Dorsal Prefrontal Cortex)", # MFG_DPFC_R
    "100": "Right Midbrain",                                           # Midbrain_R (单侧结构，建议删除_R)
    "101": "Left Corticospinal Tract",                            # CST_L
    "102": "Right Corticospinal Tract",                           # CST_R
    "103": "Left Superior Colliculus",                            # SCP_L
    "104": "Right Superior Colliculus",                           # SCP_R
    "105": "Left Middle Cerebral Artery Territory",               # MCP_L (基于血管分区命名)
    "106": "Right Middle Cerebral Artery Territory",              # MCP_R
    "107": "Left Posterior Cerebral Artery Territory",            # PCT_L
    "108": "Right Posterior Cerebral Artery Territory",           # PCT_R
    "109": "Left Internal Capsule",                              # ICP_L
    "11": "Left Inferior Frontal Gyrus (Opercularis Part)",       # IFG_opercularis_L
    "110": "Right Internal Capsule",                             # ICP_R
    "111": "Left Medial Longitudinal Fasciculus",                  # ML_L
    "112": "Right Medial Longitudinal Fasciculus",                 # ML_R
    "113": "Left Pons",                                           # Pons_L
    "114": "Right Pons",                                          # Pons_R
    "115": "Left Medulla Oblongata",                              # Medulla_L
    "116": "Right Medulla Oblongata",                             # Medulla_R
    "117": "Left Anterior Cingulate gyrus (Acr)",                   # ACR_L
    "118": "Right Anterior Cingulate gyrus (Acr)",                  # ACR_R
    "119": "Left Posterior Cingulate gyrus (Scr)",                  # SCR_L
    "12": "Right Inferior Frontal Gyrus (Opercularis Part)",      # IFG_opercularis_R
    "120": "Right Posterior Cingulate gyrus (Scr)",                 # SCR_R
    "121": "Left Paracentral Lobule (PCR)",                        # PCR_L
    "122": "Right Paracentral Lobule (PCR)",                       # PCR_R
    "123": "Left Genu of Corpus Callosum",                         # GCC_L
    "124": "Right Genu of Corpus Callosum",                        # GCC_R
    "125": "Left Body of Corpus Callosum",                         # BCC_L
    "126": "Right Body of Corpus Callosum",                        # BCC_R
    "127": "Left Splenium of Corpus Callosum",                       # SCC_L
    "128": "Right Splenium of Corpus Callosum",                      # SCC_R
    "129": "Left Periventricular White Matter Inferior (PVWl)",             # PVWl_L (拼写修正为PVW)
    "13": "Left Inferior Frontal Gyrus (Orbitalis Part)",          # IFG_orbitalis_L
    "130": "Right Periventricular White Matter Inferior (PVWl)",            # PVWl_R (拼写修正为PVW)
    "131": "Left Anterior Limb of Internal Capsule",              # ALIC_L
    "132": "Right Anterior Limb of Internal Capsule",             # ALIC_R
    "133": "Left Posterior Limb of Internal Capsule",             # PLIC_L
    "134": "Right Posterior Limb of Internal Capsule",            # PLIC_R
    "135": "Left Retrolenticular Part of Internal Capsule",         # RLIC_L
    "136": "Right Retrolenticular Part of Internal Capsule",        # RLIC_R
    "137": "Left External Capsule",                              # EC_L
    "138": "Right External Capsule",                             # EC_R
    "139": "Left Cingulum (Cingulate Gyrus)",                     # CGC_L
    "14": "Right Inferior Frontal Gyrus (Orbitalis Part)",         # IFG_orbitalis_R
    "140": "Right Cingulum (Cingulate Gyrus)",                    # CGC_R
    "141": "Left Cingulum (Hippocampal Part)",                     # CGH_L
    "142": "Right Cingulum (Hippocampal Part)",                    # CGH_R
    "143": "Left Fornix/Caudate Tail",                            # Fx/ST_L
    "144": "Right Fornix/Caudate Tail",                           # Fx/ST_R
    "145": "Left Fornix (Column and Body)",                        # Fx_L
    "146": "Right Fornix (Column and Body)",                       # Fx_R
    "147": "Left Inferior Frontal Orbital gyrus",                  # IFO_L
    "148": "Right Inferior Frontal Orbital gyrus",                 # IFO_R
    "149": "Left Putamen",                                       # PTR_L
    "15": "Left Inferior Frontal Gyrus (Triangularis Part)",       # IFG_triangularis_L
    "150": "Right Putamen",                                      # PTR_R
    "151": "Left Superior Temporal Sulcus",                       # SS_L
    "152": "Right Superior Temporal Sulcus",                      # SS_R
    "153": "Left Subolfactory Cortex",                             # SFO_L
    "154": "Right Subolfactory Cortex",                            # SFO_R
    "155": "Left Superior Longitudinal Fasciculus",                # SLF_L
    "156": "Right Superior Longitudinal Fasciculus",               # SLF_R
    "157": "Left Claustrum",                                       # Cl_L
    "158": "Right Claustrum",                                      # Cl_R
    "159": "Left Ansa Lenticularis",                      # BasalForebrain1_L
    "16": "Right Inferior Frontal Gyrus (Triangularis Part)",      # IFG_triangularis_R
    "160": "Right Ansa Lenticularis",                     # BasalForebrain1_R
    "161": "Left Anterior Commissure",                      # BasalForebrain2_L
    "162": "Right Anterior Commissure",                     # BasalForebrain2_R
    "163": "Left Lenticular Fasciculus",                      # BasalForebrain3_L
    "164": "Right Lenticular Fasciculus",                     # BasalForebrain3_R
    "165": "Left Olfactory Radiation",                      # BasalForebrain4_L
    "166": "Right Olfactory Radiation",                     # BasalForebrain4_R
    "167": "Right Mammillary Body",                                # Mammillary_R
    "168": "Left Mammillary Body",                                 # Mammillary_L
    "169": "Left Lateral Ventricular Frontal Horn",                # LV_Frontal_L
    "17": "Left Orbital Frontal Gyrus",                            # LFOG_L
    "170": "Left Lateral Ventricular Body",                        # LV_body_L
    "171": "Left Lateral Ventricular Atrium",                     # LV_atrium_L
    "172": "Left Lateral Ventricular Occipital Horn",               # LV_Occipital_L
    "173": "Left Lateral Ventricular Inferior Horn",                # LV_Inferior_L
    "174": "Right Lateral Ventricular Frontal Horn",               # LV_Frontal_R
    "175": "Right Lateral Ventricular Body",                        # LV_body_R
    "176": "Right Lateral Ventricular Atrium",                     # LV_atrium_R
    "177": "Right Lateral Ventricular Occipital Horn",              # LV_Occipital_R
    "178": "Right Lateral Ventricular Inferior Horn",               # LV_Inferior_R
    "179": "Third Ventricle",                                     # III_ventricle
    "18": "Right Orbital Frontal Gyrus",                           # LFOG_R
    "180": "Left Periventricular White Matter Frontal (PVWa)",              # PVWa_L
    "181": "Right Periventricular White Matter Frontal (PVWa)",             # PVWa_R
    "182": "Left Periventricular White Matter Posterior (PVWp)",              # PVWp_L
    "183": "Right Periventricular White Matter Posterior (PVWp)",             # PVWp_R
    "184": "Left Superficial Frontal White Matter",                # SFWM_L
    "185": "Right Superficial Frontal White Matter",               # SFWM_R
    "186": "Left Superficial Frontal White Matter (PFC)",          # SFWM_PFC_L
    "187": "Right Superficial Frontal White Matter (PFC)",         # SFWM_PFC_R
    "188": "Left Superficial Frontal White Matter (Pole)",         # SFWM_pole_L
    "189": "Right Superficial Frontal White Matter (Pole)",        # SFWM_pole_R
    "19": "Left Middle Frontal Gyrus (Dorsal Prefrontal Cortex)",   # MFOG_L
    "190": "Left Middle Frontal White Matter",                    # MFWM_L
    "191": "Right Middle Frontal White Matter",                    # MFWM_R
    "192": "Right Superficial Frontal White Matter (DPFC)",       # MFWM_DPFC_R
    "193": "Left Superficial Frontal White Matter (DPFC)",        # MFWM_DPFC_L
    "194": "Left Inferior Frontal White Matter (Opercularis)",      # IFWM_opercularis_L
    "195": "Right Inferior Frontal White Matter (Opercularis)",     # IFWM_opercularis_R
    "196": "Left Inferior Frontal White Matter (Orbitalis)",       # IFWM_orbitalis_L
    "197": "Right Inferior Frontal White Matter (Orbitalis)",      # IFWM_orbitalis_R
    "198": "Left Inferior Frontal White Matter (Triangularis)",     # IFWM_triangularis_L
    "199": "Right Inferior Frontal White Matter (Triangularis)",    # IFWM_triangularis_R
    "2": "Right Superior Frontal Gyrus (Posterior Segment)",        # SFG_R
    "20": "Right Middle Frontal Gyrus (Dorsal Prefrontal Cortex)",   # MFOG_R
    "200": "Left Lateral Frontal White Matter",                    # LFOWM_L
    "201": "Right Lateral Frontal White Matter",                    # LFOWM_R
    "202": "Left Middle Frontal White Matter",                    # MFOWM_L
    "203": "Right Middle Frontal White Matter",                    # MFOWM_R
    "204": "Left Rolandic Gyrus",                                 # RGWM_L
    "205": "Right Rolandic Gyrus",                                # RGWM_R
    "206": "Left Postcentral Gyrus White Matter",                  # PoCWM_L
    "207": "Right Postcentral Gyrus White Matter",                 # PoCWM_R
    "208": "Left Precentral Gyrus White Matter",                    # PrCWM_L
    "209": "Right Precentral Gyrus White Matter",                 # PrCWM_R
    "21": "Left Rolandic Gyrus",                                 # RG_L
    "210": "Left Superior Temporal White Matter",                   # SPWM_L
    "211": "Right Superior Temporal White Matter",                  # SPWM_R
    "212": "Left Middle Temporal White Matter",                    # SMWM_L
    "213": "Right Middle Temporal White Matter",                    # SMWM_R
    "214": "Left Angular Gyrus White Matter",                     # AGWM_L
    "215": "Right Angular Gyrus White Matter",                    # AGWM_R
    "216": "Left Precuneus White Matter",                          # PrCuWM_L
    "217": "Right Precuneus White Matter",                         # PrCuWM_R
    "218": "Left Superior Temporal White Matter (Pole)",           # STWM_L
    "219": "Right Superior Temporal White Matter (Pole)",          # STWM_R
    "220": "Left Superior Temporal White Matter (Pole)",           # STWM_L_pole
    "221": "Right Superior Temporal White Matter (Pole)",          # STWM_R_pole
    "222": "Left Middle Temporal White Matter (Pole)",             # MTWM_L
    "223": "Right Middle Temporal White Matter (Pole)",            # MTWM_R
    "224": "Left Middle Temporal White Matter (Pole)",             # MTWM_L_pole
    "225": "Right Middle Temporal White Matter (Pole)",            # MTWM_R_pole
    "226": "Left Inferior Temporal White Matter",                  # ITWM_L
    "227": "Right Inferior Temporal White Matter",                 # ITWM_R
    "228": "Left Fusiform gyrus White Matter",                     # FuWM_L
    "229": "Right Fusiform gyrus White Matter",                    # FuWM_R
    "23": "Left Postcentral Gyrus",                             # PoCG_L
    "230": "Left Superior Occipital White Matter",                 # SOWM_L
    "231": "Right Superior Occipital White Matter",                # SOWM_R
    "232": "Left Middle Occipital White Matter",                   # MOWM_L
    "233": "Right Middle Occipital White Matter",                  # MOWM_R
    "234": "Left Inferior Occipital White Matter",                  # IOWM_L
    "235": "Right Inferior Occipital White Matter",                 # IOWM_R
    "236": "Left Cuneate Gyrus White Matter",                       # CuWM_L
    "237": "Right Cuneate Gyrus White Matter",                      # CuWM_R
    "238": "Left Lateral Occipital White Matter",                  # LWM_L
    "239": "Right Lateral Occipital White Matter",                 # LWM_R
    "24": "Right Postcentral Gyrus",                             # PoCG_R
    "240": "Left Rostral ACC White Matter",                        # rostralWM_ACC_L
    "241": "Right Rostral ACC White Matter",                       # rostralWM_ACC_R
    "242": "Left Subcallosal ACC White Matter",                    # subcallosalWM_ACC_L
    "243": "Right Subcallosal ACC White Matter",                   # subcallosalWM_ACC_R
    "244": "Left Subgenual ACC White Matter",                       # subgenualWM_ACC_L
    "245": "Right Subgenual ACC White Matter",                      # subgenualWM_ACC_R
    "246": "Left Dorsal ACC White Matter",                         # dorsalWM_ACC_L
    "247": "Right Dorsal ACC White Matter",                        # dorsalWM_ACC_R
    "248": "Left Posterior Cingulate White Matter",                # PCCWM_L
    "249": "Right Posterior Cingulate White Matter",               # PCCWM_R
    "25": "Left Precentral Gyrus",                             # PrCG_L
    "250": "Right Cerebellum White Matter",                        # CerebellumWM_R
    "251": "Left Cerebellum White Matter",                        # CerebellumWM_L
    "252": "Left Middle Cerebellar Peduncle",                     # MCP_cb_L
    "253": "Right Middle Cerebellar Peduncle",                    # MCP_cb_R
    "254": "Left Superior Cerebellar Peduncle (pons part)",        # ICP_pons_L
    "255": "Right Superior Cerebellar Peduncle (pons part)",       # ICP_pons_R
    "256": "Left Frontal Sulcus",                                # FrontSul_L
    "257": "Right Frontal Sulcus",                               # FrontSul_R
    "258": "Left Central Sulcus",                                 # CentralSul_L
    "259": "Right Central Sulcus",                               # CentralSul_R
    "26": "Right Precentral Gyrus",                             # PrCG_R
    "260": "Left Sylvian Fissure (Frontal Part)",                  # SylFrontSul_L
    "261": "Right Sylvian Fissure (Frontal Part)",                 # SylFrontSul_R
    "262": "Left Sylvian Fissure (Temporal Part)",                  # SylTempSul_L
    "263": "Right Sylvian Fissure (Temporal Part)",                 # SylTempSul_R
    "264": "Left Sylvian Fissure (Parietal Part)",                  # SylParieSul_L
    "265": "Right Sylvian Fissure (Parietal Part)",                 # SylParieSul_R
    "266": "Left Parietal Sulcus",                                # ParietSul_L
    "267": "Right Parietal Sulcus",                               # ParietSul_R
    "268": "Left Cingulate Sulcus",                                # CinguSul_L
    "269": "Right Cingulate Sulcus",                               # CingulateSul_R
    "27": "Left Superior Parietal Gyrus",                         # SPG_L
    "270": "Left Occipital Sulcus",                                # OcciptSul_L
    "271": "Right Occipital Sulcus",                               # OcciptSul_R
    "272": "Left Temporal Sulcus",                                # TempSul_L
    "273": "Right Temporal Sulcus",                               # TempSul_R
    "274": "Left Caudate Tail",                                   # Caudate_tail_L
    "275": "Left Fimbria",                                        # Fimbria_L
    "276": "Right Caudate Tail",                                   # Caudate_tail_R
    "277": "Right Fimbria",                                        # Fimbria_R
    "278": "Left Choroid Plexus (LV etc.)",                        # Chroid_LVetc_L
    "279": "Right Choroid Plexus (LV etc.)",                       # Chroid_LVetc_R
    "28": "Right Superior Parietal Gyrus",                         # SPG_R
    "280": "Fourth Ventricle",                                    # IV_ventricle
    "281": "Left External Capsule (ECCL)",                         # ECCL_L
    "282": "Right External Capsule (ECCL)",                        # ECCL_R
    "29": "Left Supramarginal Gyrus",                            # SMG_L
    "3": "Left Superior Frontal Gyrus (Prefrontal Cortex)",         # SFG_PFC_L
    "30": "Left Supramarginal Gyrus",                            # SMG_R
    "31": "Left Angular Gyrus",                                 # AG_L
    "32": "Right Angular Gyrus",                                # AG_R
    "33": "Left Precuneus",                                      # PrCu_L
    "34": "Right Precuneus",                                     # PrCu_R
    "35": "Left Superior Temporal Gyrus",                         # STG_L
    "36": "Right Superior Temporal Gyrus",                       # STG_R
    "37": "Left Superior Temporal Gyrus (Pole)",                   # STG_L_pole
    "38": "Right Superior Temporal Gyrus (Pole)",                  # STG_R_pole
    "39": "Left Middle Temporal Gyrus",                            # MTG_L
    "4": "Left Superior Frontal Gyrus (Prefrontal Cortex)",         # SFG_PFC_R
    "40": "Right Middle Temporal Gyrus",                           # MTG_R
    "41": "Left Middle Temporal Gyrus (Pole)",                    # MTG_L_pole
    "42": "Right Middle Temporal Gyrus (Pole)",                    # MTG_R_pole
    "43": "Left Inferior Temporal Gyrus",                         # ITG_L
    "44": "Right Inferior Temporal Gyrus",                       # ITG_R
    "45": "Left Parahippocampal Gyrus",                            # PHG_L
    "46": "Right Parahippocampal Gyrus",                           # PHG_R
    "47": "Left Entorhinal Cortex",                                # ENT_L
    "48": "Right Entorhinal Cortex",                               # ENT_R
    "49": "Left Fusiform gyrus",                                 # FuG_L
    "5": "Left Superior Frontal Gyrus (Pole)",                     # SFG_pole_L
    "50": "Right Fusiform gyrus",                                # FuG_R
    "51": "Left Superior Occipital Gyrus",                         # SOG_L
    "52": "Right Superior Occipital Gyrus",                       # SOG_R
    "53": "Left Middle Occipital Gyrus",                         # MOG_L
    "54": "Right Middle Occipital Gyrus",                       # MOG_R
    "55": "Left Inferior Occipital Gyrus",                        # IOG_L
    "56": "Right Inferior Occipital Gyrus",                       # IOG_R
    "57": "Left Lingual Gyrus",                                  # Cu_L
    "58": "Right Lingual Gyrus",                                 # Cu_R
    "59": "Left Lateral Occipital Gyrus",                         # LG_L
    "6": "Left Superior Frontal Gyrus (Pole)",                     # SFG_pole_R
    "60": "Right Lateral Occipital Gyrus",                        # LG_R
    "61": "Left Rostral ACC",                                     # rostral_ACC_L
    "62": "Right Rostral ACC",                                    # rostral_ACC_R
    "63": "Left Subcallosal ACC",                                 # subcallosalACC_L
    "64": "Right Subcallosal ACC",                                # subcallosalACC_R
    "65": "Left Subgenual ACC",                                   # subgenualACC_L
    "66": "Right Subgenual ACC",                                  # subgenualACC_R
    "67": "Left Dorsal ACC",                                      # dorsalACC_L
    "68": "Right Dorsal ACC",                                     # dorsalACC_R
    "69": "Left Posterior Cingulate gyrus",                        # PCC_L
    "7": "Left Middle Frontal Gyrus",                            # MFG_L
    "70": "Right Posterior Cingulate gyrus",                       # PCC_R
    "71": "Left Insula",                                         # Insula_L
    "72": "Right Insula",                                        # Insula_R
    "73": "Left Amygdala",                                      # Amyg_L
    "74": "Right Amygdala",                                     # Amyg_R
    "75": "Left Hippocampus",                                    # Hippo_L
    "76": "Right Hippocampus",                                   # Hippo_R
    "77": "Left Caudate Nucleus",                                # Caud_L
    "78": "Right Caudate Nucleus",                               # Caud_R
    "79": "Left Putamen",                                       # Put_L
    "8": "Right Middle Frontal Gyrus",                            # MFG_R
    "80": "Right Putamen",                                      # Put_R
    "81": "Left Globus Pallidus",                                 # GP_L
    "82": "Right Globus Pallidus",                                # GP_R
    "83": "Left Thalamus",                                      # Thalamus_L
    "84": "Right Thalamus",                                     # Thalamus_R
    "85": "Left Hypothalamus",                                  # HypoThalamus_L
    "86": "Right Hypothalamus",                                 # HypoThalamus_R
    "87": "Left Basal Forebrain",                                 # BasalForebrain_L
    "88": "Right Basal Forebrain",                                # BasalForebrain_R
    "89": "Left Nucleus Accumbens",                              # NucAccumbens_L
    "9": "Left Middle Frontal Gyrus (Dorsal Prefrontal Cortex)",   # MFG_DPFC_L
    "90": "Right Nucleus Accumbens",                             # NucAccumbens_R
    "91": "Left Red Nucleus",                                    # RedNc_L
    "92": "Right Red Nucleus",                                   # RedNc_R
    "93": "Left Substantia Nigra (SNg)",                         # Snigra_L
    "94": "Right Substantia Nigra (SNg)",                        # Snigra_R
    "95": "Right Cerebellum Gray Matter",                         # CerebellumGM_R
    "96": "Left Cerebellum Gray Matter",                         # CerebellumGM_L
    "97": "Left Cerebellum White Matter (CP)",                      # CP_L
    "98": "Right Cerebellum White Matter (CP)",                     # CP_R
    "99": "Left Midbrain",                                       # Midbrain_L
}
