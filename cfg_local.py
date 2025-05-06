import os

model_name_to_log_path = {
    "YMT3+": {
        "log_path": "amt/logs/2024/notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72/checkpoints/model.ckpt",
        "log_s3_uri": "s3://amt-deploy-public/amt/logs/2024/notask_all_cross_v6_xk2_amp0811_gm_ext_plus_nops_b72",
        "nickname": "m0",
    },
    "YPTF+Single (noPS)": {
        "log_path":
            "amt/logs/2024/ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100/checkpoints/model.ckpt",
        "log_s3_uri":
            "s3://amt-deploy-public/amt/logs/2024/ptf_all_cross_rebal5_mirst_xk2_edr005_attend_c_full_plus_b100",
        "nickname":
            "m1",
    },
    "YPTF+Multi (PS)": {
        "log_path":
            "amt/logs/2024/mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k/checkpoints/model.ckpt",
        "log_s3_uri":
            "s3://amt-deploy-public/amt/logs/2024/mc13_256_all_cross_v6_xk5_amp0811_edr005_attend_c_full_plus_2psn_nl26_sb_b26r_800k",
        "nickname":
            "m2",
    },
    "YPTF.MoE+Multi (noPS)": {
        "log_path":
            "amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops/checkpoints/last.ckpt",
        "log_s3_uri":
            "s3://amt-deploy-public/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops",
        "nickname":
            "m3",
    },
    "YPTF.MoE+Multi (PS)": {
        "log_path":
            "amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2/checkpoints/model.ckpt",
        "log_s3_uri":
            "s3://amt-deploy-public/amt/logs/2024/mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b80_ps2",
        "nickname":
            "m4",
    },
    "ptfs_gmlp_wf_e12_d8_500k": {
        "log_path": "amt/logs/2024/ptfs_gmlp_wf_e12_d8_500k/checkpoints/epoch=466-step=500000.ckpt",
        "log_s3_uri": "s3://amt-deploy-public/amt/logs/2024/ptfs_gmlp_wf_e12_d8_500k",
        "nickname": "m5",
    }
}


class cfg_local:
    amt_output_midi_dir: str = './output'
    amt_model_bsz: int = 64
    amt_model_precision: str = 'bf16-mixed'  # ["32", "bf16-mixed", "16"]
    test_vocab = 'use_instr_info'  #"mt3_full_plus"  # or None
    # default_model_name = "YMT3+"
    default_model_name = "YPTF+Single (noPS)"
    # default_model_name = "YPTF+Multi (PS)"
    # default_model_name = "YPTF.MoE+Multi (noPS)"
    # default_model_name = "YPTF.MoE+Multi (PS)"
    # default_model_name = "ptfs_gmlp_wf_e12_d8_500k"
