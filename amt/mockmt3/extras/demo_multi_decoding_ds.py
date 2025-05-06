# Copyright 2024 The YourMT3 Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Please see the details in the LICENSE file.
from typing import Dict, Tuple
from copy import deepcopy
import soundfile as sf
import torch
from mockmt3.utils.data_modules import AMTDataModule
from mockmt3.config.data_presets import data_preset_single_cfg, data_preset_multi_cfg
from mockmt3.utils.augment import intra_stem_augment_processor
from mockmt3.utils.task_manager import TaskManager


def get_ds(data_preset_multi: Dict, train_num_samples_per_epoch: int = 90000):
    dm = AMTDataModule(data_preset_multi=data_preset_multi,
                       train_num_samples_per_epoch=train_num_samples_per_epoch,
                       task_manager=TaskManager(task_name="mc13_full_plus_256"))
    dm.setup('fit')
    dl = dm.train_dataloader()
    ds = dl.flattened[0].dataset
    return ds


def gen_audio(index: int = 0):
    # audio_arr: (b, 1, nframe), note_token_arr: (b, l), task_token_arr: (b, task_l)
    audio_arr, note_token_arr, task_token_arr = ds.__getitem__(index)

    # merge all the segments into one audio file
    audio = audio_arr.permute(0, 2, 1).reshape(-1).squeeze().numpy()

    # save the audio file
    sf.write('xaug_demo_audio.wav', audio, 16000, subtype='PCM_16')


data_preset_multi = data_preset_multi_cfg["singing_debug2"]
ds = get_ds(data_preset_multi)
ds.random_amp_range = [0.8, 1.1]
ds.stem_xaug_policy = {
    "max_k": 0,
    "tau": 0.3,
    "alpha": 1.0,
    "max_subunit_stems": 12,
    "no_instr_overlap": False,
    "no_drum_overlap": True,
    "uhat_intra_stem_augment": True,
}
audio_ar, note_token_arr, task_token_arr = ds.__getitem__(0)
