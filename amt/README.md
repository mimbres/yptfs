[![License: apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg?logo=apache&logoColor=white)](https://opensource.org/licenses/Apache-2.0) [![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://github.com/Lightning-AI/lightning) [![PyTorch](https://img.shields.io/badge/PyTorch-1.14|2.0%20-lightblue.svg?style=flat&logo=PyTorch&logoColor=EE4C2C)](https://pytorch.org)  [![Python Version](https://img.shields.io/badge/Python-3.8|3.9%20-lightblue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/) [![Tests Status](https://img.shields.io/badge/Pytest-72%20passed-98FB98?logo=pytest&logoColor=white)](src/tests/report.html) [![coverage](https://img.shields.io/badge/Coverage-88%25-98FB98)]() [![Yapf](https://img.shields.io/badge/Yapf-passing-98FB98?logo=github&logoColor=white)](https://github.com/google/yapf)

<!-- [![License: apache 2.0](https://img.shields.io/badge/License-Apache%202.0-D22128.svg?logo=apache&logoColor=white)](https://opensource.org/licenses/Apache-2.0) [![PyTorch](https://img.shields.io/badge/PyTorch-1.14|2.0%20-EE4C2C.svg?style=flat&logo=PyTorch&logoColor=EE4C2C)](https://pytorch.org)[![Python V
ersion](https://img.shields.io/badge/Python-3.8|3.9%20-lightblue.svg?logo=python&logoColor=white)](https://www.python.org/downloads/)  [![Lightning](https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white)](https://github.com/Lightning-AI/lightning) [![Tests Status](https://img.shields.io/badge/Pytest-62%20passed-pink?logo=pytest&logoColor=white)](src/tests/report.html) [![coverage](https://img.shields.io/badge/Coverage-90%25-green)]() [![Yapf](https://img.shields.io/badge/Yapf-passing-green?logo=github&logoColor=white)](https://github.com/google/yapf) -->

<!-- ![yourmt3-low-resolution-color-logo-crop](https://user-images.githubusercontent.com/26891722/204390355-001877a1-d019-46d7-a33c-d3a3adc0743c.png) -->

# :drum: YourMT3+ for P3 Delivarable (Oct 2023)


## System Requirements:
- Linux system with Python>=3.9
- Main memory > 1GB
- A single GPU with 16 GB memory. By default, the system assumes a batch size of 128 for testing purposes.
- For GPUs with memory size < 16 GB, smaller batch size is configuarable in [config.py](src/config/config.py).

## Install
```
git clone git@github.com:mimbres/mock-mt3.git
pip install -e .
```

### Install Dataset
The following command will download and preprocess the `YourMT3` version of datasets. 
- To access restricted datasets, please [request token](https://zenodo.org/records/8417804). Note that every token expires in 30 days.
- If you do not specify the `data_home` path, `shared_cfg["PATH"]["data_home"]` of [config.py](src/config/config.py#L102) will be used.  
```
python install_dataset.py
python install_dataset.py [YOUR/DATA_HOME/PATH] 
python install_dataset.py --nodown # skip download
```

## 🚀 Quick Setup
- Adjust [config.py](src/config/config.py), if neeeded.
```
shared_cfg = {
  "PATH": {
      "data_home": 'path/to/dir', # Absolute or relative path
  },
  "WANDB": {
        "mode": "disabled", # Options: {online, offline, disabled}
  },
  "BSZ": {
        "test": 128, # Test batch size
    },
}
```

## Model Checkpoints
- Four checkpoint files with `exp_id`s (Last update 20 Nov):
  - `ymt3_edr005_gm_ext_plus_b72`
    - enc: T5, dec: T5,
    - enc_dropout: 0.05,
    - vocab: [gm_ext_plus](src/config/vocabulary.py#L369) 
  - `cf_edr005_midi_plus_b80`
    - enc: Conformer, dec: T5
    - enc_dropout: 0.05
    - vocab: [mt3_midi_plus](src/config/vocabulary.py#L369) 
  - `ptf_atc_edr005_gm_full_plus_b100`
    - enc: Perceiver-TF, dec: T5
    - enc_dropout: 0.05
    - vocab: [gm_full_plus](src/config/vocabulary.py#L369) 
  - `ptf_mc13_l256_b26`
    - enc: Perceiver-TF, dec: multi-channel T5
    - enc_dropout: 0.05
    - vocab/task: [mc13_256](src/config/task.py#L70)

- The extracted `exp_id` directory should locate as follows:
```
mock-mt3/
├── logs
│   └──{project_name}             <=== 'ymt3' by default
│      └──{exp_id}
│         ├── checkpoints         <=== model.ckpt places here
│         ├── model_outputs       <=== .npy/mid files of raw outputs (after run)
│         └── result.json         <=== summary metrics (after run)
└── src
```

## Test

### 🚀 Quick Start
- Running final model checkpoint:

```
python test.py ymt3_edr005_gm_ext_plus_b72@model.ckpt -d all_eval_final -pr 32
python test.py cf_edr005_midi_plus_b80@model.ckpt -d all_eval_final -pr 32 -enc conformer
python test.py ptf_atc_edr005_gm_full_plus_b100@model.ckpt -d all_eval_final -pr 32 -enc perceiver-tf -ac spec -hop 300 -atc 1
python test.py ptf_mc13_l256_b26@model.ckpt -d all_eval_final -pr 32 -enc perceiver-tf -dec multi-t5 -nl 26 -ac spec -hop 300 -atc 1 -tk mc13 # trained with seqeunce length 256, but evaluated with length 512
```

- Bass evaluation on RWC-Pop (with `150 ms` onset tolerance)

```
python test.py ymt3_edr005_gm_ext_plus_b72@model.ckpt -d rwc_pop_bass -pr 32 -t 0.15
python test.py cf_edr005_midi_plus_b80@model.ckpt -d rwc_pop_bass -pr 32 -enc conformer -t 0.15
python test.py ptf_atc_edr005_gm_full_plus_b100@model.ckpt -d rwc_pop_bass -pr 32 -enc perceiver-tf -ac spec -hop 300 -atc 1 -t 0.15
python test.py ptf_mc13_l256_b26@model.ckpt -d rwc_pop_bass -pr 32 -enc perceiver-tf -dec multi-t5 -nl 26 -ac spec -hop 300 -atc 1 -tk mc13 -t 0.15
```

- Instrument F1 on Slakh
```
python test.py ymt3_edr005_gm_ext_plus_b72@model.ckpt -d slakh -pr 32
python test.py cf_edr005_midi_plus_b80@model.ckpt -d slakh -pr 32 -enc conformer
python test.py ptf_atc_edr005_gm_full_plus_b100@model.ckpt -d slakh -pr 32 -enc perceiver-tf -ac spec -hop 300 -atc 1
python test.py ptf_mc13_l256_b26@model.ckpt -d slakh -pr 32 -enc perceiver-tf -dec multi-t5 -nl 26 -ac spec -hop 300 -atc 1 -tk mc13
```


### Syntax
```
test.py {EXP_ID}@{CKPT_FILE} [OPTIONS]                                         
```
- Please run ```python test.py –help``` for more details.

### Test Options

| Option | Description |
|--------|-------------|
| -d     | The data preset name used for final evaluation is `all_mmegs_final`. Additional options can be found in [data_presets.py](src/config/data_presets.py). |
| -v     | This option specifies the evaluation vocabulary, and this can differ from the vocab used for training. More vocabs can be found in [vocabulary.py](src/config/vocabulary.py). |
| -edv   | This option is intended for the drum vocabulary defined in [vocabulary.py](src/config/vocabulary.py#L357). |
| -w     | This option specifies whether or not to write output MIDI files. 1 or 0 (default). |
| -t     | Tolerance in seconds. Default is 0.05 (50 ms) |

### MIDI Playback

`.mid` files can be generated by `-w 1` option. For playback, we recommend to use the cool web-based GM player, [vercel/signal](https://signal.vercel.app/).

## Train
### 🚀 Quick Start

```
python train.py EXP_ID
``` 
- Replace `EXP_ID` with your own experiment name. If any checkpoint for the same `EXP_ID` exists, it will automatically resume training from the `last.ckpt`.
- All the logs, including checkpoints, are located in the `/logs` directory by default.
- The default trainer above is equivalent to the following:
```
python train.py EXP_ID -d musicnet_thickstun_ext_em -o adamwscale -s cosine
```
- See more details in the following examples and [Advanced Training Setup](#-advanced-training-setup).


### Reproducing Final Model Training
- Final models were trained using the following commands:
```
# YMT3+
python train.py EXP_ID -d DATA_PRESET -it 400000 -vit 40000 -xk 2 -amp 0.8 1.1 -tk gm_ext_plus -ps -2 2 --strategy ddp

# CF-YMT3+
python train.py EXP_ID -d DATA_PRESET -it 400000 -vit 40000 -enc conformer -xk 2 -amp 0.8 1.1 -edr 0.05 -ddr 0.05 -tk mt3_midi_plus -ps -2 2 --strategy ddp

# PTF-Single 
python train.py EXP_ID -d DATA_PRESET -it 400000 -vit 40000 -enc perceiver-tf -xk 2 -amp 0.8 1.1 -edr 0.05 -ddr 0.05 -ac spec -hop 300 -atc 1 -ps -2 2 -sb 1 --strategy ddp

# PTF-Multi
python train.py EXP_ID -d all_cross_v6 -it 800000 -vit 40000 -enc perceiver-tf -dec multi-t5 -nl 26 -tk mc13_256 -xk 5 -amp 0.8 1.1 -edr 0.05 -ddr 0.05 -ac spec -hop 300 -atc 1 -sb 1 -bsz 13 26 -ps -2 2 --strategy ddp

```
| Flag | Description |
|------|-------------|
| d    | Dataset weights preset from [data_presets.py](src/config/data_presets.py). |
| it   | Max training iterations. |
| vit  | Validation frequency. |
| enc  | Encoder type: 't5', 'conformer', 'perceiver-tf'. |
| atc  | Attend-to-channel. Use `-atc 0` to disable (for perceiver-tf only). |
| xk   | Max datasets for cross-augmentation. Use `-xk 0` to disable. |
| amp  | Random input signal amplitude range. |
| edr  | Encoder dropout rate (default varies by encoder type). |
| ddr  | Decoder dropout rate (default is 0.05). |
| ac   | Audio-codec (default is `mel-spec`). |
| hop  | Input hop-size in frames. |
| ps   | Pitch-shift range [min, max] in semitones (default is `-ps -2 2`). |



## Token Definition

<details>

| token name    | range          | token index  | desc. |
| --------- |:-------------:| :-----:| :-----|
|  PAD | 0 | 0 | special token (padding, ignored in decoding)|
|  EOS | 0 | 1 | `end of sequence` |
|  UNK | 0 | 2 | special token (unknown, ignored in decoding) |
| shift | 0-205 | 3-208 | Time-shift on `10 ms` grid within `2.048 sec` segments. |
| pitch | 0-127 | 209-336 | [MIDI note numbers](https://fmslogo.sourceforge.io/manual/midi-instrument.html)|
|  velocity   | 0-1 | 337-338 | `0` for note-on, `1` for note-off event. |
|  tie   | 0 | 339 | A segment starting with program and pitch refers to `note-on` from the previous segment. The `tie` token marks the main events' beginning.|
|  program   | 0-127 | 340-467 | [GM_INSTR_FULL](src/conifg/vocabulary.py) |
|  drum   | 0-127 | 468-595 | [GM_DRUMSET](src/conifg/vocabulary.py)|

- Total 596 tokens.

</details>

## Advanced Training Setup

<details>
<summary> Some options are available via command line. </summary>

- To be updated...

| Argument       | Short      | Long              | Description                                                                 | Default       |
|----------------|------------|-------------------|-----------------------------------------------------------------------------|---------------|
| exp_id         |            |                   | A unique ID of the experiment, used for resuming training                   |               |
| project        | `-p`       | `--project`       | Project name                                                                |               |
| precision      | `-pr`      | `--precision`     | `32`, `16`, `bf16`, `bf16-mixed`                                       | `bf16-mixed`  |
| pretrained     | `-pt`      | `--pretrained`    | Pretrained T5 model `True` or `False`                                         | `False`         |
| strategy       | `-st`      | `--strategy`      | `auto`, `deepspeed`,  `ddp`)                                         | auto          |
| epochs         | `-e`       | `--epochs`        | Number of max epochs                                                        | 8,000           |
| learning_rate  | `-lr`      | `--learning-rate` | Learning rate                                                               | 1e-4          |
| optimizer      | `-o`       | `--optimizer`     | `AdamW`, `AdaFactor`, or `CPUAdam`)                                   | AdaFactor w/o learning rate   |

</details>

<details>
<summary>More advanced</summary>

More advanced setups are configurable in [model/config.py](src/config/config.py).
</details>

## Contribute
TBA.

## Cite
TBA.