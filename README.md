# Towards End-to-end Speech-to-text Summarization

## Description

This repository contains the source code for training and testing the models for speech-to-text summarization presented in the paper **Towards End-to-end Speech-to-text Summarization**, that will be included in the conference proceedings of the 26th International Conference for Text, Speech and Dialogue ([TSD2023](https://www.kiv.zcu.cz/tsd2023/index.php)). The following models are included in this repository:

1. Speech-to-text Cascade Abstractive Summarizer (Cascade)
2. Speech-to-text End-to-end Abstractive Summarizer (E2E)
3. Speech-to-text Cascade Extractive Summarizer (Extractive)

For further details on each of the models, please consult the paper.

## Usage

Pytorch-Lightning was used to organize the code. Some of the **run-*.py** scripts contain LightningCLIs to manage the training and testing given a YAML configuration file. The following table summarizes the usage of these scripts, the model targetted and the evaluation metrics in the `test` mode:

|       Model      |                               Script                               | Test Metric |
|:----------------:|:------------------------------------------------------------------:|:-----------:|
|   ASR (Cascade)  |     `python3 run-asr.py [fit/test] --config src/configs/asr.yaml`    |     WER     |
| T2TSum (Cascade) | `python3 run-text.py [fit/test] --config src/configs/text_summ.yaml` |    ROUGE    |
|   Adapter (E2E)  |  `python3 run-adapter.py fit --config src/configs/adapter_[TS].yaml` |      -      |
|   S2TSum (E2E)   |    `python3 run-summ.py [fit/test] --config src/configs/summ.yaml`   |    ROUGE    |

`[TS]` refers to the pre-training stage of the cross-modal adapter (further details are described in the paper). Some configuration files are included in `src/configs/` as examples, but new ones can be created using:

`python3 run-[name].py fit --model=src.models.[m_file].[m_name] --print_config > src/configs/[cf_name].yaml`

* [name] contains the **run** script name;
* [m_file] is the python file in `src/models/` where the model source code lies;
* [m_name] is the name of the model;
* [cf_name] is the name of the YAML file to be generated.

**IMPORTANT!** 
* Every YAML configuration file contains a field `ckpt_path` that must be set to the path of a particular checkpoint when resuming training. Besides, the `dirpath` for the model checkpoints must be set in the configuration files.
* The LightningDataModule `S2TSumDataMod` needs to be specified (in the YAML) the folders where the datasets are stored in `roots`, which must be in a specific format. More information about their formatting and other details can be found in **Datasets**.

**Evaluate the cascade model**

The `run-cascade.py` script serves to evaluate the cascade abstractive summarizer using trained ASR model and T2T abstractive summarizer. An example of usage is the following:

`python3 run-cascade.py --asr_ckpt=[ASR] --t2t_ckpt=[T2T] --root=[ROOT] --split=[SPLIT]`

* [ASR] is the checkpoint to the ASR;
* [T2T] is the checkpoint to the T2T abstractive summarizer;
* [ROOT] is the root folder of the BNews corpus;
* [SPLIT] is the name of the split to be evaluated (e.g. `test`).

**Evaluate the extractive model**

The `run-extractive.py` script serves to evaluate the cascade extractive summarizer using trained ASR model and a [publicly available CamemBERT-based model](https://github.com/ialifinaritra/Text_Summarization) whose source code is in `src/models/extractive`. An example of usage is the following:

`python3 run-extractive.py --asr_ckpt=[ASR] --root=[ROOT] --split=[SPLIT]`

* [ASR] is the checkpoint to the ASR;
* [ROOT] is the root folder of the BNews corpus;
* [SPLIT] is the name of the split to be evaluated (e.g. `test`).

## Datasets

The datasets used in this project were the Common Voice Corpus 10.0 and the BNews corpus introduced in the afforementioned paper.

**Folder Structure**

The `S2TSumDataMod` object implemented in `src/data/data_module.py` is built to either take the Common Voice or the BNews corpora. The folders where these datasets are stored must be specified and have the following structure:

```bash
├── root
│   ├── train.tsv
│   ├── dev.tsv
│   ├── test.tsv
│   ├── clips
│   │   ├── *.mp3
│   ├── audio_bin
│   │   ├── *.bin
│   ├── text_bin
│   │   ├── *.bin
│   ├── stats
│   │   ├── audio_mean.bin
│   │   ├── audio_std.bin
│   │   ├── text_mean.bin
│   │   ├── text_std.bin
```

This is the structure for the Common Voice dataset folder. The one for the BNews corpus is similar, where `clips` is renamed as `audio` and the `.tsv` files are instead `.jsonl` files.

**Download**

To obtain the Common Voice Corpus 10.0, one may download it from an official source and it will contain all but the `audio_bin`, `text_bin` and `stats` folders.

Details about obtaining the BNews corpus will be added to this repository when we get a public license from EuroNews to share the dataset.

**Preparation**

To speed up the training of the cross-modal adapter and the S2T abstractive summarizer, the audios and transcripts from both the Common Voice and BNews corpora are pre-processed for extracting the speech and textual features, which are then stored in the `audio_bin` and `text_bin` folders, respectively. This can be achieved with the following command:

`python3 preprocess.py --[MODE] --root=[ROOT]`

* [MODE] is either `audio` or `text`;
* [ROOT] is the root folder of the dataset.

Both the speech and textual features are normalized such that each dimension has zero mean and unit variance. The mean and standard deviation features are stored in the folder `root/stats` and can be obtained with the following command:

`python3 preprocess.py --[MODE] --stats --root=[ROOT] --splits=[SPLITS]`

* [MODE] is either `audio` or `text`;
* [ROOT] is the root folder of the dataset;
* [SPLITS] is the names of the splits separated by commas whose extracted features will be used in the calculation of the mean and std features (e.g. `train,dev`).

**IMPORTANT!** 
* Skipping these 2-step pre-processing of the datasets can cause undefined behaviour from the `S2TSumDataMod`.

## Citation

*Missing Citation*

## Acknowledgments

This work is supported by the EU H2020 SELMA project (grant agreement No. 957017).