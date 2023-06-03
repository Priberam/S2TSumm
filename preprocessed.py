import os
from glob import glob
import re
import json
from tqdm import tqdm
import torch
import torchaudio
import torchaudio.functional as F
from transformers import RobertaTokenizerFast, EncoderDecoderModel
from speechbrain.lobes.models.huggingface_wav2vec import HuggingFaceWav2Vec2
import numpy as np
import random
from math import ceil
import argparse


def audio_features(
    root = None, 
    gpu = False
):

    # Check if input folders exists
    assert os.path.isdir(root), f'you must provide a valid root folder'
    folders = os.listdir(root)
    if 'audio' in folders:
        audio = 'audio'
    elif 'clips' in folders:
        audio = 'clips'
    else:
        raise Exception('There is no \'audio\' or \'clips\' folder in the root folder.')

    # Name of the folder to store the speech features and create folder if it does not exist
    emb = 'audio_bin'
    if not os.path.isdir(os.path.join(root, emb)):
        os.mkdir(os.path.join(root, emb))    

    # The Huggingface model whose embeddings will be extracted and path to save its last checkpoint
    model_hub = 'LeBenchmark/wav2vec2-FR-7K-base'

    # Set device to process the data
    device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'

    # Load and save the Huggingface model of interest and save it
    model = HuggingFaceWav2Vec2(model_hub, output_norm=False, save_path='pretrained').to(device)
    target_rate = 16000

    # Index of the transformer encoder layer whose embedding one wishes to extract
    idx_layer = 6

    # Obtain the split files
    split_list = glob(os.path.join(root, '*.jsonl'))
    if split_list == []:
        split_list = glob(os.path.join(root, '*.tsv'))

    # List of MP3 files
    audio_files = []
    for split in split_list:
        if os.path.splitext(split)[1] == '.jsonl':
            with open(split, 'r') as f:
                data = f.readlines()
            data = [json.loads(item) for item in data]
            audio_files = audio_files + [os.path.join(root, audio, os.path.basename(item['audioFile'].strip())) for item in data]
        elif os.path.splitext(split)[1] == '.tsv':
            with open(split, 'r') as f:
                data = [[item.strip() for item in line.split('\t')] for line in f.readlines()]
            path_index = data[0].index('path')
            audio_files = audio_files + [os.path.join(root, audio, os.path.basename(item[path_index].strip())) for item in data[1:]]

    # Number of audios
    num_audios = len(audio_files)

    # Get list of binary files that were already produced
    listOfBIN = glob(os.path.join(root, emb, '*.bin'))

    # Counters
    stereo = 0
    sample_rates = []
    for idx, f_name in zip(range(num_audios), audio_files):
        print('File number', idx + 1, 'of', num_audios)
        # Path to the resulting binary file
        bin_name = os.path.splitext(os.path.join(root, emb, os.path.split(f_name)[1]))[0] + '.bin'

        # Skip file if already exists
        if bin_name in listOfBIN:
            print('File already exists\n')
            continue

        # Load the MP3 file as a pytorch tensor and convert it from stereo to mono 
        waveform, sample_rate = torchaudio.load(f_name)
        sample_rates.append(sample_rate)
        waveform = F.resample(waveform, sample_rate, target_rate).to(device)
        if waveform.size(dim=0) != 1:
            stereo = stereo + 1
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        # Aplly the model to the waveform
        # The returned dict has keys ('last_hidden_state', 'extract_features', 'hidden_states')
        # The value corresponding to key 'hidden_states' is a tuple with number of tensors corresponding to 1 + n_transformer_layers
        print('Forwarding input', f_name, 'through the model...')
        wav2vec2_dict = model.model.forward(input_values=waveform, output_hidden_states=True, return_dict=True)

        # Write the relevant embedding in the binary file
        with open(bin_name, 'wb') as f:
            print('Dumping the results in the file', bin_name, '\n')
            torch.save(torch.squeeze(wav2vec2_dict['hidden_states'][idx_layer + 1]).clone(), f)

    print('Number of stereo files:', stereo, '/', num_audios)
    print('Sample rates:', set(sample_rates))


def text_features(
    root = None,
    batch_size = 64,
    gpu = False
):

    # Check if input folders exists
    assert os.path.isdir(root), f'you must provide a valid root folder'

    # Check if batch size makes sense
    assert batch_size > 0, f'you must provide a valid value for the batch size'

    # Name of the folder to store the speech features and create folder if it does not exist
    emb = 'text_bin'
    if not os.path.isdir(os.path.join(root, emb)):
        os.mkdir(os.path.join(root, emb))    

    # Checkpoint of the model
    ckpt = 'mrm8488/camembert2camembert_shared-finetuned-french-summarization'

    # Set device to process the data
    device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'

    # Load tokenizer and model
    tokenizer = RobertaTokenizerFast.from_pretrained(ckpt)
    model = EncoderDecoderModel.from_pretrained(ckpt).to(device)

    # Obtain the split files
    split_list = glob(os.path.join(root, '*.jsonl'))
    if split_list == []:
        split_list = glob(os.path.join(root, '*.tsv'))

    for split in split_list:
        # Get list of binary files that were already produced
        listOfBIN = glob(os.path.join(emb, '*.bin'))

        # Get the names of the target binary files and the transcripts
        if os.path.splitext(split)[1] == '.jsonl':
            with open(split, 'r') as f:
                data = f.readlines()
            data = [json.loads(item) for item in data]
            paths = [os.path.join(root, emb, os.path.splitext(item['audioFile'].strip())[0] + '.bin') for item in data]
            transcripts = [item['articleBody'] for item in data]
        elif os.path.splitext(split)[1] == '.tsv':
            with open(split, 'r') as f:
                data = [[item.strip() for item in line.split('\t')] for line in f.readlines()]
            path_index = data[0].index('path')
            sentence_index = data[0].index('sentence')
            paths = [os.path.join(root, emb, os.path.splitext(item[path_index])[0]) + '.bin' for item in data[1:]]
            transcripts = [item[sentence_index] for item in data[1:]]


        num_batches = ceil(len(transcripts) / batch_size)
        idx_batch = 0
        while len(transcripts) != 0:
            print(idx_batch + 1, ' / ', num_batches)

            # Create batch
            batch_transcripts = []
            batch_paths = []
            while len(batch_transcripts) != batch_size and len(transcripts) != 0:
                transcript = transcripts.pop()
                path = paths.pop()
                # Skip file if already exists
                if path in listOfBIN:
                    continue
                else:
                    batch_transcripts.append(transcript)
                    batch_paths.append(path)

            # Break if there was nothing to add
            if len(batch_transcripts) == 0:
                break

            # Aplly the model to the text
            # The returned dict has keys ('sequences', 'encoder_hidden_states', 'decoder_hidden_states')
            print('\tForwarding input through the model...')
            inputs = tokenizer(batch_transcripts, padding="max_length", truncation=True, max_length=512, return_tensors="pt")
            input_ids = inputs.input_ids.to(device)
            attention_mask = inputs.attention_mask.to(device)
            output = model.generate(input_ids, attention_mask=attention_mask, output_hidden_states=True, return_dict_in_generate=True) 

            # Write the relevant embeddings in the binary files
            for idx, path in enumerate(batch_paths):
                with open(path, 'wb') as f:
                    print('\tDumping the results in the file', path)
                    torch.save(output['encoder_hidden_states'][-1][idx][:torch.sum(attention_mask[idx])].clone(), f)
            print('')

            idx_batch = idx_batch + 1


def save_mean_std_features(
    root = None, 
    splits = None,
    mode = 'audio',
    gpu = False
):

    # Check if input folders exists
    assert os.path.isdir(root), f'you must provide a valid root folder'

    # Check if mode is valid and whether there is a corresponding folder
    assert mode in ['audio', 'text'], f'you must provide a valid mode, got {mode}'
    emb = 'audio_bin' if mode == 'audio' else 'text_bin'
    assert os.path.isdir(os.path.join(root, emb)), f'the folder with the features binaries could not be found'

    # Check if the splits are valid
    splits = list(set(splits))
    assert isinstance(splits, list) and list(set(splits) - set(['train', 'dev', 'test'])) == [], f'you must provide valid split names'

    # Get the split files
    splits = [os.path.join(root, split + '.jsonl') if os.path.exists(os.path.join(root, split + '.jsonl')) else os.path.join(root, split + '.tsv') for split in splits]
    assert all(os.path.exists(split) for split in splits), f'some of the splits you asked for could not be found'

    # Name of the folder to store the mean and standard deviation features and create folder if it does not exist
    stats = 'stats'
    if not os.path.isdir(os.path.join(root, stats)):
        os.mkdir(os.path.join(root, stats))    

    # Set device to process the data
    device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'
    
    # Get all the feature binary files from all the given splits
    paths = []
    for split in splits:
        if os.path.splitext(split)[1] == '.jsonl':
            with open(split, 'r') as f:
                data = f.readlines()
            data = [json.loads(item) for item in data]
            paths = paths + [os.path.join(root, emb, os.path.splitext(item['audioFile'].strip())[0] + '.bin') for item in data]
        elif os.path.splitext(split)[1] == '.tsv':
            with open(split, 'r') as f:
                data = [[item.strip() for item in line.split('\t')] for line in f.readlines()]
            path_index = data[0].index('path')
            paths = paths + [os.path.join(root, emb, os.path.splitext(item[path_index])[0] + '.bin')  for item in data[1:]]
    
    # Determine mean feature
    stack = []
    _len_ = 0    
    for path in tqdm(paths):
        # Load data
        with open(path, 'rb') as f:
            features = torch.load(f, map_location=torch.device('cpu')).squeeze(dim=0).to(device)
        stack.append(torch.sum(features, dim=0, keepdim=False).unsqueeze(dim=0))
        _len_ = _len_ + features.size(dim=0)
    stack = torch.stack(stack, dim=0)
    mean = torch.sum(stack, dim=0, keepdim=False) / _len_

    # Determine STD feature
    stack = []
    for path in tqdm(paths):
        # Load data
        with open(path, 'rb') as f:
            features = torch.load(f, map_location=torch.device('cpu')).squeeze(dim=0).to(device)
        stack.append(torch.sum(torch.pow(features - mean, 2), dim=0, keepdim=False).unsqueeze(dim=0))
    stack = torch.stack(stack, dim=0)
    std = torch.sqrt(torch.sum(stack, dim=0, keepdim=False) / (_len_ - 1))

    # Save results to files
    with open(os.path.join(root, stats, mode + '_mean.bin'), 'wb') as f:
        torch.save(mean.clone(), f)
    with open(os.path.join(root, stats, mode + '_std.bin'), 'wb') as f:
        torch.save(std.clone(), f)

    # Print results shapes
    print('Mean shape: ', mean.shape)
    print('STD shape: ', std.shape)
    

def cli():   

    # Create CLI parser
    parser = argparse.ArgumentParser(
        description = 'Pre-processing', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-s', '--stats', dest='stats', action='store_true', help='Determine the mean and standard deviation features.')
    parser.add_argument('-a', '--audio', dest='audio', action='store_true', help='Work with the audio features.')
    parser.add_argument('-t', '--text', dest='text', action='store_true', help='Work with the text features.')
    parser.add_argument('-r', '--root', default='', type=str, help='The path to the root folder of the dataset.')
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('-f', '--splits', default='', type=str, help='Delimited list input with split names separated by commas.')
    parser.add_argument('-b', '--batch_size', default=1, type=int, help='Batch size for extracting the text features. If used for audio, drops to 1.')

    # Parse arguments
    args = parser.parse_args()

    if args.splits != '':
        splits = [split for split in args.splits.split(',')]

    # Check if at most only one of the audio or text arguments was provided
    assert not (args.audio and args.text), f'you must provide at most one of the audio or text arguments'

    # Check if splits is not empty if they are required
    assert not args.stats or (args.stats and splits != []), f'of you want to determine stats, provide the splits for calculation'

    if args.audio:
        if not args.stats:
            audio_features(
                root = args.root, 
                gpu = args.gpu
            )
        else:
            save_mean_std_features(
                root = args.root,
                splits = splits,
                mode = 'audio',
                gpu = args.gpu
            )
    elif args.text:
        if not args.stats:
            text_features(
                root = args.root, 
                batch_size = args.batch_size,
                gpu = args.gpu
            )
        else:
            save_mean_std_features(
                root = args.root,
                splits = splits,
                mode = 'text',
                gpu = args.gpu
            )


if __name__ == '__main__':

    cli()