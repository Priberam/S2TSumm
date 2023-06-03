import json
import torch
import re
import os
import copy
from torch.utils.data import Dataset
import torchaudio
from transformers import Wav2Vec2Processor, RobertaTokenizerFast
from typing import List


class S2TSumDataset(Dataset):
    def __init__(
        self,
        datasets = ['euronews'],
        roots = ['/mnt/SSD2/datasets/EuroNewsAudio/fr/'],
        split = 'train',
        prepare_dataset = None
    ): 
        # Convert arguments to lists
        if type(datasets) is not list:
            datasets = [datasets]
        if type(roots) is not list:
            roots = [roots]

        # Define valid datasets and check if the provided datasets are included
        self.valid_datasets = ['euronews', 'commonvoice']                
        assert list(set(datasets) - set(self.valid_datasets)) == [], f'one or more of the datasets you indicated is not valid, got {datasets}'
        # The number of datasets and the provided directories must match
        assert len(datasets) == len(roots), f'you must provide an equal number of datasets and their directories'
        self.datasets = datasets

        # Check if the provided directories exist
        assert sum(os.path.isdir(root) for root in roots) == len(roots), f'one or more of the directories you indicated could not be found'
        self.roots = roots

        # Check if the indicated split is valid and whether the corresponding file exists
        self.valid_splits = ['train', 'dev', 'test']
        assert split in self.valid_splits, f'the indicated split name is not valid, got {split}'
        self.splits = [os.path.join(root, split) + '.jsonl' if ds in ['euronews'] else os.path.join(root, split) + '.tsv' for ds, root in zip(datasets, roots)]
        assert sum(os.path.exists(spl) for spl in self.splits) == len(roots), f'missing indicated split in the provided directories'

        self.dataset = []
        for ds, spl, root in zip(self.datasets, self.splits, self.roots):
            # Add to global dataset, and pay attention to particular dataset details
            if ds == 'euronews':
                # Read JSONL file
                with open(spl, 'r') as f:
                    data = f.readlines()
                data = [json.loads(item) for item in data]
                ids = [re.match('(.+).mp3', item['audioFile']).group(1).strip() for item in data]                   
                self.dataset = self.dataset + [{
                    'ds' : ds, 
                    'audio' : {
                        'path' : os.path.join(root, 'audio', id + '.mp3'),
                        'sampling_rate' : torchaudio.info(os.path.join(root, 'audio', id + '.mp3')).sample_rate
                    }, 
                    'audio_embeddings' : {
                        'path' : os.path.join(root, 'audio_bin', id + '.bin') 
                    }, 
                    'text_embeddings' : {
                        'path' : os.path.join(root, 'text_bin', id + '.bin') 
                    }, 
                    'text' : {
                        'source' : item['articleBody'],
                        'target' : item['description']
                    }
                } for id, item in zip(ids, data)]
            elif ds == 'commonvoice':
                # Read TSV file
                with open(spl, 'r') as f:
                    data = [[item.strip() for item in line.split('\t')] for line in f.readlines()]
                path_index = data[0].index('path')
                sentence_index = data[0].index('sentence')
                ids = [re.search('(.+).mp3', item[path_index]).group(1).strip() for item in data[1:]]
                self.dataset = self.dataset + [{
                    'ds' : ds, 
                    'audio' : {
                        'path' : os.path.join(root, 'clips', item[path_index]),
                        'sampling_rate' : torchaudio.info(os.path.join(root, 'clips', item[path_index])).sample_rate
                    },  
                    'audio_embeddings' : {
                        'path' : os.path.join(root, 'audio_bin', id + '.bin') 
                    }, 
                    'text_embeddings' : {
                        'path' : os.path.join(root, 'text_bin', id + '.bin') 
                    }, 
                    'text' : {
                        'source' : item[sentence_index]
                    }
                } for id, item in zip(ids, data[1:])]   

        self.stats = {ds : {} for ds in self.datasets}
        # Get mean and std embeddings characteristic of each dataset
        for ds, root in zip(self.datasets, self.roots):
            with open(os.path.join(root, 'stats/audio_mean.bin'), 'rb') as f:
                self.stats[ds]['audio_mean'] = torch.load(f, map_location=torch.device('cpu'))
            with open(os.path.join(root, 'stats/audio_std.bin'), 'rb') as f:
                self.stats[ds]['audio_std'] = torch.load(f, map_location=torch.device('cpu'))
            with open(os.path.join(root, 'stats/text_mean.bin'), 'rb') as f:
                self.stats[ds]['text_mean'] = torch.load(f, map_location=torch.device('cpu'))
            with open(os.path.join(root, 'stats/text_std.bin'), 'rb') as f:
                self.stats[ds]['text_std'] = torch.load(f, map_location=torch.device('cpu'))      

        # Store the method to be applied to each item
        self.prepare_dataset = prepare_dataset  

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        # Get item and forward it through the prepare_dataset
        item = copy.deepcopy(self.dataset[idx])
        if self.prepare_dataset is None:
            return item
        else:
            return self.prepare_dataset(item, self.stats)


class prepare_dataset:
    def __init__(
        self,
        processor: Wav2Vec2Processor,
        tokenizer: RobertaTokenizerFast,
        sampling_rate: float = 16000,
        mono: bool = True,
        chars_to_ignore_regex: str = '',
        delimeter: str = '\u2581',
        load: List[str] = ['waveform', 'transcript', 'description', 'pre-processed']
    ):
        self.sampling_rate = sampling_rate
        self.mono = mono
        self.chars_to_ignore_regex = chars_to_ignore_regex
        self.processor = processor
        self.tokenizer = tokenizer
        self.delimeter = delimeter
        self.load = load

    def __call__(self, item, feature_stats=None):

        # Load waveform, audio and text pre-processed features, if required
        if 'waveform' in self.load:
            waveform, _ = torchaudio.load(item['audio']['path'])
            item['audio'].update([('waveform', waveform), ('num_channels', waveform.size(dim=0))])
        if 'pre-processed' in self.load:
            with open(item['audio_embeddings']['path'], 'rb') as f:
                item['audio_embeddings'].update([('embeddings', torch.load(f, map_location=torch.device('cpu')))])
            with open(item['text_embeddings']['path'], 'rb') as f:
                item['text_embeddings'].update([('embeddings', torch.load(f, map_location=torch.device('cpu')))])
        if feature_stats is not None:
            item['audio_embeddings'].update([('mean', feature_stats[item['ds']]['audio_mean']), ('std', feature_stats[item['ds']]['audio_std'])])
            item['text_embeddings'].update([('mean', feature_stats[item['ds']]['text_mean']), ('std', feature_stats[item['ds']]['text_std'])])

        if 'waveform' in self.load:
            # Resample audio and convert from stereo to mono if required
            if self.mono:
                item['audio'].update([('waveform', torch.mean(torchaudio.functional.resample(item['audio']['waveform'], item['audio']['sampling_rate'], self.sampling_rate), dim=0, keepdim=True))])
            else:
                item['audio'].update([('waveform', torchaudio.functional.resample(item['audio']['waveform'], item['audio']['sampling_rate'], self.sampling_rate))])
            item['audio'].update([('sampling_rate', self.sampling_rate)])

            # Extract audio features / batched output is "un-batched" to ensure mapping is correct
            item['audio'].update([('features', self.processor(item['audio']['waveform'][0], sampling_rate=item['audio']['sampling_rate'], return_tensors='pt').input_values.squeeze(dim=0))])

        if 'transcript' in self.load:
            # Remove special chars from transcriptions and store the resulting strings
            item['text'].update([('ctc', re.sub(self.chars_to_ignore_regex, '', item['text']['source']).strip())])
            item['text'].update([('ctc', self.delimeter + re.sub(' ', self.delimeter, item['text']['ctc']))])

            # Tokenize the text for the CTC
            with self.processor.as_target_processor():
                item['text'].update([('ctc_token_ids', self.processor(item['text']['ctc'], return_tensors='pt').input_ids.squeeze(dim=0))])

        return item