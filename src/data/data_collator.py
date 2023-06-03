import torch
from transformers import Wav2Vec2Processor, RobertaTokenizerFast
from typing import Optional, Union, List
from transformers.feature_extraction_utils import BatchFeature


class S2TSumDataCollator:
    def __init__(
        self, 
        processor: Wav2Vec2Processor,
        tokenizer: RobertaTokenizerFast,
        load: List[str] = ['waveform', 'transcript', 'description', 'pre-processed'],
        padding: Union[bool, str] = True,
        max_length_audio: Optional[int] = None,
        max_length_ctc_labels: Optional[int] = None,
        max_length_text: Optional[int] = None,
        max_length_target_labels: Optional[int] = None,
        pad_to_multiple_of_audio: Optional[int] = None,
        pad_to_multiple_of_ctc_labels: Optional[int] = None,
        pad_to_multiple_of_text: Optional[int] = None,
        pad_to_multiple_of_target_labels: Optional[int] = None,
        audio_embedding_pad_value: float = 0.,
        text_embedding_pad_value: float = 0.,
        text_embedding_padding: Union[bool, str] = None,
        text_embedding_max_length: Optional[int] = None
    ):        
        self.processor = processor
        self.tokenizer = tokenizer
        self.load = load
        self.padding = padding
        self.max_length_audio = max_length_audio
        self.max_length_ctc_labels = max_length_ctc_labels
        self.max_length_text = max_length_text
        self.max_length_target_labels = max_length_target_labels
        self.pad_to_multiple_of_audio = pad_to_multiple_of_audio
        self.pad_to_multiple_of_ctc_labels = pad_to_multiple_of_ctc_labels
        self.pad_to_multiple_of_text = pad_to_multiple_of_text
        self.pad_to_multiple_of_target_labels = pad_to_multiple_of_target_labels
        self.audio_embedding_pad_value = audio_embedding_pad_value
        self.text_embedding_pad_value = text_embedding_pad_value

        # Check if text embedding padding and max length are valid
        assert text_embedding_padding in ['longest', 'max_length'], f'the provided text embedding padding is not valid, got {text_embedding_padding}'
        self.text_embedding_padding = text_embedding_padding
        assert ( (text_embedding_padding is 'longest') or (text_embedding_max_length is not None) ), f'you must provide a text embedding max length for the given padding'
        self.text_embedding_max_length = text_embedding_max_length
  
    def __call__(self, features):

        # Object to store all the batch data
        batch = {'batch_size': len(features)}

        # Batch features for ASR
        if 'waveform' in self.load:
            batch['input_values'] = self.processor.pad(
                [BatchFeature({'input_values' : feature['audio']['features']}) for feature in features],
                padding = self.padding,
                max_length = self.max_length_audio,
                pad_to_multiple_of = self.pad_to_multiple_of_audio,
                return_tensors = 'pt',
            ).input_values
        if 'transcript' in self.load:
            with self.processor.as_target_processor():
                ctc_labels_batch = self.processor.pad(
                    [{'input_ids' : feature['text']['ctc_token_ids']} for feature in features],
                    padding = self.padding,
                    max_length = self.max_length_ctc_labels,
                    pad_to_multiple_of = self.pad_to_multiple_of_ctc_labels,
                    return_tensors = 'pt',
                )
            # replace padding with -100 to ignore loss correctly
            ctc_labels_batch['input_ids'] = ctc_labels_batch['input_ids'].masked_fill(ctc_labels_batch.attention_mask.ne(1), -100)
            batch['ctc_labels'] = ctc_labels_batch

        # Batch features related to text data
        if 'transcript' in self.load:
            # process source text features using the tokenizer
            source_batch = self.tokenizer(
                [feature['text']['source'] for feature in features],
                padding = self.padding,
                truncation = True,
                max_length = self.max_length_text,
                pad_to_multiple_of = self.pad_to_multiple_of_text,
                return_tensors = 'pt'
            )
            batch['trans_token_ids'] = source_batch['input_ids']
            batch['trans_token_ids_mask'] = source_batch['attention_mask']
        if 'description' in self.load:
            # process target text features
            target_batch = self.tokenizer(
                [feature['text']['target'] for feature in features],
                padding = self.padding,
                truncation = True,
                max_length = self.max_length_target_labels,
                pad_to_multiple_of = self.pad_to_multiple_of_target_labels,
                return_tensors = 'pt'
            )
            # replace padding with -100 in the labels to ignore loss correctly and discard BOS
            batch['labels'] = target_batch['input_ids'].masked_fill(target_batch.attention_mask.ne(1), -100)[:, 1:].contiguous()
            # remove the EOS for the decoder input ids
            non_eos_indices = torch.stack([torch.argwhere(x != self.tokenizer.eos_token_id).view(-1) for x in target_batch.input_ids], dim=0)
            batch['decoder_input_ids'] = target_batch.input_ids.gather(1, non_eos_indices)
            batch['decoder_input_ids_mask'] = target_batch.attention_mask.gather(1, non_eos_indices).type_as(batch['labels'])

        # Batch features related to pre-processed embeddings
        if 'pre-processed' in self.load:
            # Get maximum sequence lengths of pre-processed embeddings
            max_audio_embedding_length = max(feature['audio_embeddings']['embeddings'].size(dim=0) for feature in features)
            if self.text_embedding_padding == 'max_length':
                max_text_embedding_length = self.text_embedding_max_length
            else:
                max_text_embedding_length = max(feature['text_embeddings']['embeddings'].size(dim=0) for feature in features)

            # Necessary data for text embeddings EOS prediction
            total_text_embeddings = sum(feature['text_embeddings']['embeddings'].size(dim=0) for feature in features)

            for key in ['audio_embeddings', 'audio_embeddings_mask', 'text_embeddings_inputs', 'text_embeddings', 'text_embeddings_mask', 'text_embeddings_eos', 'text_embeddings_eos_weights']:
                batch[key] = []

            # Process each entry of the batch separately
            for feature in features:
            
                # Audio embeddings related data
                audio_embeddings_length = feature['audio_embeddings']['embeddings'].size(dim=0)
                batch['audio_embeddings'].append(torch.cat((feature['audio_embeddings']['embeddings'], torch.full((max_audio_embedding_length - audio_embeddings_length, feature['audio_embeddings']['embeddings'].size(dim=1)), self.audio_embedding_pad_value)), dim=0))
                batch['audio_embeddings_mask'].append(torch.cat((torch.full((audio_embeddings_length, ), True, dtype=torch.bool), torch.full((max_audio_embedding_length - audio_embeddings_length, ), False, dtype=torch.bool)), dim=0))

                # Text embeddings related data
                text_embeddings_length = feature['text_embeddings']['embeddings'].size(dim=0)
                batch['text_embeddings_inputs'].append(torch.cat((torch.full((1, feature['text_embeddings']['embeddings'].size(dim=1)), self.text_embedding_pad_value), feature['text_embeddings']['embeddings'][:-1], torch.full((max_text_embedding_length - text_embeddings_length, feature['text_embeddings']['embeddings'].size(dim=1)), self.text_embedding_pad_value)), dim=0))
                batch['text_embeddings'].append(torch.cat((feature['text_embeddings']['embeddings'], torch.full((max_text_embedding_length - text_embeddings_length, feature['text_embeddings']['embeddings'].size(dim=1)), self.text_embedding_pad_value)), dim=0))
                batch['text_embeddings_mask'].append(torch.cat((torch.full((text_embeddings_length, ), True, dtype=torch.bool), torch.full((max_text_embedding_length - text_embeddings_length, ), False, dtype=torch.bool)), dim=0))
                batch['text_embeddings_eos'].append(torch.cat((torch.full((text_embeddings_length - 1, ), 0.), torch.full((1, ), 1.), torch.full((max_text_embedding_length - text_embeddings_length, ), 0.)), dim=0))
                batch['text_embeddings_eos_weights'].append(torch.cat((torch.full((text_embeddings_length - 1, ), 0.5 / (total_text_embeddings - batch['batch_size'])), torch.tensor([0.5 / batch['batch_size']]), torch.full((max_text_embedding_length - text_embeddings_length, ), 0.)), 0))

            # Concat everything
            for key in ['audio_embeddings', 'audio_embeddings_mask', 'text_embeddings_inputs', 'text_embeddings', 'text_embeddings_mask', 'text_embeddings_eos', 'text_embeddings_eos_weights']:
                batch[key] = torch.stack(batch[key])
        
        #  Add pre-processed embeddings statistics
        if (features[0]['audio_embeddings'].get('mean', None) is not None) and (features[0]['text_embeddings'].get('mean', None) is not None):            
            batch['norm'] = [torch.stack([feature[key1][key2] for feature in features]) for key1 in ['audio_embeddings', 'text_embeddings'] for key2 in ['mean', 'std']]

        return batch
