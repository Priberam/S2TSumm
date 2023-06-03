from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
from .dataset import S2TSumDataset, prepare_dataset
from .data_collator import S2TSumDataCollator
from typing import List, Optional, Union
from transformers import Wav2Vec2Processor, RobertaTokenizerFast, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer


class S2TSumDataMod(LightningDataModule):
    def __init__(
        self,
        ctc_vocab_file: str = 'vocabs/c_vocab.json',
        ctc_delimeter: str = ' ',        
        tokenizer_ckpt: str = 'mrm8488/camembert2camembert_shared-finetuned-french-summarization',
        datasets: List[str] = ['euronews'],
        roots: List[str] = ['/mnt/SSD2/datasets/EuroNewsAudio/fr/'],
        batch_size: int = 1,
        normalize_waveform: bool = False,
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
        text_embedding_padding: Union[bool, str] = 'longest',
        text_embedding_max_length: Optional[int] = 512,
        **kwargs
    ):
        super().__init__()
        
        # Where to extract the data from plus the batch size for convenience
        self.datasets = datasets
        self.roots = roots        
        self.batch_size = batch_size

        # Simple check to what is to be loaded
        assert list(set(load) - set(['waveform', 'transcript', 'description', 'pre-processed'])) == [], f'check what you wish to load from the dataset, invalid keys'
        self.load = load

        # Instantiate Wav2Vec2 processor
        self.processor = Wav2Vec2Processor(
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size = 1,
                sampling_rate = 16000,
                padding_value = 0.0,
                do_normalize = normalize_waveform,
                return_attention_mask = False
            ),
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_file = ctc_vocab_file,
                unk_token = '<unk>',
                pad_token = '<pad>'
            )
        )

        # Instantiate the Roberta tokenizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained(tokenizer_ckpt)

        # Instantiate a prepare_dataset instance for easier manipulation of dataset raw data and a data collator to be used on the preprocessed data
        self.prepare_dataset = prepare_dataset(
            processor = self.processor, 
            tokenizer = self.tokenizer,
            delimeter = ctc_delimeter,
            load = self.load
        )
        self.collate_fn = S2TSumDataCollator(
            processor = self.processor,
            tokenizer = self.tokenizer,
            load = self.load,
            padding = padding,
            max_length_audio = max_length_audio,
            max_length_text = max_length_text,
            max_length_ctc_labels = max_length_ctc_labels,
            max_length_target_labels = max_length_target_labels,
            pad_to_multiple_of_audio = pad_to_multiple_of_audio,
            pad_to_multiple_of_text = pad_to_multiple_of_text,
            pad_to_multiple_of_ctc_labels = pad_to_multiple_of_ctc_labels,
            pad_to_multiple_of_target_labels = pad_to_multiple_of_target_labels,
            audio_embedding_pad_value = audio_embedding_pad_value,
            text_embedding_pad_value = text_embedding_pad_value,
            text_embedding_padding = text_embedding_padding,
            text_embedding_max_length= text_embedding_max_length
        )

    def setup(self, stage=None):

        # Assign val dataset for use in dataloader
        if stage == "validate" or stage is None:
            self.val_dataset = S2TSumDataset(datasets=self.datasets, roots=self.roots, split='dev', prepare_dataset=self.prepare_dataset)

        # Assign train datasets for use in dataloader
        if stage == "fit" or stage is None:
            self.fit_dataset = S2TSumDataset(datasets=self.datasets, roots=self.roots, split='train', prepare_dataset=self.prepare_dataset)
            self.val_dataset = S2TSumDataset(datasets=self.datasets, roots=self.roots, split='dev', prepare_dataset=self.prepare_dataset)
        
        # Assign val dataset for use in dataloader
        if stage == "test" or stage is None:
            self.test_dataset = S2TSumDataset(datasets=self.datasets, roots=self.roots, split='test', prepare_dataset=self.prepare_dataset)

        # Assign val datasets for use in dataloader
        if stage == "predict" or stage is None:
            self.pred_dataset = S2TSumDataset(datasets=self.datasets, roots=self.roots, split='dev', prepare_dataset=self.prepare_dataset)

    def train_dataloader(self):
        return DataLoader(self.fit_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)

    def predict_dataloader(self):
        return DataLoader(self.pred_dataset, batch_size=self.batch_size, collate_fn=self.collate_fn)