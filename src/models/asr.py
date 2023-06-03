import torch
import json
from src.models.parent import ParentModule
from transformers import Wav2Vec2ForCTC, Wav2Vec2Config, Wav2Vec2Processor, Wav2Vec2FeatureExtractor, Wav2Vec2CTCTokenizer
from torch.optim import Adam
from datasets import load_metric
import random


class ASR(ParentModule):
    def __init__(
        self,
        ctc_vocab_file: str = 'vocabs/c_vocab.json',
        ctc_delimeter: str = ' ',
        backbone: str = 'LeBenchmark/wav2vec2-FR-7K-base',
        batch_size: int = 1,
        accumulate_grad_batches: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        predictions_file: str = './predictions.jsonl'
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters()

        # Instantiate Wav2Vec2 processor
        self.processor = Wav2Vec2Processor(
            feature_extractor = Wav2Vec2FeatureExtractor(
                feature_size = 1,
                sampling_rate = 16000,
                padding_value = 0.0,
                do_normalize = True,
                return_attention_mask = False
            ),
            tokenizer = Wav2Vec2CTCTokenizer(
                vocab_file = ctc_vocab_file,
                unk_token = '<unk>',
                pad_token = '<pad>'
            )
        )

        # Instantiate Wav2Vec2ForCTC model and update some parameters from its config
        wav2vec2_config = Wav2Vec2Config.from_pretrained(backbone)
        wav2vec2_config.vocab_size = self.processor.tokenizer.vocab_size
        wav2vec2_config.ctc_loss_reduction = 'mean'
        wav2vec2_config.pad_token_id = self.processor.tokenizer.pad_token_id
        wav2vec2_config.ctc_zero_infinity = True
        self.wav2vec2 = Wav2Vec2ForCTC.from_pretrained(backbone, config=wav2vec2_config)
        self.wav2vec2.freeze_feature_encoder()

        # Load metrics
        self.wer_metric = load_metric('wer')

    def forward(
        self, 
        input_values, 
        ctc_labels = None,
        output_hidden_states = False,
        return_dict = True
    ):
        
        return self.wav2vec2(
            input_values = input_values,
            labels = ctc_labels,
            output_hidden_states = output_hidden_states,
            return_dict = return_dict
        )

    def training_step(self, batch, batch_idx):
        # Forward the input values through the acoustic encoder
        output = self(
            input_values = batch['input_values'],
            ctc_labels = batch['ctc_labels'].input_ids,
            output_hidden_states = False,
            return_dict = True
        )
        batch_size = batch['input_values'].size(dim=0)
        
        # Log relevant metrics
        self.log('train/ctc_loss', output.loss, batch_size=batch_size)

        return output.loss

    def validation_step(self, batch, batch_idx):
        # Forward the input values through the acoustic encoder
        output = self(
            input_values = batch['input_values'],
            ctc_labels = batch['ctc_labels'].input_ids,
            output_hidden_states = False,
            return_dict = True
        )
        batch_size = batch['input_values'].size(dim=0)

        # Log relevant metrics
        self.log('val/ctc_loss', output.loss, batch_size=batch_size)

        # Save prediction and target for WER computation
        predicted_ids = torch.argmax(output.logits, dim=-1)
        preds = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        # Convert -100 from the labels to the pad token id and decode target
        batch['ctc_labels'].input_ids = batch['ctc_labels'].input_ids.masked_fill(batch['ctc_labels'].attention_mask.ne(1), self.processor.tokenizer.pad_token_id)
        targets = self.processor.batch_decode(batch['ctc_labels'].input_ids, skip_special_tokens=True)

        # Output for later processing
        step_output = {
            'batch_size': batch_size,
            'preds': preds,
            'targets': targets
        }
        
        return step_output

    def validation_epoch_end(self, validation_step_outputs):
        # Compute WER
        wer_score = self.wer_metric.compute(
            predictions = [pred.replace(self.hparams.ctc_delimeter, ' ') for step_output in validation_step_outputs for pred in step_output['preds']], 
            references = [target.replace(self.hparams.ctc_delimeter, ' ') for step_output in validation_step_outputs for target in step_output['targets']]
        )
        # Log metrics
        metrics = {
            'metrics/wer': wer_score
        }        
        self.log_dict(metrics)       

    def test_step(self, batch, batch_idx):
        # Forward the input values through the acoustic encoder
        output = self(
            input_values = batch['input_values'],
            ctc_labels = batch['ctc_labels'].input_ids,
            output_hidden_states = False,
            return_dict = True
        )
        batch_size = batch['input_values'].size(dim=0)

        # Save prediction and target for WER computation
        predicted_ids = torch.argmax(output.logits, dim=-1)
        preds = [pred.replace(self.hparams.ctc_delimeter, ' ').strip() for pred in self.processor.batch_decode(predicted_ids, skip_special_tokens=True)]
        # Convert -100 from the labels to the pad token id and decode target
        batch['ctc_labels'].input_ids = batch['ctc_labels'].input_ids.masked_fill(batch['ctc_labels'].attention_mask.ne(1), self.processor.tokenizer.pad_token_id)
        targets = [target.replace(self.hparams.ctc_delimeter, ' ').strip() for target in self.processor.batch_decode(batch['ctc_labels'].input_ids, skip_special_tokens=True)]

        # Output for later processing
        step_output = {
            'batch_size': batch_size,
            'preds': preds,
            'targets': targets
        }
        
        return step_output

    def test_epoch_end(self, test_step_outputs):
        # w/o text normalisation
        preds_wo = [pred for test_step_output in test_step_outputs for pred in test_step_output['preds']]
        targets_wo = [target for test_step_output in test_step_outputs for target in test_step_output['targets']]
        
        # Compute mean hypotheses
        wer_hypo_wo = self.wer_metric.compute(predictions=preds_wo, references=targets_wo)

        # Bootstrap
        r_wer_wo = []
        num_resamples = 100
        for r_idx in range(num_resamples):
            r_indices = random.choices(range(0, len(preds_wo)), k=len(preds_wo))
            r_preds = [preds_wo[idx] for idx in r_indices]
            r_targets = [targets_wo[idx] for idx in r_indices]
            r_wer_wo.append(self.wer_metric.compute(predictions=r_preds, references=r_targets))
        r_wer_wo = sorted(r_wer_wo)         

        metrics = {}
        metrics['wer'] = wer_hypo_wo
        metrics['wer_lb'] = r_wer_wo[4]
        metrics['wer_ub'] = r_wer_wo[94]
        # Log metrics    
        self.log_dict(metrics)         

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr = self.hparams.learning_rate,
            betas = (self.hparams.beta_1, self.hparams.beta_2),
            weight_decay = self.hparams.weight_decay
        )

    def on_predict_start(self):
        # Open output file
        self._predict_f = open(self.hparams.predictions_file, 'w', encoding='utf-8')

    def predict_step(self, batch, batch_idx):
        # Forward the input values through the acoustic encoder
        output = self(
            input_values = batch['input_values'],
            output_hidden_states = False,
            return_dict = True
        )
        predicted_ids = torch.argmax(output.logits, dim=-1)

        # Transcribe speech
        preds = self.processor.batch_decode(predicted_ids, skip_special_tokens=True)
        # Convert -100 from the labels to the pad token id and decode target
        batch['ctc_labels'].input_ids = batch['ctc_labels'].input_ids.masked_fill(batch['ctc_labels'].attention_mask.ne(1), self.processor.tokenizer.pad_token_id)
        targets = self.processor.batch_decode(batch['ctc_labels'].input_ids, skip_special_tokens=True)

        # Loop through every element of the batch and print predictions
        for pred, target in zip(preds, targets):
            example = {
                'gold': target,
                'pred': pred
            }
            self._predict_f.write(json.dumps(example, indent=4, ensure_ascii=False) + '\n')

    def on_predict_end(self):
        # Close output file
        self._predict_f.close()