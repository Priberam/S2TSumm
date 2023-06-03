import torch
import torch.nn.functional as F
from torch.optim import Adam
from transformers import EncoderDecoderModel, EncoderDecoderConfig, RobertaTokenizerFast
from src.models.parent import ParentModule
import json
import numpy as np
import nltk
from datasets import load_metric


class TextSummarizer(ParentModule):
    def __init__(
        self, 
        text_summ_backbone: str = 'mrm8488/camembert2camembert_shared-finetuned-french-summarization',
        freeze_encoder: bool = False,
        batch_size: int = 10,
        accumulate_grad_batches: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        max_length: int = 128,
        min_length: int = 5,
        num_beams: int = 1,
        predictions_file: str = 'predictions.jsonl'
    ):
        super().__init__()

        # Save hyperparameters
        self.save_hyperparameters(ignore=['predictions_file'])    
        
        # Load text tokenizer with the same checkpoint as the text summarizer
        self.tokenizer = RobertaTokenizerFast.from_pretrained(text_summ_backbone)

        model_config = EncoderDecoderConfig.from_pretrained(text_summ_backbone)
        model_config.pad_token_id = self.tokenizer.pad_token_id
        # Load the text summarizer and freeze the encoder if required; the embedding layer of the decoder is automatically frozen
        self.model = EncoderDecoderModel.from_pretrained(self.hparams.text_summ_backbone, config=model_config)
        if self.hparams.freeze_encoder:
            for param in self.model.encoder.parameters():
                param.requires_grad = False 

        # Predictions file
        self.predictions_file = predictions_file
        
        # Load metrics
        self.rouge_metric = load_metric('rouge')

    def forward(
        self, 
        input_ids=None, 
        labels=None
    ):
        # The decoder input ids are automatically created using the labels
        return self.model(input_ids=input_ids, labels=labels)

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr = self.hparams.learning_rate,
            betas = (self.hparams.beta_1, self.hparams.beta_2),
            weight_decay = self.hparams.weight_decay
        )

    def training_step(self, batch, batch_idx): 
        # Forward batch through the text summarizer
        output = self(
            input_ids = batch['trans_token_ids'], 
            labels = batch['labels']
        )
        batch_size = batch['trans_token_ids'].size(dim=0)

        # Log loss and return
        self.log('train/loss', output.loss, batch_size = batch_size * self.hparams.accumulate_grad_batches)

        return output.loss

    def validation_step(self, batch, batch_idx):
        # Forward batch through the text summarizer
        output = self(
            input_ids = batch['trans_token_ids'], 
            labels = batch['labels']
        )
        batch_size = batch['trans_token_ids'].size(dim=0)
        
        # Log loss
        self.log('val/loss', output.loss, batch_size=batch_size, on_epoch=True)

        # Generate predictions to evaluate metrics
        preds = self.model.generate(
            inputs = batch['trans_token_ids'], 
            attention_mask = batch['trans_token_ids_mask'], 
            max_length = self.hparams.max_length,
            num_beams = 1,
            bos_token_id = self.tokenizer.bos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id
        )
        preds = self.postprocess_text(self.tokenizer.batch_decode(preds, skip_special_tokens=True))
        targets = self.postprocess_text(self.tokenizer.batch_decode(torch.where(batch['labels'] == -100, self.tokenizer.pad_token_id, batch['labels'] ), skip_special_tokens=True))
        rouge_scores = self.rouge_metric.compute(predictions=preds, references=targets, use_stemmer=True, use_agregator=False)

        # Step output
        step_output = {'batch_size': batch_size}
        for key, value in rouge_scores.items():
            step_output['metric/' + key] = sum(x.fmeasure * 100 for x in value) / len(value)

        return step_output

    def validation_epoch_end(self, validation_step_outputs):
        # Log metrics
        metrics = {}
        num_examples = sum(step_output['batch_size'] for step_output in validation_step_outputs)
        for key in validation_step_outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(step_output[key] * step_output['batch_size'] for step_output in validation_step_outputs) / num_examples
        self.log_dict(metrics)

    def test_step(self, batch, batch_idx):
        # Generate predictions with the combined model
        preds = self.model.generate(
            inputs = batch['trans_token_ids'], 
            attention_mask = batch['trans_token_ids_mask'], 
            max_length = self.hparams.max_length,
            num_beams = self.hparams.num_beams,
            bos_token_id = self.tokenizer.bos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id
        )
        batch_size = batch['trans_token_ids'].size(dim=0)
        preds = self.postprocess_text(self.tokenizer.batch_decode(preds, skip_special_tokens=True))
        targets = self.postprocess_text(self.tokenizer.batch_decode(torch.where(batch['labels'] == -100, self.tokenizer.pad_token_id, batch['labels'] ), skip_special_tokens=True))
        rouge_scores = self.rouge_metric.compute(predictions=preds, references=targets, use_stemmer=True, use_agregator=False)

        # Step output
        step_output = {'batch_size': batch_size}
        for key, value in rouge_scores.items():
            step_output['metric/' + key] = sum(x.fmeasure * 100 for x in value) / len(value)

        return step_output

    def test_epoch_end(self, test_step_outputs):
        # Log metrics
        metrics = {}
        num_examples = sum(step_output['batch_size'] for step_output in test_step_outputs)
        for key in test_step_outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(step_output[key] * step_output['batch_size'] for step_output in test_step_outputs) / num_examples
        for key in test_step_outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics['sigma_' + key] = np.sqrt(sum((step_output[key] - metrics[key]) * (step_output[key] - metrics[key]) * step_output['batch_size'] for step_output in test_step_outputs) / num_examples)
        self.log_dict(metrics)

    def on_predict_start(self):
        # Open output file
        self._predict_f = open(self.predictions_file, 'w', encoding='utf-8')

    def predict_step(self, batch, batch_idx):
        # Generate predictions from the transcripts
        preds = self.model.generate(
            inputs=batch['input_ids'], 
            attention_mask=batch['input_ids_mask'], 
            max_length=self.config.max_length,
            num_beams=self.config.num_beams,
            bos_token_id=self.config.bos_token_id,
            pad_token_id=self.config.pad_token_id,
            eos_token_id=self.config.eos_token_id)

        # Convert -100 from the labels to the pad token id
        batch['labels'][batch['labels'] == -100] = self.config.pad_token_id

        # Loop through every element of the batch
        for target, pred in zip(batch['labels'].tolist(), preds.tolist()):
            example = {
                'gold': self.tokenizer.decode(target, skip_special_tokens=True),
                'prediction': self.tokenizer.decode(pred, skip_special_tokens=True)
            }
            self._predict_f.write(json.dumps(example, indent=4, ensure_ascii=False) + '\n')

    def on_predict_end(self):
        # Close output file
        self._predict_f.close()

    @staticmethod
    def postprocess_text(inputs):
        inputs = [inpt.strip() for inpt in inputs]

        # rougeLSum expects newline after each sentence
        inputs = ["\n".join(nltk.sent_tokenize(inpt)) for inpt in inputs]

        return inputs

    