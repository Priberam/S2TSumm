import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from src.models.parent import ParentModule
from transformers import EncoderDecoderModel, PretrainedConfig, EncoderDecoderConfig, RobertaTokenizerFast, CamembertForMaskedLM
from src.models.adapter import LSTMCrossModalAdapter
from src.models.text_summ import TextSummarizer
from typing import Optional
from transformers import LogitsProcessorList, MinLengthLogitsProcessor, BeamSearchScorer, StoppingCriteriaList, MaxLengthCriteria
import json
import numpy as np
import nltk
from datasets import load_metric


class AudioEncoder(nn.Module):
    def __init__(
        self, 
        adapter_config_file: str = 'lstm-adapter.json',
        adapter_ckpt: Optional[str] = None,
        adapter_eos_attention_window_size: int = 2,
        emb_loss_type: str = 'mse'
    ):
        super(AudioEncoder, self).__init__()   

        # Load adapter model with given checkpoint if provided
        self.adapter = LSTMCrossModalAdapter(
            config_file = adapter_config_file,
            emb_loss_type = emb_loss_type,
            eos_attention_window_size = adapter_eos_attention_window_size
        )
        if adapter_ckpt is not None:
            self.adapter = LSTMCrossModalAdapter.load_from_checkpoint(
                adapter_ckpt, 
                strict = True, 
                config_file = adapter_config_file,
                emb_loss_type = emb_loss_type,
                eos_attention_window_size = adapter_eos_attention_window_size
            )

    def forward(
        self, 
        inputs_embeds=None, 
        attention_mask=None, 
        decoder_inputs_embeds=None, 
        decoder_attention_mask=None, 
        labels_emb=None,
        labels_eos=None,
        labels_eos_weights=None,
        norm=None,
        teacher_forcing=True,
        tf_ratio=None
    ):
        '''
        inputs_embeds: input audio embeddings
        attention_mask: input audio embeddings attention mask
        decoder_inputs_embeds: decoder input text embeddings
        decoder_attention_mask: decoder input text embeddings attention mask
        labels_emb: target text embeddings
        labels_eos: target decoder eos probabilities
        labels_eos_weights: weights for loss of eos
        norm: mean and std embeddings for both the audio and text representation spaces; tuple (audio_mean, audio_std, text_mean, text_std)
        teacher_forcing: whether or not to use teacher forcing in the adapter decoder
        tf_ratio: to be used when teacher forcing is off
        '''

        # Normalize the input embeddings
        inputs_embeds = (inputs_embeds - norm[0].unsqueeze(dim=1)) / norm[1].unsqueeze(dim=1)

        # Forward batch through the adapter, possibly using teacher forcing mask
        if teacher_forcing:
           output = self.adapter(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask, 
                decoder_inputs_embeds=decoder_inputs_embeds, 
                decoder_attention_mask=decoder_attention_mask,
                labels_emb=labels_emb,
                labels_eos=labels_eos,
                labels_eos_weights=labels_eos_weights)       
        else:
            # Get teacher forcing mask and forward through the adapter
            teacher_forcing_mask = self.adapter.peeling_back_mask(decoder_attention_mask, tf_ratio=tf_ratio)
            output = self.adapter.forward_masked_teacher_forcing(
                inputs_embeds=inputs_embeds, 
                attention_mask=attention_mask, 
                decoder_inputs_embeds=decoder_inputs_embeds, 
                decoder_attention_mask=decoder_attention_mask,
                labels_emb=labels_emb,
                labels_eos=labels_eos,
                labels_eos_weights=labels_eos_weights,
                teacher_forcing_mask=teacher_forcing_mask)

        # Denormalize the output text embeddings
        output['embeddings'] = output['embeddings'] * norm[3].unsqueeze(dim=1) + norm[2].unsqueeze(dim=1)

        return output        

    def generate(
        self, 
        inputs_embeds = None, 
        attention_mask = None, 
        eos_inference = 'naive', 
        ground_truth_lengths = None, 
        eos_threshold = 0.5,
        norm = None,
        padding = 'max_length',
        max_length = 512
    ):
        '''
        inputs_embeds: input audio embeddings
        attention_mask: input audio embeddings attention mask
        eos_inference: use 'ground_truth_lengths' or a 'naive' approach
        ground_truth_lengths: ground truth lengths
        norm: mean and std embeddings for both the audio and text representation spaces; tuple (audio_mean, audio_std, text_mean, text_std)
        '''

        # Normalize the input embeddings
        inputs_embeds = (inputs_embeds - norm[0].unsqueeze(dim=1)) / norm[1].unsqueeze(dim=1)

        # Generate text embeddings using the adapter
        # Output is a dictionary with keys ('embeddings', 'attention_mask')        
        # Check if ground truth lengths are attainable if required
        assert (ground_truth_lengths is not None) or (eos_inference != 'ground_truth_lengths'), f'using ground truth lengths in the adapter, but none were provided'
        output = self.adapter.generate(
            inputs_embeds = inputs_embeds, 
            attention_mask = attention_mask,
            eos_inference = eos_inference,
            ground_truth_lengths = ground_truth_lengths,
            eos_threshold = eos_threshold,
            padding = padding,
            max_length = max_length
        )

        # Denormalize the output embeddings
        output['embeddings'] = output['embeddings'] * norm[3].unsqueeze(dim=1) + norm[2].unsqueeze(dim=1)

        return output


class CamembertDecoder(CamembertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)

    def prepare_inputs_for_generation(self, input_ids, past=None, attention_mask=None, **model_kwargs):
        input_shape = input_ids.shape
        # if model is used as a decoder in encoder-decoder model, the decoder attention mask is created on the fly
        if attention_mask is None:
            attention_mask = input_ids.new_ones(input_shape)

        # cut decoder_input_ids if past is used
        if past is not None:
            input_ids = input_ids[:, -1:]

        # get encoder outputs
        encoder_outputs = model_kwargs.get("encoder_outputs", None)
        assert encoder_outputs is not None, f'you must provide encoder outputs for the generation'

        return {"input_ids": input_ids, "attention_mask": attention_mask, "encoder_hidden_states": encoder_outputs["hidden_states"], "encoder_attention_mask": encoder_outputs["attentions"]}


class TextDecoder(nn.Module):
    def __init__(
        self, 
        backbone: str = 'mrm8488/camembert2camembert_shared-finetuned-french-summarization',
        text_summ_ckpt: Optional[str] = None
    ):
        super(TextDecoder, self).__init__()

        # Load text summarizer decoder with given checkpoint if provided
        # The decoder is an object of the class transformers.CamembertForMaskedLM
        model_config = EncoderDecoderConfig.from_pretrained(backbone)
        decoder = TextSummarizer(text_summ_backbone=backbone).model.decoder
        if text_summ_ckpt is not None:
            decoder = TextSummarizer.load_from_checkpoint(
                text_summ_ckpt, 
                text_summ_backbone=backbone
            ).model.decoder

        # Save the weights in the wrapper CamembertDecoder so as to enable generation with encoder hidden states
        self.decoder = CamembertDecoder(model_config.decoder)
        source = dict(decoder.named_parameters())
        target = dict(self.decoder.named_parameters())
        for part in source.keys():
            target[part].data.copy_(source[part].data)  

    def forward(
        self, 
        input_ids=None, 
        attention_mask=None, 
        encoder_hidden_states=None,
        encoder_attention_mask=None, 
        labels=None
    ):
        '''
        input_ids: decoder input ids
        attention_mask: decoder input ids attention mask
        encoder_hidden_states: the encoder hidden states used for attention
        encoder_attention_mask: the mask for the encoder hidden states
        labels: the target ids of the labels
        '''
        return self.decoder(
            input_ids = input_ids, 
            attention_mask = attention_mask, 
            encoder_hidden_states = encoder_hidden_states, 
            encoder_attention_mask = encoder_attention_mask, 
            labels = labels,
            return_dict = True
        )

class SpeechSummarizer(ParentModule):
    def __init__(
        self,
        adapter_config_file: str = 'src/models/lstm-adapter.json',
        adapter_ckpt: Optional[str] = None,
        adapter_eos_attention_window_size: int = 2,
        eos_threshold: float = 0.5,
        eos_inference: str = 'naive',
        text_embedding_max_length: int = 512,
        text_embedding_pad_value: float = 0.,
        text_summ_backbone: str = 'mrm8488/camembert2camembert_shared-finetuned-french-summarization',
        text_summ_ckpt: Optional[str] = None,
        batch_size: int = 10,
        accumulate_grad_batches: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        teacher_forcing: bool = True,
        peeling_back: bool = False,
        tf_ratio_offset: float = 1.,
        tf_ratio_slope: float = 0.,     
        emb_loss_type: str = 'mse',   
        emb_loss_weight: float = 0.,
        eos_loss_weight: float = 0.,
        sum_loss_weight: float = 1.,
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

        # Load encoder and decoder  
        self.encoder = AudioEncoder(
            adapter_config_file = self.hparams.adapter_config_file,
            adapter_ckpt = self.hparams.adapter_ckpt,
            adapter_eos_attention_window_size = self.hparams.adapter_eos_attention_window_size,
            emb_loss_type = self.hparams.emb_loss_type
        ) 
        self.decoder = TextDecoder(
            backbone = self.hparams.text_summ_backbone,
            text_summ_ckpt = self.hparams.text_summ_ckpt
        )

        # Teacher forcing ratio initialization
        self.tf_ratio = self.hparams.tf_ratio_offset

        # Predictions file
        self.predictions_file = predictions_file

        # Load the complete text summarizer for comparison
        self.text_summ = TextSummarizer(text_summ_backbone=self.hparams.text_summ_backbone).model
        if self.hparams.text_summ_ckpt is not None:
            self.text_summ = TextSummarizer.load_from_checkpoint(
                self.hparams.text_summ_ckpt, 
                text_summ_backbone=self.hparams.text_summ_backbone 
            ).model
        
        # Load metrics
        self.rouge_metric = load_metric('rouge')

    def forward(
        self, 
        inputs_embeds=None, 
        attention_mask=None, 
        cross_inputs_embeds=None, 
        cross_attention_mask=None, 
        labels_emb=None,
        labels_eos=None,
        labels_eos_weights=None,
        teacher_forcing=True,
        tf_ratio=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        labels=None,
        norm=None
    ):
        # Forward input through the audio encoder
        enc_out = self.encoder(
            inputs_embeds = inputs_embeds, 
            attention_mask = attention_mask, 
            decoder_inputs_embeds = cross_inputs_embeds, 
            decoder_attention_mask = cross_attention_mask, 
            labels_emb = labels_emb,
            labels_eos = labels_eos,
            labels_eos_weights = labels_eos_weights,
            norm = norm,
            teacher_forcing = teacher_forcing,
            tf_ratio = tf_ratio
        )

        batch_size = inputs_embeds.size(dim=0)
        # The sequence length must have the text embeddings max length; pad if necessary
        if enc_out['embeddings'].size(dim=1) != self.hparams.text_embedding_max_length:
            enc_out['embeddings'] = torch.cat((enc_out['embeddings'], torch.full((batch_size, self.hparams.text_embedding_max_length - enc_out['embeddings'].size(dim=1), enc_out['embeddings'].size(dim=2)), self.hparams.text_embedding_pad_value, device=self.device)), dim=1)
        
        # Generate attention mask according to logits using a naive approach to infer the eos
        enc_out['attention_mask'] = torch.arange(self.hparams.text_embedding_max_length, dtype=torch.long, device=self.device).expand(batch_size, -1) <= torch.cat([min(torch.argwhere(x_ >= self.hparams.eos_threshold)) if len(torch.argwhere(x_ >= self.hparams.eos_threshold)) != 0 else torch.tensor([self.hparams.text_embedding_max_length - 1], dtype=torch.long, device=self.device) for x_ in enc_out['logits'].softmax(-1)], dim=0).unsqueeze(dim=1)
        
        # Forward the encoded hidden states and the decoder input ids through the text decoder
        dec_out = self.decoder(
            input_ids = decoder_input_ids, 
            attention_mask = decoder_attention_mask, 
            encoder_hidden_states = enc_out['embeddings'], 
            encoder_attention_mask = enc_out['attention_mask'],
            labels = labels
        )

        # Return encoder and decoder outputs
        return_dict = {
            'encoder' : enc_out,
            'decoder' : dec_out
        }

        return return_dict

    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr = self.hparams.learning_rate,
            betas = (self.hparams.beta_1, self.hparams.beta_2),
            weight_decay = self.hparams.weight_decay
        )

    def training_step(self, batch, batch_idx):

        # Update adapter teacher forcing ratio if peeling back active
        if self.hparams.peeling_back:
            # Update teacher forcing ratio at each optimization step; log teacher forcing ratio
            if ( (batch_idx != 0) and (batch_idx % self.hparams.accumulate_grad_batches == 0) ):
                self.tf_ratio = max(0., self.tf_ratio - self.hparams.tf_ratio_slope)

        # Forward batch through the model
        out = self(
            inputs_embeds = batch['audio_embeddings'], 
            attention_mask = batch['audio_embeddings_mask'], 
            cross_inputs_embeds = batch['text_embeddings_inputs'], 
            cross_attention_mask = batch['text_embeddings_mask'], 
            labels_emb = batch['text_embeddings'],
            labels_eos = batch['text_embeddings_eos'],
            labels_eos_weights = batch['text_embeddings_eos_weights'],
            teacher_forcing = self.hparams.teacher_forcing,
            tf_ratio = self.tf_ratio,
            decoder_input_ids = batch['decoder_input_ids'],
            decoder_attention_mask = batch['decoder_input_ids_mask'],
            labels = batch['labels'],
            norm = batch['norm']
        )

        # Get the several losses and compute a weighted sum
        loss_emb = out['encoder']['loss_emb']
        loss_eos = out['encoder']['loss_eos']
        loss_sum = out['decoder'].loss
        loss = self.hparams.emb_loss_weight * loss_emb + self.hparams.eos_loss_weight * loss_eos + self.hparams.sum_loss_weight * loss_sum

        batch_size = batch['audio_embeddings'].size(dim=0)
        # Log losses and other
        self.log('train/emb_loss', loss_emb, batch_size = batch_size * self.hparams.accumulate_grad_batches)
        self.log('train/eos_loss', loss_eos, batch_size = batch_size * self.hparams.accumulate_grad_batches)
        self.log('train/sum_loss', loss_sum, batch_size = batch_size * self.hparams.accumulate_grad_batches)
        self.log('train/loss', loss, batch_size = batch_size * self.hparams.accumulate_grad_batches)
        if not self.hparams.teacher_forcing:
            self.log('train/tf_ratio', self.tf_ratio, batch_size = batch_size * self.hparams.accumulate_grad_batches)

        # Return loss
        return loss

    def validation_step(self, batch, batch_idx):

        # Forward batch through the model
        out = self(
            inputs_embeds = batch['audio_embeddings'], 
            attention_mask = batch['audio_embeddings_mask'], 
            cross_inputs_embeds = batch['text_embeddings_inputs'], 
            cross_attention_mask = batch['text_embeddings_mask'], 
            labels_emb = batch['text_embeddings'],
            labels_eos = batch['text_embeddings_eos'],
            labels_eos_weights = batch['text_embeddings_eos_weights'],
            teacher_forcing = self.hparams.teacher_forcing,
            tf_ratio = 0.,
            decoder_input_ids = batch['decoder_input_ids'],
            decoder_attention_mask = batch['decoder_input_ids_mask'],
            labels = batch['labels'],
            norm = batch['norm']
        )

        # Get the several losses and compute a weighted sum
        loss_emb = out['encoder']['loss_emb']
        loss_eos = out['encoder']['loss_eos']
        loss_sum = out['decoder'].loss
        loss = self.hparams.emb_loss_weight * loss_emb + self.hparams.eos_loss_weight * loss_eos + self.hparams.sum_loss_weight * loss_sum

        batch_size = batch['audio_embeddings'].size(dim=0)
        # Log losses and other
        self.log('val/emb_loss', loss_emb, batch_size=batch_size, on_epoch=True)
        self.log('val/eos_loss', loss_eos, batch_size=batch_size, on_epoch=True)
        self.log('val/sum_loss', loss_sum, batch_size=batch_size, on_epoch=True)
        self.log('val/loss', loss, batch_size=batch_size, on_epoch=True)
        if not self.hparams.teacher_forcing:
            self.log('val/tf_ratio', self.tf_ratio, batch_size=batch_size, on_epoch=True)

        # Intermediate results for ROUGE scores
        preds = self.generate(
            inputs_embeds = batch['audio_embeddings'], 
            attention_mask = batch['audio_embeddings_mask'], 
            eos_inference = 'naive',
            eos_threshold = self.hparams.eos_threshold,
            norm = batch['norm'],
            num_beams = 1,
            min_length = self.hparams.min_length,
            max_length = self.hparams.max_length
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

    @staticmethod
    def postprocess_text(inputs):
        inputs = [inpt.strip() for inpt in inputs]

        # rougeLSum expects newline after each sentence
        inputs = ["\n".join(nltk.sent_tokenize(inpt)) for inpt in inputs]

        return inputs

    def generate(
        self, 
        inputs_embeds=None, 
        attention_mask=None, 
        eos_inference=None,
        ground_truth_lengths=None,
        eos_threshold=None,
        norm=None, 
        num_beams = 1,
        min_length = 5,
        max_length = 128
    ):
        '''
        inputs_embeds: input audio embeddings
        attention_mask: input audio embeddings attention mask
        eos_inference: the type of eos inference
        ground_truth_lengths: ground truth lengths
        eos_threshold: the eos threshold probability
        norm: mean and std embeddings for both the audio and text representation spaces; tuple (audio_mean, audio_std, text_mean, text_std)
        num_beams: 
        min_length:
        max_length: 
        '''

        # Get relevant token ids
        bos_token_id = self.tokenizer.bos_token_id
        pad_token_id = self.tokenizer.pad_token_id
        eos_token_id = self.tokenizer.eos_token_id

        # Forward inputs throught the audio encoder
        encoder_outputs = self.encoder.generate(
            inputs_embeds = inputs_embeds, 
            attention_mask = attention_mask, 
            eos_inference = eos_inference, 
            ground_truth_lengths = ground_truth_lengths, 
            eos_threshold = eos_threshold,
            norm = norm,
            padding = 'max_length',
            max_length = self.hparams.text_embedding_max_length
        )       

        # add encoder_outputs to model keyword arguments
        model_kwargs = {
            'encoder_outputs': {
                'hidden_states': encoder_outputs['embeddings'].repeat_interleave(num_beams, dim=0),
                'attentions': encoder_outputs['attention_mask'].repeat_interleave(num_beams, dim=0)
            }
        }

        batch_size = inputs_embeds.size(dim=0)
        # Instantiate the initial input ids
        input_ids = torch.ones((batch_size * num_beams, 1), device=self.device, dtype=torch.long)
        input_ids = input_ids * bos_token_id

        # Instantiate logits processors
        logits_processor = LogitsProcessorList([
                MinLengthLogitsProcessor(min_length, eos_token_id=eos_token_id)
        ])
        stopping_criteria = StoppingCriteriaList([MaxLengthCriteria(max_length=max_length)])

        # Greedy search
        if num_beams == 1:
             # Forward everything through the greedy search
            preds = self.decoder.decoder.greedy_search(
                input_ids = input_ids, 
                logits_processor = logits_processor, 
                stopping_criteria = stopping_criteria,
                pad_token_id = pad_token_id,
                eos_token_id = eos_token_id,
                **model_kwargs
            )
        # Beam search
        else:
            # Instantiate beam scorer
            beam_scorer = BeamSearchScorer(
                batch_size = batch_size,
                num_beams = num_beams,
                device = self.device,
            )

            # Forward everything through the beam search
            preds = self.decoder.decoder.beam_search(
                input_ids = input_ids, 
                beam_scorer = beam_scorer, 
                logits_processor = logits_processor, 
                stopping_criteria = stopping_criteria,
                pad_token_id = pad_token_id,
                eos_token_id = eos_token_id,
                **model_kwargs
            )

        return preds        

    def test_step(self, batch, batch_idx):
        # Generate predictions with the combined model
        preds = self.generate(
            inputs_embeds = batch['audio_embeddings'],
            attention_mask = batch['audio_embeddings_mask'],
            eos_inference = self.hparams.eos_inference,
            ground_truth_lengths = torch.sum(batch['text_embeddings_mask'], dim=1, keepdim=False),
            eos_threshold = self.hparams.eos_threshold,
            norm = batch['norm'], 
            num_beams = self.hparams.num_beams,
            min_length = self.hparams.min_length,
            max_length = self.hparams.max_length
        )

        batch_size = batch['audio_embeddings'].size(dim=0)
        preds = self.postprocess_text(self.tokenizer.batch_decode(preds, skip_special_tokens=True))
        targets = self.postprocess_text(self.tokenizer.batch_decode(torch.where(batch['labels'] == -100, self.tokenizer.pad_token_id, batch['labels'] ), skip_special_tokens=True))
        rouge_scores = self.rouge_metric.compute(predictions=preds, references=targets, use_stemmer=True, use_agregator=False)

        # Step output
        step_output = {'batch_size': batch_size}
        for key, value in rouge_scores.items():
            step_output['metric/' + key] = sum(x.fmeasure * 100 for x in value) / len(value)

        return step_output

    def test_epoch_end(self, outputs):
        # Log metrics
        metrics = {}
        num_examples = sum(step_output['batch_size'] for step_output in outputs)
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(step_output[key] * step_output['batch_size'] for step_output in outputs) / num_examples
        for key in outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics['sigma_' + key] = np.sqrt(sum((step_output[key] - metrics[key]) * (step_output[key] - metrics[key]) * step_output['batch_size'] for step_output in outputs) / num_examples)
        self.log_dict(metrics)

    def on_predict_start(self):
        # Open output file
        self._predict_f = open(self.predictions_file, 'w', encoding='utf-8')

    def predict_step(self, batch, batch_idx):
        # Generate predictions with the combined model
        preds = self.generate(
            inputs_embeds = batch['audio_embeddings'],
            attention_mask = batch['audio_embeddings_mask'],
            eos_inference = self.hparams.eos_inference,
            ground_truth_lengths = torch.sum(batch['text_embeddings_mask'], dim=1, keepdim=False),
            eos_threshold = self.hparams.eos_threshold,
            norm = batch['norm'], 
            num_beams = self.hparams.num_beams,
            min_length = self.hparams.min_length,
            max_length = self.hparams.max_length,
            padding = True
        )

        # Generate predictions with the text summarizer from the transcripts
        preds_ = self.text_summ.generate(
            inputs = batch['trans_token_ids'], 
            attention_mask = batch['trans_token_ids_mask'], 
            max_length = self.hparams.max_length,
            num_beams = self.hparams.num_beams,
            bos_token_id = self.tokenizer.bos_token_id,
            pad_token_id = self.tokenizer.pad_token_id,
            eos_token_id = self.tokenizer.eos_token_id
        )

        # Convert -100 from the labels to the pad token id
        batch['labels'][batch['labels'] == -100] = self.tokenizer.pad_token_id

        # Loop through every element of the batch
        for gold_ids, from_text_ids, from_audio_ids in zip(batch['labels'].tolist(), preds_.tolist(), preds.tolist()):
            example = {
                'gold': self.tokenizer.decode(gold_ids, skip_special_tokens=True),
                'from_transcript': self.tokenizer.decode(from_text_ids, skip_special_tokens=True),
                'from_audio': self.tokenizer.decode(from_audio_ids, skip_special_tokens=True)
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
