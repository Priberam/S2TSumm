import os
import torch
import numpy as np
from transformers import RobertaTokenizerFast, EncoderDecoderModel, EncoderDecoderConfig
from src.models.asr import ASR
from src.models.text_summ import TextSummarizer
from src.data.data_module import S2TSumDataMod
import nltk
from datasets import load_metric
from tqdm import tqdm
import argparse


def main(
    asr_ckpt = None,
    t2t_ckpt = None,
    root = None,
    split = 'test',
    gpu = True
):

    # Check if the provided checkpoints exist
    assert os.path.exists(asr_ckpt), f'the provided checkpoint for the ASR does not exist'
    assert os.path.exists(t2t_ckpt), f'the provided checkpoint for the T2T abstractive summarizer does not exist'

    # Check if the root folder for the dataset exists
    assert os.path.isdir(root), f'you must provide a valid root folder for the dataset'

    # Check if the split has a valid name
    assert split in ['train', 'dev', 'test'], f'you must provide a valid split for evaluation, got {split}'
    split = 'fit' if split == 'train' else 'validate' if split == 'dev' else 'test'

    # Set device to process the data
    device = 'cuda' if torch.cuda.is_available() and gpu else 'cpu'

    # Instantiate the Roberta tokenizer
    tokenizer = RobertaTokenizerFast.from_pretrained('mrm8488/camembert2camembert_shared-finetuned-french-summarization')

    # Load the ROUGE metric
    rouge_metric = load_metric('rouge')

    # Load the ASR system
    asr = ASR.load_from_checkpoint(asr_ckpt, ctc_vocab_file='vocabs/c_vocab.json').to(device)
    # Load the text-to-text summariser
    summariser = TextSummarizer.load_from_checkpoint(t2t_ckpt).model.to(device)

    # Load the data module
    data_module = S2TSumDataMod(
        ctc_vocab_file = 'vocabs/c_vocab.json',   
        ctc_delimeter = ' ',     
        tokenizer_ckpt = 'mrm8488/camembert2camembert_shared-finetuned-french-summarization',
        datasets = ['euronews'],
        roots = [root],
        batch_size = 1,
        normalize_waveform = True,
        load = ['waveform', 'transcript', 'description'],
        padding = True
    )
    data_module.setup(split)

    step_outputs = []
    with torch.no_grad():
        for batch in tqdm(data_module.test_dataloader()):

            # Forward the audio through the ASR system and obtain transcript
            asr_logits = asr.forward(
                batch['input_values'].to(device),
                output_hidden_states = False,
                return_dict = True
            ).logits
            predicted_ids = torch.argmax(asr_logits, dim=-1)
            transcript = data_module.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

            # Tokenize transcript
            tokenized_transcript = tokenizer(
                [transcript],
                padding = True,
                truncation = True,
                max_length = 512,
                return_tensors = 'pt'
            )

            # Generate predictions to evaluate metrics
            preds = summariser.generate(
                inputs = tokenized_transcript['input_ids'].to(device), 
                attention_mask = tokenized_transcript['attention_mask'].to(device), 
                max_length = 128,
                num_beams = 8,
                bos_token_id = tokenizer.bos_token_id,
                pad_token_id = tokenizer.pad_token_id,
                eos_token_id = tokenizer.eos_token_id
            )
            preds = postprocess_text(tokenizer.batch_decode(preds, skip_special_tokens=True))
            targets = postprocess_text(tokenizer.batch_decode(torch.where(batch['labels'] == -100, tokenizer.pad_token_id, batch['labels'] ), skip_special_tokens=True))
            rouge_scores = rouge_metric.compute(predictions=preds, references=targets, use_stemmer=True, use_agregator=False)

            # Step output
            step_output = {'batch_size': batch['input_values'].size(dim=0)}
            for key, value in rouge_scores.items():
                step_output['metric/' + key] = sum(x.fmeasure * 100 for x in value) / len(value)
            step_outputs.append(step_output)

        # Compute final results
        metrics = {}
        num_examples = sum(step_output['batch_size'] for step_output in step_outputs)
        for key in step_outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics[key] = sum(step_output[key] * step_output['batch_size'] for step_output in step_outputs) / num_examples
        for key in step_outputs[0].keys():
            if key == 'batch_size':
                continue
            metrics['sigma_' + key] = np.sqrt(sum((step_output[key] - metrics[key]) * (step_output[key] - metrics[key]) * step_output['batch_size'] for step_output in step_outputs) / num_examples)
        
        for key, value in metrics.items():
            print(key, value)


def postprocess_text(inputs):
    inputs = [inpt.strip() for inpt in inputs]

    # rougeLSum expects newline after each sentence
    inputs = ["\n".join(nltk.sent_tokenize(inpt)) for inpt in inputs]

    return inputs


def cli():

    # Create CLI parser
    parser = argparse.ArgumentParser(
        description = 'S2T Cascade Abstractive Summarizer', 
        formatter_class = argparse.ArgumentDefaultsHelpFormatter
    )    
    parser.add_argument('-a', '--asr_ckpt', default='', type=str, help='The path to the ASR checkpoint.')
    parser.add_argument('-t', '--t2t_ckpt', default='', type=str, help='The path to the T2T abstractive summarizer checkpoint.')
    parser.add_argument('-r', '--root', default='', type=str, help='The path to the root folder of the BNews dataset.')
    parser.add_argument('-g', '--gpu', dest='gpu', action='store_true', help='Use GPU if available.')
    parser.add_argument('-s', '--split', default='', type=str, help='Name of the split to be evaluated.')

    # Parse arguments
    args = parser.parse_args()

    main(
        asr_ckpt = args.asr_ckpt,
        t2t_ckpt = args.t2t_ckpt,
        root = args.root,
        split = args.split,
        gpu = args.gpu
    )

if __name__ == '__main__':

    cli()