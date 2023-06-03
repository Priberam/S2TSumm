import torch as T
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from src.models.parent import ParentModule
from transformers import PretrainedConfig
from torch.optim import Adam


class LSTMEncoder(nn.Module):
    def __init__(self, config):
        super(LSTMEncoder, self).__init__()
        # Encoder convolutional layers
        self.conv1 = nn.Conv1d(config.wav2vec2_emb_dim, config.num_channels[0], config.kernel_size[0], stride=config.stride[0], dilation=config.dilation[0])
        self.conv2 = nn.Conv1d(config.num_channels[0], config.num_channels[1], config.kernel_size[1], stride=config.stride[1], dilation=config.dilation[1])

        # Encoder bidirectional LSTM
        self.lstm = nn.LSTM(input_size=config.num_channels[1], hidden_size=config.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)

        # Layer normalization to avoid increasingly larger hidden states norms
        self.layer_norm = nn.LayerNorm(2 * config.hidden_dim)

        # Linear layer to project the concatenation of the forward and backward final hidden state to the internal hidden dimension
        self.reduce_h = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # Linear layer to project the concatenation of the forward and backward final cell state to the internal hidden dimension
        self.reduce_c = nn.Linear(config.hidden_dim * 2, config.hidden_dim)

        # Configuration settings
        self.config = config

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, seq_lens):
        # Forward the input embeddings through the two convolutional layers
        x_reduced = self.conv1(T.swapaxes(x, 1, 2))
        x_reduced = T.swapaxes(self.conv2(x_reduced), 1, 2)
        reduced_seq_lens = (seq_lens - self.config.dilation[0] * (self.config.kernel_size[0] - 1) - 1) / self.config.stride[0] + 1
        reduced_seq_lens = (reduced_seq_lens.long() - self.config.dilation[1] * (self.config.kernel_size[1] - 1) - 1) / self.config.stride[1] + 1
        reduced_seq_lens = reduced_seq_lens.long()

        # Feeding to the LSTM a padded_sequence prevents it from considering the padded elements
        packed = pack_padded_sequence(x_reduced, reduced_seq_lens.tolist(), batch_first=True, enforce_sorted=False)
        enc_out, (h, c) = self.lstm(packed)
        enc_out,_ = pad_packed_sequence(enc_out, batch_first=True)
        enc_out = self.layer_norm(enc_out.contiguous())

        # Concatenate the forward and backward last hidden and cell states                                              
        h = T.cat(list(h), dim=1)                                   
        c = T.cat(list(c), dim=1)
        h_reduced = F.relu(self.reduce_h(h))                        
        c_reduced = F.relu(self.reduce_c(c))

        # Generate masks for the encoder output
        max_reduced_seq_len = T.max(reduced_seq_lens)
        enc_mask = []
        for seq_len in reduced_seq_lens:
            enc_mask.append(T.cat((T.full((seq_len, ), True, dtype=T.bool), T.full((max_reduced_seq_len - seq_len, ), False, dtype=T.bool)), dim=0))
        enc_mask = T.stack(enc_mask).type_as(reduced_seq_lens).bool()

        return enc_out, (h_reduced, c_reduced), enc_mask


class LSTMEncoderAttention(nn.Module):
    def __init__(self, config):
        super(LSTMEncoderAttention, self).__init__()
        # A bilinear function is used to compute the alignment scores
        self.w_attention = nn.Bilinear(config.hidden_dim, config.hidden_dim * 2, 1, bias=False)

        # Configuration settings
        self.config = config

    def forward(self, h_enc, h_dec, enc_mask, prev_e_enc):
        ''' Perform attention over encoder hidden states
        h_dec: decoder hidden state at current time step
        h_enc: encoder hidden states
        enc_mask: mask tensor with boolean values 
        prev_e_enc: contains the encoder alignment scores from previous decoder time steps
        '''
        
        # Get traditional alignment scores (eq 2 in https://arxiv.org/pdf/1705.04304.pdf)
        e_enc = self.w_attention(h_dec.unsqueeze(1).expand(-1, h_enc.size(dim=1), -1), h_enc).squeeze(2) # bs,n_seq

        # Set masked scores to -inf
        e_enc = e_enc.masked_fill(T.logical_not(enc_mask), -float('inf'))

        # Intra-temporal attention (eq 3 in https://arxiv.org/pdf/1705.04304.pdf)   
        if prev_e_enc is None:
            # Add encoder energy scores to memory
            prev_e_enc = e_enc.unsqueeze(1)

            # Calculate attention weights
            at_weights = F.softmax(e_enc, dim=1)

        else:
            # Find maximum energy score value among the previous ones excluding the current and set -inf to 0 in masked elements to avoid -inf - (-inf) = NaN
            e_maxima = T.maximum(e_enc, T.amax(prev_e_enc, dim=1, keepdim=False))
            e_maxima = e_maxima.masked_fill(T.logical_not(enc_mask), 0.)
           
            # Determine temporal scores
            e_enc_temp = T.exp(e_enc - e_maxima) / (T.exp(prev_e_enc - e_maxima.unsqueeze(1)).sum(1, keepdim=False) + 1e-3)

            # Calculate attention weights
            at_weights = e_enc_temp / (e_enc_temp.sum(1, keepdim=True) + 1e-3)
            
            # Add last encoder energy scores
            prev_e_enc = T.cat([prev_e_enc, e_enc.unsqueeze(1)], dim=1)

        # Compute encoder context vector
        context_enc = T.bmm(at_weights.unsqueeze(1), h_enc).squeeze(1)

        return context_enc, at_weights, prev_e_enc


class LSTMDecoderAttention(nn.Module):
    def __init__(self, config):
        super(LSTMDecoderAttention, self).__init__()     
        # A bilinear function is used to compute the alignment scores   
        self.w_attention = nn.Bilinear(config.hidden_dim, config.hidden_dim, 1, bias=False)

        # Configuration settings
        self.config = config

    def forward(self, h_dec, prev_h_dec, dec_mask):
        ''' Perform intra_decoder attention
        h_dec: hidden state of decoder at current time step
        prev_h_dec: contains list of previous decoder hidden states
        dec_mask: contains mask of previous decoder steps
        '''
        
        if prev_h_dec is None:
        # If first step, the context vector is identically zero
            context_dec = T.zeros_like(h_dec)
            prev_h_dec = h_dec.unsqueeze(1)               
        else:
            # Get traditional alignment scores (eq 2 in https://arxiv.org/pdf/1705.04304.pdf)
            e_dec = self.w_attention(h_dec.unsqueeze(1).expand_as(prev_h_dec), prev_h_dec).squeeze(2) # bs,n_seq
            # Set masked scores to -inf and subtract maxima for numerical stability
            e_dec = e_dec.masked_fill(T.logical_not(dec_mask), -float('inf')) 
            e_dec = e_dec - T.amax(e_dec, dim=1, keepdim=True).masked_fill(T.logical_not(dec_mask), 0.)

            # Intra-decoder attention (eq 7 & 8 in https://arxiv.org/pdf/1705.04304.pdf)
            # Calculate attention weights
            at_weights = F.softmax(e_dec, dim=1).unsqueeze(1)  #bs, 1, t-1

            # Compute decoder context vector
            context_dec = T.bmm(at_weights, prev_h_dec).squeeze(1)     #bs, n_hid

            prev_h_dec = T.cat([prev_h_dec, h_dec.unsqueeze(1)], dim=1)    #bs, t, n_hid

        return context_dec, prev_h_dec


class LSTMDecoder(nn.Module):
    def __init__(self, config):
        super(LSTMDecoder, self).__init__()
        # Encoder attention module
        self.enc_attention = LSTMEncoderAttention(config)

        # Decoder attention module
        self.dec_attention = LSTMDecoderAttention(config)

        # A single LSTM cell is used
        self.lstm = nn.LSTMCell(config.summarizer_emb_dim, config.hidden_dim)

        # Layer normalization to avoid increasingly larger hidden states norms
        self.layer_norm = nn.LayerNorm(config.hidden_dim)

        # A single linear layer is used to convert all the context information to textual embeddings
        self.embedding_generator = nn.Linear(config.hidden_dim * 4, config.summarizer_emb_dim)

        # Configuration settings
        self.config = config

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x_t, s_t, enc_out, enc_mask, prev_e_enc, prev_h_dec, dec_mask):
        '''
        x_t: decoder input at current step as a token id
        s_t: hidden and cell states (tupple) of previous iteration
        enc_out: encoder output
        enc_mask: mask tensor with boolean values for encoder output
        prev_e_enc: contains the encoder alignment scores from previous decoder time steps
        prev_h_dec: contains list of previous decoder hidden states
        dec_mask: contains mask of previous decoder steps
        '''

        # Compute next hidden and cell states
        s_t = self.lstm(x_t, s_t)

        h_dec, c_dec = s_t
        h_dec = self.layer_norm(h_dec)
        # Encoder attention
        context_enc, at_weights, prev_e_enc = self.enc_attention(enc_out, h_dec, enc_mask, prev_e_enc)

        # Intra-decoder attention
        context_dec, prev_h_dec = self.dec_attention(h_dec, prev_h_dec, dec_mask)

        # Determine the embeddings and logits, given all the context information
        context_info = T.cat([h_dec, context_enc, context_dec], dim=1)     # (batch_size, 4 * hidden_dim)
        embeddings = self.embedding_generator(context_info)  # (batch_size, embed_size)
        
        return embeddings, s_t, prev_e_enc, prev_h_dec, context_info     


class EOSGenerator(nn.Module):
    def __init__(self, config):
        super(EOSGenerator, self).__init__()
        # A bilinear function is used to compute the alignment scores   
        self.w_attention = nn.Bilinear(config.summarizer_emb_dim, config.summarizer_emb_dim, 1, bias=False)

        # A single linear layer is used to convert all the context information to whether the end of sequence has been reached
        self.linear = nn.Linear(config.hidden_dim * 4 + config.summarizer_emb_dim, 1)

        # Configuration settings
        self.config = config

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, context_info, window, window_mask, ctr):
        ''' Perform eos probability generation
        context_info: the contextual information tensor of the time step whose eos probability is under inference
        window: window of embeddings centred in the target time step
        window_mask: mask for the window of embeddings
        ctr: the center embedding
        '''

        # Get traditional alignment scores 
        e_eos = self.w_attention(ctr.unsqueeze(1).expand_as(window), window).squeeze(2) # bs,window_size
        # Set masked scores to -inf and subtract maxima for numerical stability
        e_eos = e_eos.masked_fill(T.logical_not(window_mask), -float('inf')) 
        e_eos = e_eos - T.amax(e_eos, dim=1, keepdim=True).masked_fill(T.logical_not(window_mask), 0.)
        
        # Calculate attention weights
        at_weights = F.softmax(e_eos, dim=1)  
        at_weights = at_weights.masked_fill(T.logical_not(window_mask), 0.).unsqueeze(dim=1) #bs, 1, window_size

        # Compute context vector
        context_eos = T.bmm(at_weights, window).squeeze(dim=1)     #bs, n_hid

        # Determine the eos logits, given all the context information
        context_info = T.cat([context_info, context_eos], dim=1)     # (batch_size, 4 * hidden_dim + summarizer_emb_dim)
        logits = self.linear(context_info)

        return logits


class LSTMCrossModalAdapter(ParentModule):
    def __init__(self, 
        config_file: str = 'src/models/lstm-adapter.json',
        batch_size: int = 10,
        accumulate_grad_batches: int = 1,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.,
        beta_1: float = 0.9,
        beta_2: float = 0.99,
        emb_loss_type: str = 'mse',
        emb_loss_weight: float = 0.5,
        eos_loss_weight: float = 0.5,
        teacher_forcing: bool = True,
        peeling_back: bool = False,
        tf_ratio_offset: float = 1.,
        tf_ratio_slope: float = 0.,
        freeze_encoder_decoder: bool = False,
        freeze_eos_generator: bool = False,
        eos_attention_window_size: int = 2,
        masked_modelling: bool = False,
        p_mask: float = 0.,
        M_mask: int = 5,
        predictions_file: str = 'eos_predictions.txt'
    ):
        super().__init__()
        # Save hyperparameters
        self.save_hyperparameters(ignore=['predictions_file'])
        # Loss weights must sum to 1
        self.hparams.eos_loss_weight = 1 - self.hparams.emb_loss_weight

        # Load the model configuration
        self.config = PretrainedConfig.from_json_file(config_file)

        # Instantiate encoder, decoder and eos generator
        self.encoder = LSTMEncoder(self.config)
        self.decoder = LSTMDecoder(self.config)        
        self.eos_generator = EOSGenerator(self.config)
        # First embedding fed to the decoder
        self.emb1 = nn.Parameter(T.full((self.config.summarizer_emb_dim, ), 0.))
        # Mask embedding
        self.mask_embedding = nn.Parameter(T.full((self.config.wav2vec2_emb_dim, ), 0.))

        # Freeze parameters
        if self.hparams.freeze_encoder_decoder:
            self.encoder.freeze()
            self.decoder.freeze()
            self.emb1.requires_grad = False
        if self.hparams.freeze_eos_generator:
            self.eos_generator.freeze()
        if not self.hparams.masked_modelling:
            self.mask_embedding.requires_grad = False

        # Teacher forcing ratio initialization
        self.tf_ratio = self.hparams.tf_ratio_offset

        # Save predictions file name
        self.predictions_file = predictions_file

    def forward(
        self, 
        inputs_embeds=None, 
        attention_mask=None, 
        decoder_inputs_embeds=None, 
        decoder_attention_mask=None,
        labels_emb=None,
        labels_eos=None,
        labels_eos_weights=None
    ):
        '''
        inputs_embeds: input audio embeddings
        attention_mask: input audio embeddings attention mask
        decoder_inputs_embeds: decoder input text embeddings
        decoder_attention_mask: decoder input text embeddings attention mask
        labels_emb: the target embeddings
        labels_eos: the target eos
        labels_eos_weights: weights for balancing the eos loss
        '''
        # Encoder and decoder inputs must be already normalized

        # Forward input through the encoder
        enc_out, s_t, enc_mask = self.encoder(inputs_embeds, T.sum(attention_mask, dim=1, keepdim=False))

        # Set the initial embedding to the trainable one
        decoder_inputs_embeds[:, 0 , :] = self.emb1.unsqueeze(0).expand(decoder_inputs_embeds.size(dim=0), -1)

        # Auxiliary variables for attention mechanisms
        prev_h_dec = None              # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        prev_e_enc = None      # Used for encoder intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        context_info = None
        
        decoder_emb = []
        decoder_eos = []
        decoder_info = []
        # Loop through every decoder input id
        max_dec_length = decoder_inputs_embeds.size(dim=1)
        for t in range(max_dec_length):
            out, s_t, prev_e_enc, prev_h_dec, context_info = self.decoder(decoder_inputs_embeds[:, t, :], s_t, enc_out, enc_mask, prev_e_enc, prev_h_dec, decoder_attention_mask[:, :t])
            decoder_emb.append(out.unsqueeze(dim=1))
            decoder_info.append(context_info)

        decoder_emb = T.cat(decoder_emb, dim=1)

        # Generate eos probabilities after embedding generation        
        for t in range(max_dec_length):
            eos_logits = self.eos_generator(
                context_info=decoder_info[t],
                window=decoder_emb[:, max(0, t - self.hparams.eos_attention_window_size):min(max_dec_length, t + self.hparams.eos_attention_window_size), :],
                window_mask=decoder_attention_mask[:, max(0, t - self.hparams.eos_attention_window_size):min(max_dec_length, t + self.hparams.eos_attention_window_size)],
                ctr=decoder_emb[:, t, :]
            )
            decoder_eos.append(eos_logits)

        output = {}
        # Return dict
        output['embeddings'] = decoder_emb
        output['logits'] = T.cat(decoder_eos, dim=1)
        # and compute losses
        if labels_emb is not None:
            if self.hparams.emb_loss_type == 'mse':
                output['loss_emb'] = F.mse_loss(output['embeddings'][decoder_attention_mask], labels_emb[decoder_attention_mask])
            elif self.hparams.emb_loss_type == 'cs':
                output['loss_emb'] = F.cosine_embedding_loss(output['embeddings'][decoder_attention_mask], labels_emb[decoder_attention_mask], T.tensor([1.], device=self.device))
        if (labels_eos is not None) and (labels_eos_weights is not None):
            output['loss_eos'] = F.binary_cross_entropy_with_logits(output['logits'] , labels_eos, weight=labels_eos_weights, reduction='sum')

        return output

    def forward_masked_teacher_forcing(
        self, 
        inputs_embeds=None, 
        attention_mask=None, 
        decoder_inputs_embeds=None, 
        decoder_attention_mask=None,
        labels_emb=None,
        labels_eos=None,
        labels_eos_weights=None,
        teacher_forcing_mask=None):
        '''
        inputs_embeds: input audio embeddings
        attention_mask: input audio embeddings attention mask
        decoder_inputs_embeds: decoder input text embeddings
        decoder_attention_mask: decoder input text embeddings attention mask        
        labels_emb: the target embeddings
        labels_eos: the target eos
        labels_eos_weights: weights for balancing the eos loss
        teacher_forcing_mask: mask for teacher forcing
        '''

        # Batch size
        batch_size = inputs_embeds.size(dim=0)

        # Check is the teacher forcing mask is valid
        assert T.sum(teacher_forcing_mask[:, 0]) == batch_size, f'teacher forcing is always required in the first decoding step'

        # Forward input through the encoder
        enc_out, s_t, enc_mask = self.encoder(inputs_embeds, T.sum(attention_mask, dim=1, keepdim=False))

        # Set the initial embedding to the trainable one
        decoder_inputs_embeds[:, 0 , :] = self.emb1.unsqueeze(0).expand(batch_size, -1)

        # Auxiliary variables for attention mechanisms
        prev_h_dec = None              # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        prev_e_enc = None      # Used for encoder intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        context_info = None
        
        decoder_emb = []
        decoder_eos = []
        decoder_info = []
        
        max_dec_length = decoder_inputs_embeds.size(dim=1)
        # Loop through every decoder input id
        for t in range(max_dec_length):
            # At each decoding step, determine both model predictions with and without teacher forcing if elements of the batch differ on the teacher forcing mask
            # Otherwise, evaluate only one of them
            num_teacher_forcing = T.sum(teacher_forcing_mask[:, t])
            # When forwarding through the decoder with teacher forcing
            if num_teacher_forcing != 0:
                out_tf, s_t_tf, prev_e_enc_tf, prev_h_dec_tf, context_info_tf = self.decoder(decoder_inputs_embeds[:, t, :], s_t, enc_out, enc_mask, prev_e_enc, prev_h_dec, decoder_attention_mask[:, :t])
            # When forwarding through the decoder without teacher forcing
            if num_teacher_forcing != batch_size:
                out_ntf, s_t_ntf, prev_e_enc_ntf, prev_h_dec_ntf, context_info_ntf = self.decoder(decoder_emb[t-1].squeeze(dim=1).clone().detach(), s_t, enc_out, enc_mask, prev_e_enc, prev_h_dec, decoder_attention_mask[:, :t])

            # Case only teacher forcing was applied
            if num_teacher_forcing == batch_size:
                s_t = s_t_tf
                prev_e_enc = prev_e_enc_tf
                prev_h_dec = prev_h_dec_tf
                out_emb = out_tf
                context_info = context_info_tf 
            # Case teacher forcing was not applied
            elif num_teacher_forcing == 0:
                s_t = s_t_ntf
                prev_e_enc = prev_e_enc_ntf
                prev_h_dec = prev_h_dec_ntf
                out_emb = out_ntf
                context_info = context_info_ntf
            # Case both
            else:
                s_t_0 = s_t_ntf[0].clone()
                s_t_1 = s_t_ntf[1].clone()
                s_t_0[teacher_forcing_mask[:, t]] = s_t_tf[0][teacher_forcing_mask[:, t]]
                s_t_1[teacher_forcing_mask[:, t]] = s_t_tf[1][teacher_forcing_mask[:, t]]
                s_t = (s_t_0, s_t_1)
                prev_e_enc = prev_e_enc_ntf.clone()
                prev_e_enc[teacher_forcing_mask[:, t]] = prev_e_enc_tf[teacher_forcing_mask[:, t]]
                prev_h_dec = prev_h_dec_ntf.clone()
                prev_h_dec[teacher_forcing_mask[:, t]] = prev_h_dec_tf[teacher_forcing_mask[:, t]]
                out_emb = out_ntf.clone()
                context_info = context_info_ntf.clone()
                out_emb[teacher_forcing_mask[:, t]] = out_tf[teacher_forcing_mask[:, t]]
                context_info[teacher_forcing_mask[:, t]] = context_info_tf[teacher_forcing_mask[:, t]]

            # Concat results
            decoder_emb.append(out_emb.unsqueeze(dim=1))
            decoder_info.append(context_info)

        decoder_emb = T.cat(decoder_emb, dim=1)
        # Generate eos probabilities after embedding generation
        for t in range(max_dec_length):
            eos_logits = self.eos_generator(
                context_info=decoder_info[t],
                window=decoder_emb[:, max(0, t - self.hparams.eos_attention_window_size):min(max_dec_length, t + self.hparams.eos_attention_window_size), :],
                window_mask=decoder_attention_mask[:, max(0, t - self.hparams.eos_attention_window_size):min(max_dec_length, t + self.hparams.eos_attention_window_size)],
                ctr=decoder_emb[:, t, :]
            )
            decoder_eos.append(eos_logits)

        output = {}
        # Return dict
        output['embeddings'] = decoder_emb
        output['logits'] = T.cat(decoder_eos, dim=1)
        # and compute losses
        if labels_emb is not None:
            if self.hparams.emb_loss_type == 'mse':
                output['loss_emb'] = F.mse_loss(output['embeddings'][decoder_attention_mask], labels_emb[decoder_attention_mask])
            elif self.hparams.emb_loss_type == 'cs':
                output['loss_emb'] = F.cosine_embedding_loss(output['embeddings'][decoder_attention_mask], labels_emb[decoder_attention_mask], T.tensor([1.], device=self.device))
        if (labels_eos is not None) and (labels_eos_weights is not None):
            output['loss_eos'] = F.binary_cross_entropy_with_logits(output['logits'] , labels_eos, weight=labels_eos_weights, reduction='sum')

        return output

    def random_mask(self, batch, batch_len): 
        '''
        batch: sequences of embeddings
        batch_len: corresponding sequence lengths
        '''
        # the hyperparameters 'p_mask' and 'M_mask' correspond to the probability of starting a masked span of embeddings and the length of that span, respectively
        # different spans may overlap
        batch_ = []
        for seq, seq_len in zip(batch, batch_len):
            idx2mask = T.argwhere(T.rand((seq_len, )) < self.hparams.p_mask).to(next(self.parameters()).device)
            idx2mask = T.tensor(list(set([idx.item() + i for idx in idx2mask for i in range(self.hparams.M_mask)])), dtype=T.long).to(next(self.parameters()).device)
            idx2mask = idx2mask[T.where(idx2mask < seq_len)[0]]
            seq = seq.index_copy(0, idx2mask, self.mask_embedding.unsqueeze(0).repeat(idx2mask.size(dim=0), 1))
            batch_.append(seq.unsqueeze(0))
        batch_ = T.cat(batch_, 0)
        return batch_

    def peeling_back_mask(self, attention_mask=None, tf_ratio=None):
        if tf_ratio is None:
            tf_ratio = self.tf_ratio
        # Determine the lengths of the sequences that shall be forwarded through the model with teacher forcing
        seq_lens = T.sum(attention_mask, dim=1, keepdim=True)
        seq_lens = seq_lens.float() * tf_ratio
        seq_lens = seq_lens.long()

        # Generate the teacher forcing mask based on the computed lengths
        teacher_forcing_mask = T.arange(0, attention_mask.size(dim=1), 1, dtype=T.long, device=next(self.parameters()).device).expand_as(attention_mask)
        teacher_forcing_mask = teacher_forcing_mask < seq_lens.expand_as(attention_mask)
        teacher_forcing_mask[:, 0] = True

        return teacher_forcing_mask

    def training_step(self, batch, batch_idx): 
        # Normalize the audio embeddings
        batch['audio_embeddings'] = (batch['audio_embeddings'] - batch['norm'][0].unsqueeze(dim=1)) / batch['norm'][1].unsqueeze(dim=1)
        # Randomly mask embeddings, if masked language modelling
        if self.hparams.masked_modelling:
            batch['audio_embeddings'] = self.random_mask(batch['audio_embeddings'], T.sum(batch['audio_embeddings_mask'], dim=1, keepdim=False))

        # Forward batch through the adapter, possibly using teacher forcing mask
        if self.hparams.teacher_forcing:
            out = self(
                inputs_embeds=batch['audio_embeddings'], 
                attention_mask=batch['audio_embeddings_mask'], 
                decoder_inputs_embeds=(batch['text_embeddings_inputs'] - batch['norm'][2].unsqueeze(dim=1)) / batch['norm'][3].unsqueeze(dim=1),
                decoder_attention_mask=batch['text_embeddings_mask'],
                labels_emb=(batch['text_embeddings'] - batch['norm'][2].unsqueeze(dim=1)) / batch['norm'][3].unsqueeze(dim=1),
                labels_eos=batch['text_embeddings_eos'],
                labels_eos_weights=batch['text_embeddings_eos_weights'])       
        else:
            # Get teacher forcing mask when peeling back
            if self.hparams.peeling_back:
                # Update teacher forcing ratio at each optimization step; log teacher forcing ratio
                if ( (batch_idx != 0) and (batch_idx % self.hparams.accumulate_grad_batches == 0) ):
                    self.tf_ratio = max(0., self.tf_ratio - self.hparams.tf_ratio_slope)
                teacher_forcing_mask = self.peeling_back_mask(batch['text_embeddings_mask'])
            # Otherwise, forward as in inference / same results if using generate
            else:
                # Forward with teacher forcing mask
                teacher_forcing_mask = self.peeling_back_mask(batch['text_embeddings_mask'], tf_ratio=self.tf_ratio)
            out = self.forward_masked_teacher_forcing(
                inputs_embeds=batch['audio_embeddings'], 
                attention_mask=batch['audio_embeddings_mask'], 
                decoder_inputs_embeds=(batch['text_embeddings_inputs'] - batch['norm'][2].unsqueeze(dim=1)) / batch['norm'][3].unsqueeze(dim=1),
                decoder_attention_mask=batch['text_embeddings_mask'],
                teacher_forcing_mask=teacher_forcing_mask,
                labels_emb=(batch['text_embeddings'] - batch['norm'][2].unsqueeze(dim=1)) / batch['norm'][3].unsqueeze(dim=1),
                labels_eos=batch['text_embeddings_eos'],
                labels_eos_weights=batch['text_embeddings_eos_weights'])       
         
        # Get losses and compute a weighted sum
        loss_emb = out['loss_emb']
        loss_eos = out['loss_eos']
        loss = self.hparams.emb_loss_weight * loss_emb + self.hparams.eos_loss_weight * loss_eos

        # Log losses and teacher forcing ratio, if necessary
        self.log('train/loss_emb', loss_emb)
        self.log('train/loss_eos', loss_eos)
        self.log('train/loss', loss)
        if not self.hparams.teacher_forcing:
            self.log('train/tf_ratio', self.tf_ratio)

        return loss

    def validation_step(self, batch, batch_idx):
        # Normalize the audio embeddings
        batch['audio_embeddings'] = (batch['audio_embeddings'] - batch['norm'][0].unsqueeze(dim=1)) / batch['norm'][1].unsqueeze(dim=1)

        # Forward batch through the adapter, possibly using teacher forcing mask
        if self.hparams.teacher_forcing:
            out = self(
                inputs_embeds=batch['audio_embeddings'], 
                attention_mask=batch['audio_embeddings_mask'], 
                decoder_inputs_embeds=(batch['text_embeddings_inputs'] - batch['norm'][2].unsqueeze(dim=1)) / batch['norm'][3].unsqueeze(dim=1),
                decoder_attention_mask=batch['text_embeddings_mask'],
                labels_emb=(batch['text_embeddings'] - batch['norm'][2].unsqueeze(dim=1)) / batch['norm'][3].unsqueeze(dim=1),
                labels_eos=batch['text_embeddings_eos'],
                labels_eos_weights=batch['text_embeddings_eos_weights'])    
        else:
            # Forward with teacher forcing mask
            teacher_forcing_mask = T.full_like(batch['text_embeddings_mask'], False)
            teacher_forcing_mask[:, 0] = True
            out = self.forward_masked_teacher_forcing(
                inputs_embeds=batch['audio_embeddings'], 
                attention_mask=batch['audio_embeddings_mask'], 
                decoder_inputs_embeds=(batch['text_embeddings_inputs'] - batch['norm'][2].unsqueeze(dim=1)) / batch['norm'][3].unsqueeze(dim=1),
                decoder_attention_mask=batch['text_embeddings_mask'],
                teacher_forcing_mask=teacher_forcing_mask,
                labels_emb=(batch['text_embeddings'] - batch['norm'][2].unsqueeze(dim=1)) / batch['norm'][3].unsqueeze(dim=1),
                labels_eos=batch['text_embeddings_eos'],
                labels_eos_weights=batch['text_embeddings_eos_weights'])         
         
        # Get losses and compute a weighted sum
        loss_emb = out['loss_emb']
        loss_eos = out['loss_eos']
        loss = self.hparams.emb_loss_weight * loss_emb + self.hparams.eos_loss_weight * loss_eos

        batch_size = batch['audio_embeddings'].size(dim=0)
        # Log losses and teacher forcing ratio, if necessary
        self.log('val/loss_emb', loss_emb, batch_size=batch_size)
        self.log('val/loss_eos', loss_eos, batch_size=batch_size)
        self.log('val/loss', loss, batch_size=batch_size)
        if not self.hparams.teacher_forcing:
            self.log('val/tf_ratio', self.tf_ratio, batch_size=batch_size)

        return loss

    def generate(
        self, 
        inputs_embeds = None, 
        attention_mask = None, 
        eos_inference = None,
        ground_truth_lengths = None,
        eos_threshold = None,
        padding = None, 
        max_length = None
    ):
        '''
        inputs_embeds: input audio embeddings
        attention_mask: input audio embeddings attention mask
        eos_inference: the type of eos inference to be applied to the predicted eos probabilities; choose from 'ground_truth_lengths' or 'naive'
        ground_truth_lengths: ground truth lengths of the text embeddings sequence
        eos_threshold: the eos probability threshold for the naive approach
        padding: the type of padding to apply to the output; choose from ('do_not_pad', True), ('longest', False) and 'max_length'
        max_length: maximum text embedding sequence length
        '''

        # Check if eos_inference is a valid option
        assert (eos_inference is not None) and (eos_inference in ['ground_truth_lengths', 'naive', 'bayesian']), f'the provided eos inference is not valid, got {eos_inference}'
        # Check if padding and max length is a valid option
        if type(padding) is bool:
            if padding:
                padding = 'longest'
            else:
                padding = 'do_not_pad'
        assert (padding is not None) and (padding in ['longest', 'max_length', 'do_not_pad']), f'the provided padding is not valid, got {padding}'
        assert (max_length is not None) and (max_length > 0), f'the provided max length is not valid, got {max_length}'
        
        # Check if the ground truth lengths are provided if required
        assert ( (eos_inference in ['naive', 'bayesian']) or (ground_truth_lengths is not None) ), f'when converting embeddings using ground truth lengths, you must provide the latter'

        # Check if the eos probability threshold is provided if required
        assert ( (eos_inference == 'ground_truth_lengths') or (eos_threshold is not None) ), f'when inferring eos using the naive approach, you must provide the threshold'
        assert (eos_inference == 'ground_truth_lengths') or ( (eos_threshold >= 0.) and (eos_threshold <= 1.) ), f'the eos probability threshold must be between 0 and 1, got {threshold}' 

        # Get batch size
        batch_size = inputs_embeds.size(dim=0)

        # Forward input through the encoder
        enc_out, s_t, enc_mask = self.encoder(inputs_embeds, T.sum(attention_mask, dim=1, keepdim=False))

        # Auxiliary variables for attention mechanisms
        prev_h_dec = None              # Used for intra-decoder attention (section 2.2 in https://arxiv.org/pdf/1705.04304.pdf)
        prev_e_enc = None      # Used for encoder intra-temporal attention (section 2.1 in https://arxiv.org/pdf/1705.04304.pdf)
        context_info = None
        
        # Set first element of decoder inputs to the initial embedding to the trainable one, set decoder attention mask and initialize decoder eos probabilities
        decoder_inputs_embeds = self.emb1.unsqueeze(0).expand(batch_size, -1).unsqueeze(dim=1)
        decoder_attention_mask = T.full((batch_size, max_length), True, dtype=T.bool, device=self.device)
        decoder_eos_probs = T.full((batch_size, max_length), 0, dtype=T.float, device=self.device)
        decoder_info = []

        # Loop through every decoder input
        for t in range(max_length):  
            # Decode and transform logits to eos probabilities
            out, s_t, prev_e_enc, prev_h_dec, context_info = self.decoder(decoder_inputs_embeds[:, t, :], s_t, enc_out, enc_mask, prev_e_enc, prev_h_dec, decoder_attention_mask[:, :t])     
            decoder_inputs_embeds = T.cat((decoder_inputs_embeds, out.unsqueeze(dim=1)), dim=1)
            decoder_info.append(context_info)

        # Generate eos probabilities after embedding generation if required
        for t in range(max_length):
            eos_logits = self.eos_generator(
                context_info = decoder_info[t],
                window = decoder_inputs_embeds[:, max(1, t + 1 - self.hparams.eos_attention_window_size):min(max_length + 1, t + 1 + self.hparams.eos_attention_window_size), :],
                window_mask = decoder_attention_mask[:, max(1, t + 1 - self.hparams.eos_attention_window_size):min(max_length + 1, t + 1 + self.hparams.eos_attention_window_size)],
                ctr = decoder_inputs_embeds[:, t + 1, :]
            )
            decoder_eos_probs[:, t] = T.sigmoid(eos_logits.squeeze(dim=1))

        # Infer eos
        if eos_inference == 'ground_truth_lengths':
            sequences = decoder_inputs_embeds[:, 1:, :]
            mask = T.arange(max_length, dtype=T.long, device=self.device).expand(batch_size, -1) < ground_truth_lengths.unsqueeze(dim=1)
            sequences[T.logical_not(mask)] = 0.
        elif eos_inference == 'naive':
            sequences = decoder_inputs_embeds[:, 1:, :]
            mask = T.arange(max_length, dtype=T.long, device=self.device).expand(batch_size, -1) < T.cat([min(T.argwhere(x_ >= eos_threshold)) if len(T.argwhere(x_ >= eos_threshold)) != 0 else T.tensor([max_length - 1], dtype=T.long, device=self.device) for x_ in decoder_eos_probs], dim=0).unsqueeze(dim=1)
            sequences[T.logical_not(mask)] = 0.

        # Set output
        if padding == 'max_length':
            return_dict = {'embeddings' : sequences, 'attention_mask' : mask, 'decoded_embeddings' : decoder_inputs_embeds[:, 1:, :], 'decoded_eos_probs' : decoder_eos_probs}
        elif padding == 'longest':
            longest_length = max(T.sum(mask, dim=1))
            return_dict = {'embeddings' : sequences[:, :longest_length, :], 'attention_mask' : mask[:, :longest_length], 'decoded_embeddings' : decoder_inputs_embeds[:, 1:, :], 'decoded_eos_probs' : decoder_eos_probs}
        elif padding == 'do_not_pad':
            return_dict = {'embeddings' : [seq[:T.sum(seq_mask)] for seq, seq_mask in zip(sequences, mask)], 'decoded_embeddings' : decoder_inputs_embeds[:, 1:, :], 'decoded_eos_probs' : decoder_eos_probs}

        return return_dict
    
    def configure_optimizers(self):
        return Adam(
            self.parameters(),
            lr = self.hparams.learning_rate,
            betas = (self.hparams.beta_1, self.hparams.beta_2),
            weight_decay = self.hparams.weight_decay
        )