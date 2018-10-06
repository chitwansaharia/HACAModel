import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils import *
import numpy as np

class customRNN(nn.Module):
    """
        Custom RNN : Used to initialize one layer LSTMs with different sizes
                     initHidden() initializes the hidden_state, and the cell_state of the LSTM
    """
    def __init__(self, input_size, hidden_size, isbidirectional, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.isbidirectional = isbidirectional
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=isbidirectional)

    def forward(self, input, hidden_state, cell_state):
        output, (hidden_state, cell_state) = self.lstm(input, (hidden_state,cell_state))
        return output, hidden_state, cell_state

    def initHidden(self):
        if self.isbidirectional:
            hidden_state = torch.zeros([2, self.hidden_size], device=self.device)
            cell_state = torch.zeros([2, self.hidden_size], device=self.device)
        else:
            hidden_state = torch.zeros([1, self.hidden_size], device=self.device)
            cell_state = torch.zeros([1, self.hidden_size], device=self.device)
        return hidden_state, cell_state

class encoder(nn.Module):
    def __init__(self, input_size, hidden_size, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.rnn = customRNN(input_size, hidden_size["low"], True, device)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)


    def forward(self, input):
        batch_size = input.shape[0]
        # Initialized memory of  encoder
        memory, cell_state = self.rnn.initHidden()
        # Created the memory and cell_state for the entire batch
        memory, cell_state = torch.stack([memory]*batch_size, dim=1), torch.stack([cell_state]*batch_size, dim=1)
        # Forward pass on encoder
        outputs, memory, _ = self.rnn(input, memory, cell_state)

        return outputs

class decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, input_size, hidden_size, encoder_output_size, device, embedding_size=None, vocab_size=None, use_audio_modality=False):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder_rnn = customRNN(input_size, hidden_size, False, device)
        self.input_size = input_size
        self.device = device
        self.use_audio_modality = use_audio_modality

        self.context_size = self.input_size-embedding_size
        # Attention matrices used to calculate visual context at time t
        self.visual_attention_weight_1 = nn.Linear(self.hidden_size,self.context_size, bias=False)
        self.visual_attention_weight_2 = nn.Linear(encoder_output_size["visual"], self.context_size, bias=False)
        self.visual_attention_weight_3 = nn.Linear(self.context_size,1, bias=False)
        self.visual_enc_to_dec_mat = nn.Linear(encoder_output_size["visual"], self.context_size, bias=False)

        if use_audio_modality:
            self.audio_attention_weight_1 = nn.Linear(self.hidden_size,self.context_size, bias=False)
            self.audio_attention_weight_2 = nn.Linear(encoder_output_size["audio"], self.context_size, bias=False)
            self.audio_attention_weight_3 = nn.Linear(self.context_size,1, bias=False)
            self.audio_enc_to_dec_mat = nn.Linear(encoder_output_size["audio"], self.context_size, bias=False)

            self.context_attention_weight_1 = nn.Linear(self.hidden_size,self.context_size, bias=False)
            self.context_attention_weight_2 = nn.Linear(self.context_size, self.context_size, bias=False)
            self.context_attention_weight_3 = nn.Linear(self.context_size,1, bias=False)

        self.output_layer = nn.Sequential(
                            nn.Linear(self.hidden_size, vocab_size)
                            )

    def get_context(self, vec1, vec2_list, weights):
        # batch_size x seq_len x alpha  -> batch_size x alpha x seq_len
        vec2_list = torch.transpose(vec2_list, 1, 2)
        attns = []
        # batch_size x 1 x vec1_size -> batch_size x common_size
        v1 = weights[0](vec1[:,0,:])
        tanh = nn.Tanh()
        softmax = nn.Softmax(dim=2)
        for time_step in range(vec2_list.shape[2]):
            # batch_size x alpha -> batch_size x common_size
            v2 = weights[1](vec2_list[:,:,time_step])
            # batch_size x common_size + batch_size x common_size -> batch_size x 1
            attns.append(weights[2](tanh(v1+v2)))
        # batch_size x 1 x seq_len
        attns = softmax(torch.stack(attns, dim=2))
        # batch_size x alpha x seq_len  -> batch_size x seq_len x alpha
        vec2_list = torch.transpose(vec2_list, 1, 2)
        # batch_size x alpha
        return torch.squeeze(torch.bmm(attns, vec2_list), dim=1)


    def forward(self, input, encoder_outputs, hidden_states, cell_state):
        batch_size = input.shape[0]
        hidden_state = hidden_states[-1]

        # getting contexts for the time_step
        attention_weights = [self.visual_attention_weight_1, self.visual_attention_weight_2, self.visual_attention_weight_3]
        visual_context = self.get_context(torch.transpose(hidden_state,0,1), encoder_outputs["visual"], attention_weights)
        visual_context = self.visual_enc_to_dec_mat(torch.unsqueeze(visual_context, dim=1))

        if self.use_audio_modality:
            attention_weights = [self.audio_attention_weight_1, self.audio_attention_weight_2, self.audio_attention_weight_3]
            audio_context = self.get_context(torch.transpose(hidden_state,0,1), encoder_outputs["audio"], attention_weights)
            audio_context = self.audio_enc_to_dec_mat(torch.unsqueeze(audio_context, dim=1))
            all_context = torch.cat([visual_context, audio_context], dim=1)

            attention_weights = [self.context_attention_weight_1, self.context_attention_weight_2, self.context_attention_weight_3]
            final_context = self.get_context(torch.transpose(hidden_state,0,1), all_context, attention_weights)
            final_context = torch.unsqueeze(final_context, dim=1)
        else:
            final_context = visual_context

        tanh = nn.Tanh()
        output, hidden_state, cell_state = self.decoder_rnn(torch.cat([input,tanh(final_context)],dim=2), hidden_state, cell_state)

        output = torch.unsqueeze(self.output_layer(torch.squeeze(output, dim=1)), dim=1)
        return output, hidden_state, cell_state


class EncDecModel(nn.Module):
    """
    Simple Encoder Decoder Model
    """
    def __init__(self, config, device, batch_size, max_caption_length, use_audio_modality):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.max_caption_length = max_caption_length
        self.use_audio_modality = use_audio_modality

        # Encoders (Video)

        self.visual_encoder = encoder(config["visual_input_size"], config["visual_hidden_size"], device)
        if self.use_audio_modality:
            self.audio_encoder = encoder(config["audio_input_size"], config["audio_hidden_size"], device)
        encoder_output_size = {"audio" : 2*config["audio_hidden_size"]["low"], "visual" : 2*config["visual_hidden_size"]["low"]}
        #  Decoder
        self.local_decoder = decoder(config["local_input_size"], config["local_hidden_size"], encoder_output_size,
                                device, config["embedding_size"], config["vocab_size"], use_audio_modality)


        self.input_embeddings = nn.Embedding(config["vocab_size"], config["embedding_size"])

        self.device = device

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

    def forward(self, input, ss_epsilon):
        # obtaining outputs from low level and high level encoders of both modalities

        visual_outputs = self.visual_encoder(input["visual_features"])
        encoder_outputs = {"visual" : visual_outputs, "audio" : None}

        if self.use_audio_modality:
            audio_outputs = self.audio_encoder(input["audio_features"])
            encoder_outputs["audio"] = audio_outputs

        # Hidden states and cell states for local decoder
        memory_local, cell_state_local = self.local_decoder.decoder_rnn.initHidden()
        memory_local_list, cell_state_local = [torch.stack([memory_local]*self.batch_size, dim=1)], torch.stack([cell_state_local]*self.batch_size, dim=1)

        output_caption = []
        input_ts = input["caption_x"][:, 0]
        for time_step in range(self.max_caption_length):
            input_ts = torch.unsqueeze(self.input_embeddings(input_ts), dim=1)

            output_local, memory_local, cell_state_local = self.local_decoder(input_ts, encoder_outputs, memory_local_list, cell_state_local)
            memory_local_list.append(memory_local)

            output_caption.append(output_local)
            # Incase of scheduled sampling, deciding the input at next time step.
            if time_step != self.max_caption_length-1:
                if ss_epsilon == None:
                    input_ts = input["caption_x"][:, time_step+1]
                else:
                    val = np.random.binomial(1, ss_epsilon)
                    if val == 1:
                        input_ts = input["caption_x"][:, time_step+1]
                    else:
                        max_pred = torch.squeeze(torch.max(output_local.data, dim=2, keepdim=False)[1], dim=1)
                        input_ts = max_pred

        return output_caption