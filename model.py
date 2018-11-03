import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils import *
import numpy as np
import pickle
from custom_rnn import *

# Vocabulary (Used while testing)
vocab_path = "./meta_data/vocab.pkl"
vocab = pickle.load(open(vocab_path,'rb'))
end_token = vocab["<END>"]

class encoder(nn.Module):
    """
    Hierarchichal Encoder : Consists of Two LSTMs, with different granularities.
    """
    def __init__(self, input_size, hidden_size, chunk_size, device):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.device = device
        self.chunk_size = chunk_size
        # low level RNN
        self.low_encoder_rnn = customRNN(input_size, hidden_size["low"], True, device)
        # high level RNN
        self.high_encoder_rnn = customRNN(2*hidden_size["low"], hidden_size["high"], False, device)
        # Attention Matrix to be multiplied to the hidden states of low level RNN (mat_high)
        self.low_enc_attn_wts = nn.Linear(2*hidden_size["low"], hidden_size["low"], bias=False)
        # Attention Matrix to be multiplied to the hidden states of high level RNN (mat_low)
        self.high_enc_attn_wts = nn.Linear(hidden_size["high"], hidden_size["low"], bias=False)
        self.attn_vector = nn.Linear(hidden_size["low"], 1, bias=False)
        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)


    def compute_attn(self, vec1, vec2_list):
        # batch_size x seq_len x 2*h_low  -> batch_size x 2*h_low x seq_len
        vec2_list = torch.transpose(vec2_list, 1, 2)
        attns = []
        # 1 x batch_size x h_high -> batch_size x h
        v1 = self.high_enc_attn_wts(vec1[0,:,:])
        tanh = nn.Tanh()
        softmax = nn.Softmax(dim=2)
        for time_step in range(vec2_list.shape[2]):
            # batch_size x 2*h_low -> batch_size x h
            v2 = self.low_enc_attn_wts(vec2_list[:,:,time_step])
            # batch_size x h + batch_size x h -> batch_size x 1
            attns.append(self.attn_vector(tanh(v1+v2)))
        # batch_size x 1 x seq_len
        attns = softmax(torch.stack(attns, dim=2))

        # batch_size x 2*h_low x seq_len  -> batch_size x seq_len x 2*h_low
        vec2_list = torch.transpose(vec2_list, 1, 2)
        # batch_size x 1 x 2*h_low
        return torch.bmm(attns, vec2_list)


    def forward(self, input):
        batch_size = input.shape[0]
        # Initialized memory of low level encoder
        memory_low, cell_state_low = self.low_encoder_rnn.initHidden()
        # Created the memory and cell_state for the entire batch
        memory_low, cell_state_low = torch.stack([memory_low]*batch_size, dim=1), torch.stack([cell_state_low]*batch_size, dim=1)
        # Forward pass on low level encoder
        outputs_low, memory_low, _ = self.low_encoder_rnn(input, memory_low, cell_state_low)

        # applying dropout to the outputs from low level encoder
        outputs_low = self.dropout(outputs_low)

        # splitting output states for high level encoder
        contexts = list(chunks(outputs_low,self.chunk_size))

        # high level encoder
        memory_high, cell_state_high = self.high_encoder_rnn.initHidden()
        memory_high, cell_state_high = torch.stack([memory_high]*batch_size, dim=1), torch.stack([cell_state_high]*batch_size, dim=1)

        outputs_high = []

        # running for each context
        for i in range(len(contexts)):
            # computes attention weighted input for high level RNN
            input = self.compute_attn(memory_high, contexts[i])
            # input -> batch_size x 1 x input_dim (input for one time_step)
            output, memory_high, _ = self.high_encoder_rnn(input, memory_high, cell_state_high)
            outputs_high.append(output.squeeze(1))
        outputs_high = torch.stack(outputs_high, dim=1)

        outputs_high = self.dropout(outputs_high)

        return outputs_low, outputs_high

class decoder(nn.Module):
    """
    Decoder
    """
    def __init__(self, input_size, hidden_size, isglobal, encoder_output_size, device, embedding_size=None, vocab_size=None, global_output_size=None):
        super().__init__()
        self.hidden_size = hidden_size
        self.decoder_rnn = customRNN(input_size, hidden_size, False, device)
        self.isglobal = isglobal
        self.input_size = input_size
        self.device = device
        if isglobal:
            self.context_size = self.input_size-embedding_size
        else:
            self.output_layer = nn.Sequential(
                                nn.Linear(self.hidden_size, vocab_size)
                                )
            self.context_size = self.input_size-global_output_size-embedding_size
        # Attention matrices used to calculate visual context at time t
        self.visual_attention_weight_1 = nn.Linear(self.hidden_size,self.context_size, bias=False)
        self.visual_attention_weight_2 = nn.Linear(encoder_output_size["visual"], self.context_size, bias=False)
        self.visual_attention_weight_3 = nn.Linear(self.context_size,1, bias=False)
        self.visual_enc_to_dec_mat = nn.Linear(encoder_output_size["visual"], self.context_size, bias=True)

        # Attention matrices used to calculate audio context at time t
        self.audio_attention_weight_1 = nn.Linear(self.hidden_size,self.context_size, bias=False)
        self.audio_attention_weight_2 = nn.Linear(encoder_output_size["audio"], self.context_size, bias=False)
        self.audio_attention_weight_3 = nn.Linear(self.context_size,1, bias=False)
        self.audio_enc_to_dec_mat = nn.Linear(encoder_output_size["audio"], self.context_size, bias=True)


        # Attention matrices used to calculate hidden vectors' context at time t
        self.hidden_attention_weight_1 = nn.Linear(self.hidden_size,self.context_size, bias=False)
        self.hidden_attention_weight_2 = nn.Linear(self.hidden_size,self.context_size, bias=False)
        self.hidden_attention_weight_3 = nn.Linear(self.context_size,1, bias=False)
        self.hid_to_dec_mat = nn.Linear(self.hidden_size, self.context_size, bias=True)

        # Attention matrices used to compute attention over scaled vectors
        self.context_attention_weight_1 = nn.Linear(self.hidden_size,self.context_size, bias=False)
        self.context_attention_weight_2 = nn.Linear(self.context_size, self.context_size, bias=False)
        self.context_attention_weight_3 = nn.Linear(self.context_size,1, bias=False)


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


    def forward(self, input, encoder_outputs, hidden_states, cell_state, context):
        batch_size = input.shape[0]
        hidden_state = hidden_states[-1]
        # cf = torch.zeros([batch_size, 1, self.context_size], device=self.device)

        # getting contexts for the time_step
        attention_weights = [self.visual_attention_weight_1, self.visual_attention_weight_2, self.visual_attention_weight_3]
        visual_context = self.get_context(torch.transpose(hidden_state,0,1), encoder_outputs["visual"], attention_weights)
        visual_context = torch.unsqueeze(self.visual_enc_to_dec_mat(visual_context),dim=1)

        attention_weights = [self.audio_attention_weight_1, self.audio_attention_weight_2, self.audio_attention_weight_3]
        audio_context = self.get_context(torch.transpose(hidden_state,0,1), encoder_outputs["audio"], attention_weights)
        audio_context = torch.unsqueeze(self.audio_enc_to_dec_mat(audio_context),dim=1)

        concatenated_hs = torch.transpose(torch.cat(hidden_states, dim=0),0,1)
        attention_weights = [self.hidden_attention_weight_1, self.hidden_attention_weight_2, self.hidden_attention_weight_3]
        hidden_context = self.get_context(torch.transpose(hidden_state,0,1), concatenated_hs, attention_weights)
        hidden_context = self.hid_to_dec_mat(torch.unsqueeze(hidden_context, dim=1))

        # stacking these contexts together
        tanh = nn.Tanh()
        all_context = torch.cat([visual_context, hidden_context, audio_context], dim=1)

        attention_weights = [self.context_attention_weight_1, self.context_attention_weight_2, self.context_attention_weight_3]
        final_context = self.get_context(torch.transpose(hidden_state,0,1), all_context, attention_weights)
        final_context = tanh(torch.unsqueeze(final_context, dim=1))

        input = torch.cat([final_context, input], dim=2)
        output, hidden_state, cell_state = self.decoder_rnn(input, hidden_state, cell_state)

        if not self.isglobal:
            output = torch.unsqueeze(self.output_layer(torch.squeeze(output, dim=1)), dim=1)
        return output, hidden_state, cell_state, final_context


class HACAModel(nn.Module):
    """
    Main Model : Consists of encoders of audio and visual modality, and a local as well as global decoder.
    """
    def __init__(self, config, device, batch_size, max_caption_length):
        super().__init__()
        self.config = config
        self.batch_size = batch_size
        self.max_caption_length = max_caption_length

        # Encoders (Audio & Video)

        self.audio_encoder = encoder(config["audio_input_size"], config["audio_hidden_size"], config["chunk_size"]["audio"], device)
        self.visual_encoder = encoder(config["visual_input_size"], config["visual_hidden_size"], config["chunk_size"]["visual"], device)

        # Decoders (Global & Local)
        encoder_output_size = {"audio" : 2*config["audio_hidden_size"]["low"], "visual" : 2*config["visual_hidden_size"]["low"]}
        # Local Decoder
        self.local_decoder = decoder(config["local_input_size"], config["local_hidden_size"], False, encoder_output_size,
                                device, config["embedding_size"], config["vocab_size"], config["global_hidden_size"])
        encoder_output_size = {"audio" : config["audio_hidden_size"]["high"], "visual" : config["visual_hidden_size"]["high"]}

        # Global Decoder
        self.global_decoder = decoder(config["global_input_size"], config["global_hidden_size"], True, encoder_output_size,
                                device, config["embedding_size"], config["vocab_size"], None)
        self.input_embeddings = nn.Embedding(config["vocab_size"], config["embedding_size"])

        self.device = device

        self.dropout = torch.nn.Dropout(p=0.5, inplace=False)

    def forward(self, input, ss_epsilon, beam_search=False, beam_width=-1):
        # obtaining outputs from low level and high level encoders of both modalities
        audio_outputs_low, audio_outputs_high = self.audio_encoder(input["audio_features"])
        visual_outputs_low, visual_outputs_high = self.visual_encoder(input["visual_features"])

        # decoder steps
        local_encoder_outputs = {"audio" : audio_outputs_low, "visual" : visual_outputs_low}
        global_encoder_outputs = {"audio" : audio_outputs_high, "visual" : visual_outputs_high}

        # Hidden states and cell states for local decoder
        memory_local, cell_state_local = self.local_decoder.decoder_rnn.initHidden()
        memory_local_list, cell_state_local = [torch.stack([memory_local]*self.batch_size, dim=1)], torch.stack([cell_state_local]*self.batch_size, dim=1)

        # Hidden states and cell states for global decoder
        memory_global, cell_state_global = self.global_decoder.decoder_rnn.initHidden()
        memory_global_list, cell_state_global = [torch.stack([memory_global]*self.batch_size, dim=1)], torch.stack([cell_state_global]*self.batch_size, dim=1)

        local_context  = torch.zeros([self.batch_size, 1, self.local_decoder.context_size], device=self.device)
        global_context = torch.zeros([self.batch_size, 1, self.global_decoder.context_size], device=self.device)

        output_caption = []
        input_ts = input["caption_x"][:, 0]

        if beam_search:
            lengths = torch.ones([self.batch_size*beam_width, 1], device=self.device)
            mask = torch.ones([self.batch_size*beam_width, 1], device=self.device)
            # Reinitializing the memory and cell states
            ########################################################################################################################
            # Hidden states and cell states for local decoder
            memory_local, cell_state_local = self.local_decoder.decoder_rnn.initHidden()
            memory_local_list, cell_state_local = [torch.stack([memory_local]*self.batch_size*beam_width, dim=1)], torch.stack([cell_state_local]*self.batch_size*beam_width, dim=1)

            # Hidden states and cell states for global decoder
            memory_global, cell_state_global = self.global_decoder.decoder_rnn.initHidden()
            memory_global_list, cell_state_global = [torch.stack([memory_global]*self.batch_size*beam_width, dim=1)], torch.stack([cell_state_global]*self.batch_size*beam_width, dim=1)

            local_context  = torch.zeros([self.batch_size*beam_width, 1, self.local_decoder.context_size], device=self.device)
            global_context = torch.zeros([self.batch_size*beam_width, 1, self.global_decoder.context_size], device=self.device)
            ########################################################################################################################

            softmax = nn.Softmax(dim=1)
            sentences = torch.zeros([self.batch_size*beam_width, 1], device=self.device)
            sentences[:,0] = input_ts[0]
            probability_list = torch.zeros([self.batch_size*beam_width, 1], device=self.device)

            for key in local_encoder_outputs:
                temp = local_encoder_outputs[key]
                temp2 = torch.zeros([temp.shape[0]*beam_width, temp.shape[1], temp.shape[2]], device = self.device)
                for index in range(self.batch_size):
                    temp2[index*beam_width:(index+1)*beam_width, :, :] = temp[index, :, :]
                local_encoder_outputs[key] = temp2

            for key in global_encoder_outputs:
                temp = global_encoder_outputs[key]
                temp2 = torch.zeros([temp.shape[0]*beam_width, temp.shape[1], temp.shape[2]], device = self.device)
                for index in range(self.batch_size):
                    temp2[index*beam_width:(index+1)*beam_width, :, :] = temp[index, :, :]
                global_encoder_outputs[key] = temp2

            for time_step in range(self.max_caption_length):
                input_ts = torch.unsqueeze(self.input_embeddings(sentences[:,-1].long()), dim=1)
                # One time_step for global decoder
                output_global, memory_global, cell_state_global, global_context = self.global_decoder(input_ts, global_encoder_outputs, memory_global_list, cell_state_global, global_context)
                memory_global_list.append(memory_global)

                # One time_step for local decoder
                output_local, memory_local, cell_state_local, local_context = self.local_decoder(torch.cat([input_ts, output_global], dim=2), local_encoder_outputs, memory_local_list, cell_state_local, local_context)
                memory_local_list.append(memory_local)

                output_local = softmax(torch.squeeze(output_local, dim=1))
                output_local = torch.log(output_local)
                new_sentences = torch.zeros([self.batch_size*beam_width, sentences.shape[1]+1], device = self.device)

                current_index = 0
                carry_forward_indexes = [0]*self.batch_size*beam_width
                temp_mask = torch.ones([self.batch_size*beam_width, 1], device=self.device)
                for index in range(self.batch_size):
                    output_flattened = output_local[index*beam_width : (index+1)*beam_width,:]
                    output_flattened = (output_flattened + (mask[index*beam_width : (index+1)*beam_width, :])*(probability_list[index*beam_width : (index+1)*beam_width, :]))/torch.sqrt(lengths[index*beam_width : (index+1)*beam_width, :])
                    output_flattened = torch.reshape(output_flattened, (-1,))
                    topk = torch.topk(output_flattened, k=beam_width, dim=0)
                    for index_ in range(len(topk[1])):
                        beam_ind = topk[1][index_].item()//self.config["vocab_size"]
                        new_sentences[current_index, :sentences.shape[1]] = sentences[index*beam_width + beam_ind, :]
                        new_sentences[current_index, sentences.shape[1]] = topk[1][index_].item()%self.config["vocab_size"]
                        if topk[1][index_].item()%self.config["vocab_size"] == end_token:
                            temp_mask[current_index, 0] = 0
                        else:
                            temp_mask[current_index, 0] = mask[index*beam_width + beam_ind]
                        carry_forward_indexes[current_index] = index*beam_width + beam_ind
                        current_index += 1

                memory_global_list = [item[:,carry_forward_indexes,:] for item in memory_global_list]
                memory_local_list = [item[:,carry_forward_indexes,:] for item in memory_local_list]
                cell_state_global = cell_state_global[:,carry_forward_indexes,:]
                cell_state_local = cell_state_local[:,carry_forward_indexes,:]
                local_context = local_context[carry_forward_indexes, :, :]
                global_context = global_context[carry_forward_indexes, :, :]
                probability_list = probability_list[carry_forward_indexes,:]
                lengths = lengths[carry_forward_indexes, :]

                assert current_index == self.batch_size*beam_width
                mask = temp_mask
                sentences = new_sentences

                probability_list = probability_list + (mask*torch.gather(output_local, dim=1, index=new_sentences[:,-1].long().view(-1,1)))
                lengths += mask

            best_sentences = [sentences[index*beam_width, :] for index in range(self.batch_size)]
            best_sentences = torch.stack(best_sentences, dim=0)
            return best_sentences[:, 1:]


        for time_step in range(self.max_caption_length):
            input_ts = torch.unsqueeze(self.input_embeddings(input_ts), dim=1)
            # One time_step for global decoder
            output_global, memory_global, cell_state_global, global_context = self.global_decoder(input_ts, global_encoder_outputs, memory_global_list, cell_state_global, global_context)
            memory_global_list.append(memory_global)

            # One time_step for local decoder
            output_local, memory_local, cell_state_local, local_context = self.local_decoder(torch.cat([input_ts, output_global], dim=2), local_encoder_outputs, memory_local_list, cell_state_local, local_context)
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