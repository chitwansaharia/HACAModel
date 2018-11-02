import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F
from utils import *
import numpy as np
import pickle
from custom_rnn import customRNN

# Vocabulary (Used while testing for beam search)
vocab_path = "./meta_data/vocab.pkl"
vocab = pickle.load(open(vocab_path,'rb'))
end_token = vocab["<END>"]

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


    def forward(self, input, encoder_outputs, hidden_state, cell_state):
        batch_size = input.shape[0]

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

    def forward(self, input, ss_epsilon, beam_search = False, beam_width = None):
        # obtaining outputs from low level and high level encoders of both modalities

        visual_outputs = self.visual_encoder(input["visual_features"])
        encoder_outputs = {"visual" : visual_outputs, "audio" : None}

        if self.use_audio_modality:
            audio_outputs = self.audio_encoder(input["audio_features"])
            encoder_outputs["audio"] = audio_outputs

        # Hidden states and cell states for local decoder
        memory_local, cell_state_local = self.local_decoder.decoder_rnn.initHidden()
        memory_local, cell_state_local = torch.stack([memory_local]*self.batch_size, dim=1), torch.stack([cell_state_local]*self.batch_size, dim=1)

        output_caption = []
        input_ts = input["caption_x"][:, 0]

        # Beam Search Code (Can only be used while testing)
        if beam_search:
            lengths = torch.ones([self.batch_size*beam_width, 1], device=self.device)
            mask = torch.ones([self.batch_size*beam_width, 1], device=self.device)

            # Reinitializing the memory and cell states
            # Hidden states and cell states for local decoder
            memory_local, cell_state_local = self.local_decoder.decoder_rnn.initHidden()
            memory_local, cell_state_local = torch.stack([memory_local]*self.batch_size*beam_width, dim=1), torch.stack([cell_state_local]*self.batch_size*beam_width, dim=1)

            softmax = nn.Softmax(dim=1)
            sentences = torch.zeros([self.batch_size*beam_width, 1], device=self.device)
            sentences[:,0] = input_ts[0]
            probability_list = torch.zeros([self.batch_size*beam_width, 1], device=self.device)

            # Expanding the encoder outputs by repeating each instance beam_width times
            for key in encoder_outputs:
                temp = encoder_outputs[key]
                temp2 = torch.zeros([temp.shape[0]*beam_width, temp.shape[1], temp.shape[2]], device = self.device)
                for index in range(self.batch_size):
                    temp2[index*beam_width:(index+1)*beam_width, :, :] = temp[index, :, :]
                encoder_outputs[key] = temp2

            for time_step in range(self.max_caption_length):
                # Prediction at last time step
                input_ts = torch.unsqueeze(self.input_embeddings(sentences[:,-1].long()), dim=1)
                # Pass through the local decoder to obtain the new predictions
                output_local, memory_local, cell_state_local = self.local_decoder(input_ts, encoder_outputs, memory_local, cell_state_local)

                # Applying softmax to the decoder output,  and then taking the log to produce log probabilities
                output_local = softmax(torch.squeeze(output_local, dim=1))
                output_local = torch.log(output_local)
                # new_sentences store the fresh batch of sentences (which got highest probability)
                new_sentences = torch.zeros([self.batch_size*beam_width, sentences.shape[1]+1], device = self.device)

                current_index = 0
                # Stores the indices of present sentences that will be carried forward
                carry_forward_indexes = [0]*self.batch_size*beam_width
                temp_mask = torch.ones([self.batch_size*beam_width, 1], device=self.device)
                # Processing each batch element separately
                for index in range(self.batch_size):
                    # Decoder output corresponding to a particular batch output
                    output_flattened = output_local[index*beam_width : (index+1)*beam_width,:]
                    # Adding the log probabilities (masking those sentences that have already encountered <END>)
                    output_flattened = (output_flattened + (mask[index*beam_width : (index+1)*beam_width, :])*(probability_list[index*beam_width : (index+1)*beam_width, :]))/torch.sqrt(lengths[index*beam_width : (index+1)*beam_width, :])
                    # Flattening and finding out the top k tokens
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

                # Updating the memory, cell state, probability_list
                memory_local = memory_local[:,carry_forward_indexes,:]
                cell_state_local = cell_state_local[:,carry_forward_indexes,:]
                probability_list = probability_list[carry_forward_indexes,:]
                lengths = lengths[carry_forward_indexes, :]

                assert current_index == self.batch_size*beam_width
                mask = temp_mask
                sentences = new_sentences
                # Updating the probability list and lengths
                probability_list = probability_list + (mask*torch.gather(output_local, dim=1, index=new_sentences[:,-1].long().view(-1,1)))
                lengths += mask

            # Picking the top sentences from each batch element
            best_sentences = [sentences[index*beam_width, :] for index in range(self.batch_size)]
            best_sentences = torch.stack(best_sentences, dim=0)
            return best_sentences[:, 1:]


        for time_step in range(self.max_caption_length):
            input_ts = torch.unsqueeze(self.input_embeddings(input_ts), dim=1)

            output_local, memory_local, cell_state_local = self.local_decoder(input_ts, encoder_outputs, memory_local, cell_state_local)

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