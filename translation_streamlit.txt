import pickle
import torch
from lang import Lang
from EncoderRNN import EncoderRNN
from AttnDecoderRNN import AttnDecoderRNN
import re
import string
from string import digits

import streamlit as st

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30

with open('input_lang.pkl', 'rb') as f:
    input_lang = pickle.load(f)
    
with open('output_lang.pkl', 'rb') as f:
    output_lang = pickle.load(f)

attn_model = 'general'
hidden_size = 128
n_layers = 1
dropout_p = 0.05

encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=dropout_p)

encoder.load_state_dict(torch.load('model_weights/encoder.pth', map_location='cpu'))
decoder.load_state_dict(torch.load('model_weights/decoder.pth', map_location='cpu'))

st.title('Translation App')

def preprocess_text(text):
    text = text.lower()
    text = re.sub("'", '', text)
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    text = text.strip()
    text = re.sub(" +", " ", text)
    return text

def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()
        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)
        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei], encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]
        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS
        decoder_hidden = encoder_hidden
        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)
        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])
            decoder_input = topi.squeeze().detach()
        return decoded_words

if st.button('Translate'):
    get_string = st.text_area("Enter English string:")
    if get_string:
        clean = preprocess_text(get_string)
        input_sentence = ' '.join(clean.split()[:30])
        output_words = evaluate(encoder, decoder, input_sentence)
        output_sentence = ' '.join(output_words)
        st.write('Translated text:', output_sentence)
