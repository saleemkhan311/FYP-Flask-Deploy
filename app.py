import pickle
from flask import Flask, request, jsonify, render_template

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim
import torch.nn.functional as F

from lang import Lang
from EncoderRNN import EncoderRNN
from AttnDecoderRNN import AttnDecoderRNN
#import EncoderRNN,  AttnDecoderRNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SOS_token = 0
EOS_token = 1
MAX_LENGTH = 30
with open('input_lang.pkl', 'rb') as f:
    input_lang = pickle.load(f)
    
with open('output_lang.pkl', 'rb') as f:
    output_lang = pickle.load(f)
 
 # Function to get index from language
def indexesFromSentence(lang, sentence):
    return [lang.word2index[word] for word in sentence.split(' ')]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

# Function to get variables from string
def variable_from_sentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    var = Variable(torch.LongTensor(indexes).view(-1, 1))
    return var
    
# Evaluation function
def evaluate(encoder, decoder, sentence, max_length=MAX_LENGTH):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_length = input_tensor.size()[0]
        encoder_hidden = encoder.initHidden()

        encoder_outputs = torch.zeros(max_length, encoder.hidden_size, device=device)

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(input_tensor[ei],
                                                     encoder_hidden)
            encoder_outputs[ei] += encoder_output[0, 0]

        decoder_input = torch.tensor([[SOS_token]], device=device)  # SOS

        decoder_hidden = encoder_hidden

        decoded_words = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            if topi.item() == EOS_token:
                decoded_words.append('<EOS>')
                break
            else:
                decoded_words.append(output_lang.index2word[topi.item()])

            decoder_input = topi.squeeze().detach()

        return decoded_words, decoder_attentions[:di + 1]
    
attn_model = 'general'
hidden_size = 128
n_layers = 1
dropout_p = 0.05

# Initialize models
encoder = EncoderRNN(input_lang.n_words, hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_words, dropout_p=dropout_p)

# Load trained weights ('cpu' since I am using CPU to calculate instead of CUDA GPU)
encoder.load_state_dict(torch.load('model_weights/encoder.pth',map_location='cpu'))
decoder.load_state_dict(torch.load('model_weights/decoder.pth',map_location='cpu'))


import re
import string
from string import digits

def preprocess_text(text):
    # Lowercase all characters
    text = text.lower()
    
    # Remove quotes
    text = re.sub("'", '', text)
    
    # Remove all the special characters
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    
    # Remove all numbers from text
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    
    # Remove extra spaces
    text = text.strip()
    text = re.sub(" +", " ", text)
    
    return text


def handle_key_error(input_sentence, lang):
    words = input_sentence.split()
    clean_words = []
    for word in words:
        try:
            lang.word2index[word]  # Check if the word exists in the dictionary
            clean_words.append(word)
        except KeyError:
            pass  # Ignore the word if it is not found in the dictionary
    clean_sentence = ' '.join(clean_words)
    print(input_sentence)
    print(clean_sentence)
    return clean_sentence

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    if request.method == 'POST':
        get_string = request.form["english_string"]
        prepross = preprocess_text(get_string)
        clean = handle_key_error(prepross,input_lang)
        output_words, decoder_attn = evaluate(encoder, decoder,clean)
        output_sentence = ' '.join(output_words)
     
    return render_template('result.html', output_sentence = output_sentence)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)