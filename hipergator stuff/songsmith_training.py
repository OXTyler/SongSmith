#!/usr/bin/env python
# coding: utf-8

# 

# In[1]:


import os
import random
import math
import torch
from torch import nn
from torch.nn import functional as F
import torchtext
import torchdata
from torch.utils import data
import matplotlib.pyplot as plt
import numpy as np



device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")



path = 'data'



files = os.listdir(path)
print(len(files))
#print(files[:5])



songs = []
for file in files:
    try:
        song = np.load(os.path.join(path, file), allow_pickle=True)[0]
        songs.append(song)
    except:
        continue
# for each song
# first list = [seconds since beginning, length, pitch (hz)]
# second list = [pitch (MIDI), duration, duration of the rest before the note] 
# third list = words of the song
# fourth list = syllables of the song
#songs


# reformat songs to be more readable
songs_reformatted = []
for s in songs:
    s_reformatted = []
    notes = s[1]
    words = s[2]
    syllables = s[3]
    for i in range(0, len(notes)):
        for j in range(0, len(notes[i])):
            note_reformatted = notes[i][j].copy()
            
            note_reformatted.append(words[i][j].lower()) # lowercase while we're at it
            note_reformatted.append(syllables[i][j].lower())

            s_reformatted.append(note_reformatted)
    songs_reformatted.append(s_reformatted)
songs = songs_reformatted
#songs # songs are now [pitch (MIDI), duration, rest duration, word, syllable]


# build the vocabulary
word_vocab_size = 0
word_vocab = []

syll_vocab_size = 0
syll_vocab = []

# add start and end tokens. <START> = 1 <END> = 2 in encoding
word_vocab.append("<START>")
word_vocab.append("<END>")

syll_vocab.append("<START>")
syll_vocab.append("<END>")

for s in songs:
    for note in s:
        if note[3] not in word_vocab:
            word_vocab.append(note[3])
        if note[4] not in syll_vocab:
            syll_vocab.append(note[4])

#print(word_vocab)
#print(len(word_vocab))
#print(syll_vocab)
#print(len(syll_vocab))



# encode the words and syllables 
for s in songs:
    for note in s:
        note[3] = word_vocab.index(note[3])
        note[4] = syll_vocab.index(note[4]) 
#songs



# do the same thing as the paper and sample a random 20 note melody from each song
melodies = []
for s in songs:
    #TODO: some songs have less than 20 notes so this breaks!!
    #added loop to fix 
    if len(s) < 22:
        continue
    idx = random.randint( 0, (len(s) - 1 - 20))
    m = s[idx:idx + 20]

    m.insert(0, [0,0,0,1,1]) # add start token
    m.append([0,0,0,2,2]) # add end token

    melodies.append(m)

#melodies


# turn the melodies into tensors
temp = []
for m in melodies:
    m_tensor = torch.tensor(m).to(device)
    temp.append(m_tensor)
melodies = temp
print(len(melodies))
#melodies # each melody is now a tensor



class MelodyDataset(data.Dataset):
    def __init__(self, melodies, word_vocab, syll_vocab):
       self.melodies = melodies
       self.word_vocab = word_vocab
       self.syll_vocab = syll_vocab
       self.word_vocab_size = len(word_vocab)
       self.syll_vocab_size = len(syll_vocab)

    def __len__(self):
        return len(self.melodies)
    
    def __getitem__(self, idx):
        return self.melodies[idx]

    def get_word_vocab_size(self):
        return self.word_vocab_size

    def get_syll_vocab_size(self):
        return self.syll_vocab_size

    def word2int(self, word):
        return self.word_vocab.index(word)

    def int2word(self, idx):
        return self.word_vocab[idx]

    def syll2int(self, syll):
        return syll_vocab.index(syll) 
    
    def int2syll(self, idx):
        return syll_vocab[idx]
    
    #TODO: not implemented
    # def gen_train_data():
        


dataset = MelodyDataset(melodies, word_vocab, syll_vocab)
dataloader = data.DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True, pin_memory = False)
#for batch in dataloader:
    #print(batch.shape)
    #print(batch[0])
dataset[0]



class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        device = torch.device("cuda")
        pe = torch.zeros(max_len, d_model).to(device)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        if d_model%2 != 0:
            pe[:, 1::2] = torch.cos(position * div_term)[:,0:-1]
        else:
            pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [sequence length, batch size, embed dim]
            output: [sequence length, batch size, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)



# input: sequence of {random noise, word, syllable}
# output: one sequence of {MIDI, duration, rest duration, word, syllable}
class Generator(nn.Module):
    def __init__(self, unembedded_input_size, word_vocab_size, syll_vocab_size, embed_size = 10):
        super(Generator, self).__init__()
        device = torch.device("cuda")
        self.unembedded_input_size = unembedded_input_size
        size_wo_lyrics = unembedded_input_size - 2
        self.embedded_input_size = size_wo_lyrics + embed_size * 2
        self.embed_size = embed_size

        self.word_embedding = nn.Embedding(word_vocab_size, embed_size)
        self.syll_embedding = nn.Embedding(syll_vocab_size, embed_size)

        self.pos_encoder = PositionalEncoding(self.embedded_input_size, dropout = 0) # for now, no dropout cause no trust

        encoder_layers = nn.TransformerEncoderLayer(
            d_model = self.embedded_input_size, 
            nhead = 2,
            dim_feedforward=4,
            dropout=0,
            batch_first=True
            )

        self.norm = nn.LayerNorm(self.embedded_input_size).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=4, norm=self.norm).to(device)

        # need linear layer to match input sizes for cross-attention in decoder
        self.encoder_out = nn.Linear(self.embedded_input_size, 3) 

        decoder_layers = nn.TransformerDecoderLayer(
            d_model = 3, 
            nhead = 1,
            dim_feedforward=4,
            dropout=0,
            batch_first=True
            )
        
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=4).to(device)

        self.init_weights()

    def forward(self, src):
        device = torch.device("cuda")
        batch_size = src.shape[0]
        seq_len = src.shape[1]

        src_emb = self.embed_lyrics(src)
        src_emb = self.pos_encoder(src_emb) 

        memory = self.encoder(src_emb)
        memory = self.encoder_out(memory)

        # generate a melody with same seq_len as src
        tgt = torch.zeros((batch_size, 1, 3)).to(device) # start with zero note 
        for i in range(seq_len - 1): # -1 cause we already have 0 note
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
            output = self.decoder(tgt, memory, tgt_mask).to(device)
            output = output[:, -1] # extracts the inference because the output is formatted weirdly 
            output = output.view(batch_size, 1, -1) # reshapes output to be batch friendly
            tgt = torch.cat((tgt, output), dim=1)

        # add the lyrics
        words = src[:, :, self.unembedded_input_size - 2].reshape(batch_size, seq_len, 1)
        sylls = src[:, :, self.unembedded_input_size - 1].reshape(batch_size, seq_len, 1)        
        tgt = torch.cat((tgt, words), dim=2)
        tgt = torch.cat((tgt, sylls), dim=2)

        return tgt
    
    # returns X but with the words and syllables embedded and positional encoded
    def embed_lyrics(self, X):
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        assert X.shape[2] == self.unembedded_input_size

        # extract words and syllables from X
        words = X[:, :, self.unembedded_input_size - 2].long()
        sylls = X[:, :, self.unembedded_input_size - 1].long()

        # do embedding
        words_embedded = self.word_embedding(words)
        sylls_embedded = self.syll_embedding(sylls)

        # reshape so you can concate
        words_embedded = words_embedded.view(batch_size, seq_len, self.embed_size)
        sylls_embedded = sylls_embedded.view(batch_size, seq_len, self.embed_size)

        # concat everything and return
        X_embedded = X[:, :, :self.unembedded_input_size - 2] # minus 2 to kill words and syllables
        X_embedded = torch.cat((X_embedded, words_embedded), dim=2)
        X_embedded = torch.cat((X_embedded, sylls_embedded), dim=2)

        return X_embedded

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.word_embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.syll_embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.encoder_out.bias)
        nn.init.uniform_(self.encoder_out.weight, -initrange, initrange)


# input: a melody example formatted as {MIDI, duration, rest duration, word, syllable}
# output: a single number that represents if it's real/fake. real = 1, fake = 0
class Discriminator(nn.Module):
    def __init__(self, unembedded_input_size, word_vocab_size, syll_vocab_size, embed_size = 9):
        super(Discriminator, self).__init__()
        self.unembedded_input_size = unembedded_input_size
        size_wo_lyrics = unembedded_input_size - 2
        self.embedded_input_size = size_wo_lyrics + embed_size * 2
        self.embed_size = embed_size

        self.word_embedding = nn.Embedding(word_vocab_size, embed_size)
        self.syll_embedding = nn.Embedding(syll_vocab_size, embed_size)

        self.pos_encoder = PositionalEncoding(self.embedded_input_size, dropout = 0) # for now, no dropout cause no trust
 
        encoder_layers = nn.TransformerEncoderLayer(
            d_model = self.embedded_input_size, 
            nhead = 3,
            dim_feedforward=4,
           dropout=0,
            batch_first=True
            )

        self.norm = nn.LayerNorm(self.embedded_input_size)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=4, norm=self.norm)
    
        self.encoder_out = nn.Linear(self.embedded_input_size, 1)
        self.sigmoid = nn.Sigmoid()

        self.init_weights()

    def forward(self, src):
        src = self.embed_lyrics(src)
        src = self.pos_encoder(src) 
        output = self.encoder(src)
        # transGAN paper does this (https://github.com/asarigun/TransGAN/blob/846c067b69f25f65b512e37a9bb78dba6058334c/models.py#L184)
        # I don't know exactly why and I feel like there are better ways
        # but it makes the shapes work out and outputs sensible numbers upon init so...
        output = output[:, 0] # just takes first of every sequence
        output = self.encoder_out(output)
        output = self.sigmoid(output)
        return output
 
    # returns X but with the words and syllables embedded and positional encoded
    def embed_lyrics(self, X):
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        assert X.shape[2] == self.unembedded_input_size

        # extract words and syllables from X
        words = X[:, :, self.unembedded_input_size - 2].long()
        sylls = X[:, :, self.unembedded_input_size - 1].long()

        # do embedding
        words_embedded = self.word_embedding(words)
        sylls_embedded = self.syll_embedding(sylls)

        # reshape so you can concate
        words_embedded = words_embedded.view(batch_size, seq_len, self.embed_size)
        sylls_embedded = sylls_embedded.view(batch_size, seq_len, self.embed_size)

        # concat everything and return
        X_embedded = X[:, :, :self.unembedded_input_size - 2] # minus 2 to kill words and syllables
        X_embedded = torch.cat((X_embedded, words_embedded), dim=2)
        X_embedded = torch.cat((X_embedded, sylls_embedded), dim=2)

        return X_embedded

    def init_weights(self):
        initrange = 0.1
        nn.init.uniform_(self.word_embedding.weight, -initrange, initrange)
        nn.init.uniform_(self.syll_embedding.weight, -initrange, initrange)
        nn.init.zeros_(self.encoder_out.bias)
        nn.init.uniform_(self.encoder_out.weight, -initrange, initrange)
        
Gen = Generator(6, dataset.get_word_vocab_size(), dataset.get_syll_vocab_size())

Disc = Discriminator(5, dataset.get_word_vocab_size(), dataset.get_syll_vocab_size())

ngpu = torch.cuda.device_count()
if(device.type == 'cuda') and (ngpu > 1):
    Gen = nn.DataParallel(Gen, list(range(ngpu)))
    Disc = nn.DataParallel(Disc, list(range(ngpu)))

Gen.to(device)
Disc.to(device)

gen_learn_rate = 0.05
disc_learn_rate = 0.05
num_epochs = 1000

Gen_Optim = torch.optim.Adam(Gen.parameters(), lr = gen_learn_rate)
Disc_Optim = torch.optim.Adam(Disc.parameters(), lr = disc_learn_rate)

batch_size = 2
seq_len = 22

batch = list(dataloader)[0]
batch.shape


import matplotlib.pyplot as plt


def train(train_data, Gen, Disc, Disc_Optim, Gen_Optim, num_epochs, device, train_steps_D, train_steps_G):
    criterion = nn.BCELoss() # Make this an input param so we can change loss function
    #TODO: default values for parameters
    batch = list(dataloader)[0].to(device)

    noise = torch.normal(0, 1, size=(batch_size ,seq_len, 4), device=device)

    words = batch[:, :, 3].reshape(batch_size, seq_len, 1)
    syllables = batch[:, :, 4].reshape(batch_size, seq_len, 1)

    src = torch.cat((noise, words), dim=2)
    src = torch.cat((src, syllables), dim=2)
    loss_G = []
    loss_D = []
    print("Training started!")
    for epoch in range(num_epochs):


        Gen.train()
        Disc.train()

        #need to make training data
            # train the discriminator
        total_D_Loss = 0
        for num_steps_D, data in enumerate(train_data, 0):

            fake_examples = Gen(src.to(device)).detach()
            fake_predictions = Disc(fake_examples)
            fake_targets = torch.zeros(fake_predictions.shape).to(device) # want discrminiator to predict fake
            fake_D_loss = criterion(fake_predictions, fake_targets)
            fake_D_loss.backward() 

            #train using real data from the batch
            #data should be dataloader iterator
            real_D_predictions = Disc(data.to(device))
            real_D_target = torch.ones(real_D_predictions.shape).to(device)
            real_D_target = real_D_target.to(device)
            real_D_loss = criterion(real_D_predictions, real_D_target)
            real_D_loss.backward

            Disc_Optim.step
            total_D_Loss += ((real_D_loss.item() + fake_D_loss.item())/2)
            total_D_Loss

            if num_steps_D == train_steps_D:
                break

        loss_D.append((total_D_Loss))

        #print("Disc loss: {}".format(total_D_Loss))
        total_G_Loss = 0
        for num_steps_G, data in enumerate(train_data, 0):
            # train the Generator
            Gen_Optim.zero_grad()

            G_examples = Gen(src)
            D_examples = Disc(G_examples)
            G_target = torch.ones(D_examples.shape)
            G_target = G_target.to(device)
            fake_G_loss = criterion(D_examples, G_target)

            fake_G_loss.backward()
            Gen_Optim.step()

            total_G_Loss += fake_G_loss.item()

            if num_steps_G == train_steps_G:
                break

        loss_G.append((total_G_Loss))
	
        if epoch % 50 == 0:
            #torch.save(Gen, "GenModel_{}.pt".format(epoch))
            torch.save({'epoch': epoch,
                'model_state_dict': Gen.state_dict(),
                'optimizer_state_dict': Gen_Optim.state_dict(),
                'loss': criterion}, 
                '/blue/cis4914/transfer/models/gen.pth')

            #torch.save(Disc, "DiscModel_{}.pt".format(epoch))
            torch.save({'epoch': epoch,
                'model_state_dict': Disc.state_dict(),
                'optimizer_state_dict': Disc_Optim.state_dict(),
                'loss': criterion}, 
                '/blue/cis4914/transfer/models/disc.pth')
    

    plt.title("DiscLoss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(loss_D, color="red")
    plt.savefig("DiscLoss.png")
    plt.show()
    plt.title("GenLoss")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.plot(loss_G, color="red")
    plt.savefig("GenLoss.png")
    plt.show()
    torch.save(Gen, "GenModel.pt")
    torch.save(Disc, "DiscModel.pt")

train(dataloader, Gen, Disc, Disc_Optim, Gen_Optim, num_epochs, device, 5, 5)
