'''
SongSmith HiperGator Training Script

The following script is designed to train the SongSmith transformer GAN in HiperGator.
The script loads the training data and formats it, defines the model architectures used, and executes the training loop.
Once the training is finished, the resulting models and state dictionaries are saved to be transfered into the application backend/

'''

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
import matplotlib.colors as colors
import numpy as np



device = torch.device("cpu")
if torch.cuda.is_available():
  device = torch.device("cuda")
else:
  device = torch.device("cpu")



path = 'data'



files = os.listdir(path)

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

# add start and end tokens. <START> = 0 <END> = 1 in encoding
word_vocab.append("<START>")
word_vocab.append("<END>")

syll_vocab.append("<START>")
syll_vocab.append("<END>")

for s in songs:
    # add start and end tokens for each song
    s.insert(0,[0,0,0,"<START>", "<START>"]) 
    s.append([0,0,0,"<END>", "<END>"]) 
    for note in s:
        if note[3] not in word_vocab:
            word_vocab.append(note[3])
        if note[4] not in syll_vocab:
            syll_vocab.append(note[4])

# encode the words and syllables 
for s in songs:
    for note in s:
        note[3] = word_vocab.index(note[3])
        note[4] = syll_vocab.index(note[4]) 

# do the same thing as the paper and sample a random 20 note melody from each song
melodies = []
for s in songs:
    #TODO: some songs have less than 20 notes so this breaks!!
    #added loop to fix 
    if len(s) <= 20:
        continue
    idx = random.randint( 0, (len(s) - 1 - 20))
    m = s[idx:idx + 20]

    melodies.append(m)


# turn the melodies into tensors
temp = []
for m in melodies:
    m_tensor = torch.tensor(m).to(device)
    temp.append(m_tensor)
melodies = temp
print(len(melodies)) # each melody is now a tensor

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
    def __init__(self, unembedded_input_size, word_vocab_size, syll_vocab_size, embed_size = 62):
        super(Generator, self).__init__()
        device = torch.device("cuda")
        self.unembedded_input_size = unembedded_input_size
        size_wo_lyrics = unembedded_input_size - 2
        self.embedded_input_size = size_wo_lyrics + embed_size * 2
        self.embed_size = embed_size

        self.word_embedding = nn.Embedding(word_vocab_size, embed_size)
        self.syll_embedding = nn.Embedding(syll_vocab_size, embed_size)

        self.src_pos_encoder = PositionalEncoding(self.embedded_input_size, dropout = 0.5)

        encoder_layers = nn.TransformerEncoderLayer(
            d_model = self.embedded_input_size, 
            nhead = 32,
            dim_feedforward=512,
            dropout=0.5,
            batch_first=True
            )

        self.norm = nn.LayerNorm(self.embedded_input_size).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=16, norm=self.norm).to(device)

        # need linear layer to match input sizes for cross-attention in decoder
        self.encoder_out = nn.Linear(self.embedded_input_size, 3) 

        self.tgt_pos_encoder = PositionalEncoding(3, dropout = 0.5)

        decoder_layers = nn.TransformerDecoderLayer(
            d_model = 3, 
            nhead = 3,
            dim_feedforward=512,
            dropout=0.5,
            batch_first=True
            )
        
        self.decoder = nn.TransformerDecoder(decoder_layers, num_layers=16).to(device)

        self.init_weights()

    def forward(self, src, tgt, tgt_mask):
        batch_size = src.shape[0]
        seq_len = src.shape[1]
            
        src_emb = self.embed_lyrics(src)
        src_emb = src_emb * math.sqrt(self.embedded_input_size)
        src_emb = self.src_pos_encoder(src_emb.permute(1,0,2)).permute(1,0,2) 

        memory = self.encoder(src_emb)
        memory = self.encoder_out(memory)

        tgt = tgt * math.sqrt(3)
        tgt = self.tgt_pos_encoder(tgt.permute(1,0,2)).permute(1,0,2)
        output = self.decoder(tgt, memory, tgt_mask = torch.squeeze(tgt_mask))
        
        return output
    
    # returns X but with the words and syllables embedded and positional encoded
    def embed_lyrics(self, X):
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        assert X.shape[2] == self.unembedded_input_size, f"expected {self.unembedded_input_size} but got {X.shape[2]}"

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
    def __init__(self, unembedded_input_size, word_vocab_size, syll_vocab_size, embed_size = 60):
        super(Discriminator, self).__init__()
        self.unembedded_input_size = unembedded_input_size
        size_wo_lyrics = unembedded_input_size - 2
        self.embedded_input_size = size_wo_lyrics + embed_size * 2
        self.embed_size = embed_size

        self.word_embedding = nn.Embedding(word_vocab_size, embed_size)
        self.syll_embedding = nn.Embedding(syll_vocab_size, embed_size)

        self.pos_encoder = PositionalEncoding(self.embedded_input_size, dropout = 0.5) 
 
        encoder_layers = nn.TransformerEncoderLayer(
            d_model = self.embedded_input_size, 
            nhead = 3,
            dropout=0.5,
            batch_first=True
            )

        self.norm = nn.LayerNorm(self.embedded_input_size).to(device)
        self.encoder = nn.TransformerEncoder(encoder_layers, num_layers=16, norm=self.norm)
    
        self.encoder_out = nn.Linear(self.embedded_input_size, 1)
        self.activation = nn.Sigmoid()

        self.init_weights()

    def forward(self, src):
        batch_size = src.shape[0]
        seq_len = src.shape[1]

        src = self.embed_lyrics(src)
        src = src * math.sqrt(self.embedded_input_size)
        src = self.pos_encoder(src.permute(1,0,2)).permute(1,0,2) 

        output = self.encoder(src)
        output = self.encoder_out(output)
        output = output[:,-1,:].reshape(batch_size, 1, 1)

        return output
    
    # returns X but with the words and syllables embedded and positional encoded
    def embed_lyrics(self, X):
        batch_size = X.shape[0]
        seq_len = X.shape[1]
        assert X.shape[2] == self.unembedded_input_size, f"expected {self.unembedded_input_size} but got {X.shape[2]}"

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
        

batch_size = 64
seq_len = 20

dataset = MelodyDataset(melodies, word_vocab, syll_vocab)
dataloader = data.DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True, pin_memory = False)

Gen = Generator(6, dataset.get_word_vocab_size(), dataset.get_syll_vocab_size())
Disc = Discriminator(5, dataset.get_word_vocab_size(), dataset.get_syll_vocab_size())

#loads state dictonary to complete training from last stage
#state dictionary allows resuming training in different device
Gen.load_state_dict(torch.load("GenModel_deep.pt", map_location = "cuda")) #not in first training set
Disc.load_state_dict(torch.load("DiscModel_deep.pt", map_location = "cuda")) #not in first training set

#parallelize on gpus if possible
ngpu = torch.cuda.device_count()
if(device.type == 'cuda') and (ngpu > 1):
    Gen = nn.DataParallel(Gen, list(range(ngpu)))
    Disc = nn.DataParallel(Disc, list(range(ngpu)))

Gen.to(device)
Disc.to(device)

#set model parameters
gen_learn_rate = 5e-5
disc_learn_rate = 5e-5
num_epochs = 500

Gen_Optim = torch.optim.Adam(Gen.parameters(), lr = gen_learn_rate, betas=(0.5, 0.999))
Disc_Optim = torch.optim.Adam(Disc.parameters(), lr = disc_learn_rate, betas =(0.5, 0.999))


#generates generator examples for training
def gen_fake_train_examples(Gen, batch, batch_size, seq_len, detach, device):
        #gaussian (normally distributed) noise
        noise = torch.normal(0, 1, size=(batch_size, seq_len, 4), device=device)

        words = batch[:, :, 3].reshape(batch_size, seq_len, 1)
        syllables = batch[:, :, 4].reshape(batch_size, seq_len, 1)

        src = torch.cat((noise, words), dim=2)
        src = torch.cat((src, syllables), dim=2)
        src = src.to(device)

        tgt = batch[:, :, :3].to(device)
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt.shape[1]).to(device)
        
        #calls generator to generate melody from noise input
        fake_examples = Gen(src, tgt, tgt_mask.unsqueeze(0).repeat(8, 1, 1)).to(device)

        if (detach == True):
            fake_examples = fake_examples.detach()

        fake_examples = torch.cat((fake_examples, words), dim=2)
        fake_examples = torch.cat((fake_examples, syllables), dim=2)

        return fake_examples

#plots the losses
def plot(loss_D_fake, loss_D_real, loss_G):
    norm = colors.Normalize(0,1)
    norm(loss_D_real)
    norm(loss_D_fake)
    norm(loss_G)
    plt.title("Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot(loss_D_fake, color="orange", label = "d-fake")
    plt.plot(loss_D_real, color="green", label = "d-real")
    plt.plot(loss_G, color="blue", label = "gen")
    plt.legend(loc = "upper right")
    plt.savefig("Loss.png")
    plt.show()
            
def train(dataloader, batch_size, seq_len, Gen, Disc, Disc_Optim, Gen_Optim, num_epochs, device, train_steps_D, train_steps_G):
    criterion = nn.BCEWithLogitsLoss() #binary classification makes bcewithlogits ideal
    loss_G = []
    loss_D_fake = []
    loss_D_real = []
    
    for epoch in range(num_epochs):
        Gen.train()
        Disc.train()
        # train the discriminator
        total_D_Loss_Fake = 0
        total_D_Loss_Real = 0
        total_G_Loss = 0
        #trains for every batch in the dataloader
        for i, batch in enumerate(dataloader, 0):
            #Discriminator training
            Disc_Optim.zero_grad() #not sure if it should be optim

            fake_examples = gen_fake_train_examples(Gen, batch, batch_size, seq_len, detach = True, device = device)
            fake_predictions = Disc(fake_examples).squeeze()
            fake_targets = torch.zeros_like(fake_predictions).to(device) # want discrminiator to predict fake
            fake_D_loss = criterion(fake_predictions, fake_targets)

            #train using real data from the batch
            real_D_predictions = Disc(batch.to(device)).squeeze()
            real_D_target = torch.ones_like(real_D_predictions).to(device)
            real_D_loss = criterion(real_D_predictions, real_D_target)

            D_loss = ((real_D_loss + fake_D_loss)/2)
            
            D_loss.backward()
            Disc_Optim.step()
            total_D_Loss_Fake += fake_D_loss.item()
            total_D_Loss_Real += real_D_loss.item()

            #Generator training
            Gen_Optim.zero_grad() #not sure if it should be optim
            fake_examples = gen_fake_train_examples(Gen, batch, batch_size, seq_len, detach = False, device = device)
            D_predictions = Disc(fake_examples).squeeze()
            D_targets = torch.ones_like(D_predictions).to(device)
            G_loss = criterion(D_predictions, D_targets)

            G_loss.backward()
            Gen_Optim.step()

            total_G_Loss += G_loss.item()

        #keep track of losses
        loss_D_fake.append((total_D_Loss_Fake)/len(dataloader))
        loss_D_real.append((total_D_Loss_Real)/len(dataloader))
        loss_G.append((total_G_Loss)/len(dataloader))
	
        #saves model after certain number of epochs and prints to console the current losses
        if epoch % 50 == 0:
            torch.save(Gen, "models/GenModel_{}.pt".format(epoch))
            torch.save(Disc, "models/DiscModel_{}.pt".format(epoch))
            print("Fake Discriminator Loss: {} after {} epochs".format(loss_D_fake[epoch], epoch+1))
            print("Real Discriminator Loss: {} after {} epochs".format(loss_D_real[epoch], epoch+1))
            print("Generator Loss: {} after {} epochs".format(loss_G[epoch], epoch+1))
            
            
    plot(loss_D_fake, loss_D_real, loss_G)
    #saves both the trained models and their state dictionaries
    #discriminator is saved for consistency
    #generator is saved and transfered to backend
    torch.save(Gen.module.state_dict(), "GenModel_deep.pt")
    torch.save(Disc.module.state_dict(), "DiscModel_deep.pt")
    torch.save(Gen, "Gen_full.pt")
    torch.save(Disc, "Disc_full.pt")

#executes training with the hyperparameters defined above
train(dataloader, batch_size, seq_len, Gen, Disc, Disc_Optim, Gen_Optim, num_epochs, device, len(dataloader), len(dataloader))
