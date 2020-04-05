import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        
        # Embedding layer that turns words into a vector of a specified size
        self.word_embeddings = nn.Embedding(num_embeddings=vocab_size, embedding_dim=embed_size)

        # LSTM takes embedded word vectors (of a specified size) as inputs 
        # and outputs hidden states of size hidden_dim        
        # Pass "batch_first=True" since our inputs and outputs have the batch as first dim
        self.lstm = nn.LSTM(input_size=embed_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            batch_first=True)  
        
        # Linear layer that maps the hidden state output dimension 
        # to the vocabulary size we want as output
        self.hidden2tag = nn.Linear(hidden_size, vocab_size)        
       
    def forward(self, features, captions):
        # Remove the last token from "captions" (<end>), see picture in Step 4
        captions = captions[:, :-1]
        
        # Create embeddings from captions
        captions_embeddings = self.word_embeddings(captions)
        
        # Concatenate features and embeddings. Need to add an extra dimension
        # to the input features, since:
        # features.shape            = [batch_size,                  embed_size]
        # captions_embeddings.shape = [batch_size, sequence_length, embed_size]
        features = features.view((features.shape[0], 1, features.shape[1]))
        lstm_input = torch.cat((features, captions_embeddings), dim=1)
        
        # Pass to LSTM. Use default zero initialization for the hidden state
        out, hidden = self.lstm(lstm_input)
        
        # Convert from hidden space to vocabulary space        
        return self.hidden2tag(out)
        
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        pass