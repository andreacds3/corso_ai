import torch
import torch.nn as nn


class RNNClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, dropout_rate=0.1):
        super(RNNClassifier, self).__init__()

        # 1. Embedding: Trasforma l'indice (es. 45) in un vettore denso
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)

        # 2. LSTM Layer
        # batch_first=True significa che l'input è (batch, seq_len, features)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, bidirectional=True,
                            batch_first=True)

        # 3. Fully Connected Layer (Output). Devo moltiplicare l'hidden_dim per 2 perchè la rete è bidirezionale
        self.fc = nn.Linear(hidden_dim * 2, output_dim)

        #Il dropout serve a regolarizzare la rete. Praticamente spegne il dropout_rate per cento dei neuroni ad ogni iterazione
        self.dropout = nn.Dropout(dropout_rate)
        # 4. Sigmoide (per ottenere probabilità tra 0 e 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, text):
        # text shape: [batch_size, seq_len]
        embedded = self.embedding(text)

        # rnn output: [batch_size, seq_len, hidden_dim]
        #hidden ha shape: [num_layers * num_directions, batch_size, hidden_dim]
        output, (hidden, cell) = self.rnn(embedded)
        hidden_forward = hidden[-2,:,:]
        hidden_backward = hidden[-1,:,]
        hidden_final = torch.cat((hidden_forward, hidden_backward), dim=1)
        # shape risultante: [batch_size, hidden_dim * 2]
        hidden_final = self.dropout(hidden_final)
        return self.sigmoid(self.fc(hidden_final))