"""
Classes for sequence learning
"""
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm


class SequenceClassifierNonRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim):
        """
        Network for the classification of sequences without sequential learning.
        
        Args:
            vocab_size (int): Vocabulary size.
            embedding_dim (int): Embedding dimension.
            output_dim (int): Number of output classes.
        """
        super(SequenceClassifierNonRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc = nn.Linear(embedding_dim, output_dim)

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input of dimension (batch_size, sequence_length)
        """
        # Creazione degli embedding: (batch_size, sequence_length, embedding_dim)
        embedded = self.embedding(x)

        # Media degli embedding lungo la dimensione della sequenza
        # Resulting shape: (batch_size, embedding_dim)
        pooled = embedded.mean(dim=1)

        # Passaggio attraverso il livello Fully Connected
        out = self.fc(pooled)
        return out


class SequenceClassifierRNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, output_dim, hidden_dim=None):
        super(SequenceClassifierRNN, self).__init__()
        if hidden_dim is None:
            hidden_dim = 2 * output_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x è di dimensione (batch_size, sequence_length)
        embedded = self.embedding(x)
        # output di RNN: (batch_size, sequence_length, hidden_dim)
        rnn_out, _ = self.rnn(embedded)
        # prendere solo l'ultimo hidden state per la classificazione
        final_out = rnn_out[:, -1, :]
        out = self.fc(final_out)
        return out


def training(model_class, dataset, loader, embedding_dim: int = 8, learning_rate: float = 0.001, epochs: int = 20):
    vocab_size = len(dataset.char_to_idx)
    output_dim = len(dataset.target_to_idx)
    model = model_class(vocab_size=vocab_size, embedding_dim=embedding_dim, output_dim=output_dim)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    history = []
    run = list(range(epochs))
    for epoch in tqdm(run):
        for batch_sequences, batch_labels in loader:
            optimizer.zero_grad()
            outputs = model(batch_sequences)
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()
        history.append(loss.item())
    return model, history