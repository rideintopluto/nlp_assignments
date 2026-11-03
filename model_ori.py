from torch import nn

class CBOW(nn.Module):
    def __init__(self, vocab_size, embedding_dim):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.output = nn.Linear(embedding_dim, vocab_size)

    def forward(self, context_ids):
        word_vectors = self.embedding(context_ids)
        context_vector = word_vectors.mean(dim=1)
        logits = self.output(context_vector)

        return logits