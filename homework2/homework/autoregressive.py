import abc

import torch


def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "AutoregressiveModel"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


class Autoregressive(abc.ABC):
    """
    Base class for all autoregressive models.
    Implement a specific model below.
    """

    @abc.abstractmethod
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Take a tensor x (B, h, w) if integers as input.
        Produce a probability over the next token as an output (B, h, w, n_token).
        Make sure the model is auto-regressive:
          - The first output result[:, 0, 0] does not depend on any input
          - The second output result[:, 0, 1] depends only on x[:, 0, 0]
          - etc.

        Hint 1: Flatten the tensor into a sequence.
        Hint 2: A positional embedding can help, but is not required.
        Hint 3: You need to shift the input sequence by 1 position. Do this after embedding the
                values, and before passing them through your model. (torch.concat or
                torch.nn.ConstantPad1d both work)
        """

    def generate(self, B: int = 1, h: int = 20, w: int = 30, device=None) -> torch.Tensor:  # noqa
        """
        Use your generative model to produce B new token images of size (B, h, w) and type (int/long).
        """


class AutoregressiveModel(torch.nn.Module, Autoregressive):
    """
    Implement an auto-regressive model.
    The input is a set of patch tokens (integers), the output is an image of probability.
    You need to implicitly shift your inputs by one position in the forward pass.
    Make sure n_tokens matches your BSQ dimension (2**codebook_bits_).

    Hint: You will need the torch.nn.Embedding function
    Hint: You can use torch.nn.TransformerEncoderLayer if you'd like
    Hint: You can complete this homework without using positional embeddings
    """

    def __init__(self, d_latent: int = 128, n_tokens: int = 2**10):
        super().__init__()
        # Store parameters for later use
        # Notes about the parameters:
        # d_latent is the dimension of the latent space
        # n_tokens is the number of tokens in the vocabulary
        self.d_latent = d_latent
        self.n_tokens = n_tokens

        # Hyperparameters
        N_HEADS = 8

        # Learnable lookup table that stores fixed-size vector representations (embeddings / latent variables) for each token
        # This layer starts off as random, and is learned during training
        self.token_embedding = torch.nn.Embedding(n_tokens, d_latent)

        # Initial prediction embedding (learnable parameter)
        # 1 - initial batch dimension
        # 1 - Initiation token for the initial embedding
        # d_latent - The dimensionality of the latent space (embedding size)
        self.initial_embedding = torch.nn.Parameter(torch.randn(1, 1, d_latent))

        # Use a TransformerEncoderLayer
        # d_model = the number of expected features in the input (required)
        # nhead = the number of heads in the multiheadattention models (required).
        self.transformer_encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_latent, nhead=N_HEADS, batch_first=True)

        # Create output projection that maps the transformer output to token logits
        self.linear_output = torch.nn.Linear(d_latent, n_tokens)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # input is a image of tokens (B, h, w) with integer values in [0, n_tokens-1]
        # setup local variables
        B, h, w = x.shape
        seq_len = h * w
        device = x.device

        metrics_dict = {}

        # flatten the input token image with view and reshape to (B, seq_len)
        # Use view instead of reshape to avoid potential issues with non-contiguous tensors
        # Convert to long type for embedding layer because the indices must be of type LongTensor
        x_long = x.view(B, seq_len).long # shape (B, seq_len)

        # Embed the input tokens using the token embedding layer
        x_embedding = self.token_embedding(x_long)

        # Prepare the initial embedding to prepend to the sequence
        # Expand the initial embedding to match the batch size to concatenate with x_embedding
        initial_embedding_expanded = self.initial_embedding.expand(B, -1, -1)  # shape (B, 1, d_latent)
        
        # Shift the embedded sequence by prepending the initial embedding and removing the last token
        # x_embedding[:, :-1, :] gets all tokens except the last one
        # dim=1 indicates concatenation along the sequence length dimension
        x_embedding = torch.cat([initial_embedding_expanded, x_embedding[:, :-1, :]], dim=1)  # shape (B, seq_len, d_latent)

        # Intantiate a casual mask using generate_square_subsequent_mask
        causal_mask = torch.nn.Transformer.generate_square_subsequent_mask(sz=seq_len, device=device)

        # Pass through the transformer encoder
        x_transform_encoding = self.transformer_encoder_layer(src=x_embedding, src_mask=causal_mask, is_causal=True) # shape (B, seq_len, d_latent)

        # Project to output logits using the linear output layer to predict token probabilities
        logits = self.linear_output(x_transform_encoding) # shape (B, seq_len, n_tokens)

        # Reshape logits to (B, h, w, n_tokens)
        logits = logits.view(B, h, w, self.n_tokens)

        return logits, metrics_dict

    def generate(self, B: int = 1, h: int = 30, w: int = 20, device=None) -> torch.Tensor:  # noqa
        # setup local variables
        seq_len = h * w

        # Initialize the output tensor with zeros (B, h, w) of type long
        output_tokens = torch.zeros((B, h, w), dtype=torch.long, device=device)

        # Generate tokens one by one
        for i in range(h):
            for j in range(w):
                # Forward pass to get logits for the current state of output_tokens
                logits, _ = self.forward(output_tokens)

                # Get the logits for the current position (i, j) for all batches
                current_logits = logits[:, i, j, :]  # shape (B, n_tokens)

                # Sample from the logits to get the next token for all batches
                probabilities = torch.nn.functional.softmax(current_logits, dim=-1)  # shape (B, n_tokens)
                next_tokens = torch.multinomial(probabilities, num_samples=1).squeeze(-1)  # shape (B,)

                # Assign the sampled tokens to the output tensor
                output_tokens[:, i, j] = next_tokens

        return output_tokens
