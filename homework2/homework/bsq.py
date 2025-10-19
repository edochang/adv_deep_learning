import abc

import torch

from .ae import PatchAutoEncoder

# Copilot was used to assist and learn the concepts for this implementation.
def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "BSQPatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def diff_sign(x: torch.Tensor) -> torch.Tensor:
    """
    A differentiable sign function using the straight-through estimator.
    Returns -1 for negative values and 1 for non-negative values.
    """
    sign = 2 * (x >= 0).float() - 1
    return x + (sign - x).detach()


class Tokenizer(abc.ABC):
    """
    Base class for all tokenizers.
    Implement a specific tokenizer below.
    """

    @abc.abstractmethod
    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Tokenize an image tensor of shape (B, H, W, C) into
        an integer tensor of shape (B, h, w) where h * patch_size = H and w * patch_size = W
        """

    @abc.abstractmethod
    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tokenized image into an image tensor.
        """


class BSQ(torch.nn.Module):
    def __init__(self, codebook_bits: int, embedding_dim: int):
        super().__init__()

        # Store codebook_bits for later use
        self._codebook_bits = codebook_bits

        # Define the linear layers for down-projection and up-projection with residual connections
        self.linear_down = torch.nn.Linear(embedding_dim, codebook_bits)
        self.linear_up = torch.nn.Linear(codebook_bits, embedding_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ encoder:
        - A linear down-projection into codebook_bits dimensions
        - L2 normalization
        - differentiable sign
        """
        encoded_output = self.linear_down(x)
        # Normalize to unit length to get better quantization
        # use default p=2 for L2 norm
        # dim=-1 to normalize the last dimension
        # use default eps=1e-12 to avoid division by zero
        encoded_output_norm = torch.nn.functional.normalize(input=encoded_output, p=2, dim=-1)
        return diff_sign(encoded_output_norm)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Implement the BSQ decoder:
        - A linear up-projection into embedding_dim should suffice
        """
        decoded_output = self.linear_up(x)
        return decoded_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run BQS and encode the input tensor x into a set of integer tokens
        """
        return self._code_to_index(self.encode(x))

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        return self.decode(self._index_to_code(x))

    def _code_to_index(self, x: torch.Tensor) -> torch.Tensor:
        x = (x >= 0).int()
        return (x * 2 ** torch.arange(x.size(-1)).to(x.device)).sum(dim=-1)

    def _index_to_code(self, x: torch.Tensor) -> torch.Tensor:
        return 2 * ((x[..., None] & (2 ** torch.arange(self._codebook_bits).to(x.device))) > 0).float() - 1


class BSQPatchAutoEncoder(PatchAutoEncoder, Tokenizer):
    """
    Combine your PatchAutoEncoder with BSQ to form a Tokenizer.

    Hint: The hyper-parameters below should work fine, no need to change them
          Changing the patch-size of codebook-size will complicate later parts of the assignment.
    """

    def __init__(self, patch_size: int = 5, latent_dim: int = 128, codebook_bits: int = 10):
        super().__init__(patch_size=patch_size, latent_dim=latent_dim)
        
        # Print out the parameters for easy debugging.
        print(f"BSQPatchAutoEncoder(patch_size={patch_size}, latent_dim={latent_dim}, codebook_bits={codebook_bits})")
        
        # Add a BSQ layer after the encoder and before the decoder
        self.bsq = BSQ(codebook_bits=codebook_bits, embedding_dim=latent_dim)

    def encode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Run Auto Encoder and BQS, then encode the input tensor x into a set of integer tokens
        """
        code = self.encode(x)
        return self.bsq._code_to_index(code)

    def decode_index(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a set of integer tokens into an image.
        """
        code = self.bsq._index_to_code(x)
        return self.decode(code)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # run the auto-encoder's encode method
        auto_encoded_output = super().encode(x)
        #print("Shape of auto_encoded_output:", auto_encoded_output.shape)
        bsq_encoded_output = self.bsq.encode(auto_encoded_output)
        return bsq_encoded_output

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        bsq_decoded_output = self.bsq.decode(x)
        auto_decoded_output = self.decoder(bsq_decoded_output)
        return auto_decoded_output

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        Hint: It can be helpful to monitor the codebook usage with

              cnt = torch.bincount(self.encode_index(x).flatten(), minlength=2**self.codebook_bits)

              and returning

              {
                "cb0": (cnt == 0).float().mean().detach(),
                "cb2": (cnt <= 2).float().mean().detach(),
                ...
              }
        """
        encoded_index = self.encode_index(x)
        num_codes = 2 ** self.bsq._codebook_bits
        cnt = torch.bincount(encoded_index.flatten(), minlength=num_codes)
        metrics_dictionary = {
            "cb0": (cnt == 0).float().mean().detach(),
            "cb2": (cnt <= 2).float().mean().detach()
        }
        return self.decode_index(encoded_index), metrics_dictionary
