import abc

import torch

# Copilot was used to assist and learn the concepts for this implementation.
def load() -> torch.nn.Module:
    from pathlib import Path

    model_name = "PatchAutoEncoder"
    model_path = Path(__file__).parent / f"{model_name}.pth"
    print(f"Loading {model_name} from {model_path}")
    return torch.load(model_path, weights_only=False)


def hwc_to_chw(x: torch.Tensor) -> torch.Tensor:
    """
    Convert an arbitrary tensor from (H, W, C) to (C, H, W) format.
    This allows us to switch from trnasformer-style channel-last to pytorch-style channel-first
    images. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-1]] + [dims[-3]] + [dims[-2]]
    return x.permute(*dims)


def chw_to_hwc(x: torch.Tensor) -> torch.Tensor:
    """
    The opposite of hwc_to_chw. Works with or without the batch dimension.
    """
    dims = list(range(x.dim()))
    dims = dims[:-3] + [dims[-2]] + [dims[-1]] + [dims[-3]]
    return x.permute(*dims)


class PatchifyLinear(torch.nn.Module):
    """
    Takes an image tensor of the shape (B, H, W, 3) and patchifies it into
    an embedding tensor of the shape (B, H//patch_size, W//patch_size, latent_dim).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        # use a convolution to patchify the image:: in_channels: 3 (RGB) -> out_channels: latent_dim
        # no bias, as the model can learn a bias in later layers if needed.
        self.patch_conv = torch.nn.Conv2d(in_channels=3, out_channels=latent_dim, kernel_size=patch_size, stride=patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, H, W, 3) an image tensor dtype=float normalized to -1 ... 1

        return: (B, H//patch_size, W//patch_size, latent_dim) a patchified embedding tensor
        """
        return chw_to_hwc(self.patch_conv(hwc_to_chw(x)))


class UnpatchifyLinear(torch.nn.Module):
    """
    Takes an embedding tensor of the shape (B, w, h, latent_dim) and reconstructs
    an image tensor of the shape (B, w * patch_size, h * patch_size, 3).
    It applies a linear transformation to each input patch

    Feel free to use this directly, or as an inspiration for how to use conv the the inputs given.
    """

    def __init__(self, patch_size: int = 25, latent_dim: int = 128):
        super().__init__()
        # use transposed convolution to unpatchify the image:: in_channels = latent_dim -> out_channels = 3 (RGB)
        # No bias, as the model can learn a bias in later layers if needed.
        self.unpatch_conv = torch.nn.ConvTranspose2d(latent_dim, 3, patch_size, patch_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, w, h, latent_dim) an embedding tensor

        return: (B, H * patch_size, W * patch_size, 3) a image tensor
        """
        return chw_to_hwc(self.unpatch_conv(hwc_to_chw(x)))


class PatchAutoEncoderBase(abc.ABC):
    @abc.abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode an input image x (B, H, W, 3) into a tensor (B, h, w, bottleneck),
        where h = H // patch_size, w = W // patch_size and bottleneck is the size of the
        AutoEncoders bottleneck.
        """

    @abc.abstractmethod
    def decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Decode a tensor x (B, h, w, bottleneck) into an image (B, H, W, 3),
        We will train the auto-encoder such that decode(encode(x)) ~= x.
        """


class PatchAutoEncoder(torch.nn.Module, PatchAutoEncoderBase):
    """
    Implement a PatchLevel AutoEncoder

    Hint: Convolutions work well enough, no need to use a transformer unless you really want.
    Hint: See PatchifyLinear and UnpatchifyLinear for how to use convolutions with the input and
          output dimensions given.
    Hint: You can get away with 3 layers or less.
    Hint: Many architectures work here (even a just PatchifyLinear / UnpatchifyLinear).
          However, later parts of the assignment require both non-linearities (i.e. GeLU) and
          interactions (i.e. convolutions) between patches.
    """

    class PatchEncoder(torch.nn.Module):
        """
        (Optionally) Use this class to implement an encoder.
                     It can make later parts of the homework easier (reusable components).
        """

        # latent_dim is the dimension of the PatchifyLinear output
        # bottleneck is the dimension of the encoder output
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int, kernel_size: int):
            super().__init__()
            # patchify layer
            self.patchify = PatchifyLinear(patch_size, latent_dim)

            # optional layer for decoder bottleneck architecture
            padding = (kernel_size - 1) // 2
            hidden_dim = latent_dim * 2

            # added dropout improved the difference between train and val loss.
            # recommended that GroupNorm should not be used with bias in convolution layers.
            self.conv_layer = torch.nn.Sequential(
                # first convolution layer to process patches to learn interactions; learn richer features with hidden_dim
                torch.nn.Conv2d(in_channels=latent_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding),
                torch.nn.GroupNorm(1, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout2d(0.1),
                # refine learned features from first layer hidden_dim output
                torch.nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=kernel_size, padding=padding),
                torch.nn.GroupNorm(1, hidden_dim),
                torch.nn.GELU(),
                torch.nn.Dropout2d(0.1),
                # project to bottleneck with 1x1 convolution to reduce spatial interactions
                torch.nn.Conv2d(in_channels=hidden_dim, out_channels=bottleneck, kernel_size=1, padding=0),
                torch.nn.GroupNorm(1, bottleneck)
            )

            self.skip = torch.nn.Sequential(
                torch.nn.Conv2d(latent_dim, bottleneck, kernel_size=1, padding=0),
                torch.nn.GroupNorm(1, bottleneck)
            )

            self.gelu = torch.nn.GELU()

            """
            self.conv_layer = torch.nn.Sequential(
                # first convolution layer to process patches to learn interactions
                torch.nn.Conv2d(latent_dim, latent_dim, kernel_size=kernel_size, padding=padding, bias=False),
                torch.nn.GroupNorm(1, latent_dim),
                torch.nn.GELU(),
                # project to bottleneck with 1x1 convolution to reduce spatial interactions
                torch.nn.Conv2d(latent_dim, bottleneck, kernel_size=1, padding=0, bias=False),
                torch.nn.GroupNorm(1, bottleneck),
                torch.nn.GELU(),
            )
            """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = self.patchify(x) # Patch x, Shape (B, h, w, latent_dim)
            x = hwc_to_chw(x) # Shape (B, latent_dim, h, w)
            x_conv = self.conv_layer(x) # Shape (B, bottleneck, h, w)
            x_skip = self.skip(x)
            encoded_output = chw_to_hwc(x_conv + x_skip) # Shape (B, h, w, bottleneck)
            return encoded_output

    class PatchDecoder(torch.nn.Module):
        def __init__(self, patch_size: int, latent_dim: int, bottleneck: int, kernel_size: int):
            super().__init__()
            # unpatchify layer
            self.unpatchify = UnpatchifyLinear(patch_size, latent_dim)

            # optional layer for decoder bottleneck architecture
            padding = (kernel_size - 1) // 2
            hidden_dim = latent_dim * 2

            # recommended that GroupNorm should not be used with bias in convolution layers.
            self.convtranspose_layer = torch.nn.Sequential(
                # first transpose convolution layer to process patches to learn interactions; learn richer features with hidden_dim
                torch.nn.ConvTranspose2d(bottleneck, hidden_dim, kernel_size=1, padding=0),
                torch.nn.GroupNorm(1, hidden_dim),
                torch.nn.GELU(),
                # refine learned features from first layer hidden_dim output
                torch.nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=kernel_size, padding=padding),
                torch.nn.GroupNorm(1, hidden_dim),
                torch.nn.GELU(),
                # project to latent_dim with final transpose convolution
                torch.nn.ConvTranspose2d(hidden_dim, latent_dim, kernel_size=kernel_size, padding=padding),
                torch.nn.GroupNorm(1, latent_dim),
            )

            self.skip = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(bottleneck, latent_dim, kernel_size=1, padding=0),
                torch.nn.GroupNorm(1, latent_dim)
            )

            self.gelu = torch.nn.GELU()
            
            """
            self.convtranspose_layer = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(bottleneck, latent_dim, kernel_size=1, padding=0, bias=False),
                torch.nn.GroupNorm(1, latent_dim),
                torch.nn.GELU(),
                torch.nn.ConvTranspose2d(latent_dim, latent_dim, kernel_size=kernel_size, padding=padding, bias=False),
                torch.nn.GroupNorm(1, latent_dim),
                torch.nn.GELU()
            )
            """

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = hwc_to_chw(x)
            x_conv = self.convtranspose_layer(x)
            x_skip = self.skip(x)
            decoded_patches = chw_to_hwc(self.gelu(x_conv + x_skip)) # Shape (B, h, w, latent_dim)
            reconstructed_image = self.unpatchify(decoded_patches) # Shape (B, H, W, 3)
            return reconstructed_image

    def __init__(self, patch_size: int = 25, latent_dim: int = 128, bottleneck: int = 128):
        super().__init__()
        # Print out the parameters for easy debugging.
        print(f"PatchAutoEncoder(patch_size={patch_size}, latent_dim={latent_dim}, bottleneck={bottleneck})")

        # Define hyperparameters for convolution layer for encoder and decoder
        # With patch_size=25 a kernel_size of 3-5 provided low difference between train and val.  When testing with 11, the model underfitted.  
        kernel_size = 5

        self.encoder = self.PatchEncoder(patch_size, latent_dim, bottleneck, kernel_size)
        self.decoder = self.PatchDecoder(patch_size, latent_dim, bottleneck, kernel_size)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Return the reconstructed image and a dictionary of additional loss terms you would like to
        minimize (or even just visualize).
        You can return an empty dictionary if you don't have any additional terms.
        """
        return self.decode(self.encode(x)), {}

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        #print("Called PatchAutoEncoder.encode()")
        #print(f"Input shape: {x.shape}")
        return self.encoder(x)

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)
