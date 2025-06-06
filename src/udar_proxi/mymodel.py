import torch
import torch.nn as nn


class EMConvModel(nn.Module):
    def __init__(self, input_shape, output_shape, nb_filter=40, num_nested_blocks=7):
        """
        PyTorch equivalent of the TensorFlow model.

        Args:
            input_shape (tuple): (seq_length,)
            output_shape (tuple): (seq_length, num_channels)
            nb_filter (int): Base number of filters
            num_nested_blocks (int): Number of convolutional blocks
        """
        super(EMConvModel, self).__init__()

        self.num_blocks = num_nested_blocks
        self.conv_layers = nn.ModuleList()

        # Create convolutional blocks
        for i in range(num_nested_blocks):
            self.conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_channels=input_shape[0] if i == 0 else nb_filter * i,
                              out_channels=nb_filter * (i + 1),
                              kernel_size=63,
                              padding="same"),
                    nn.ReLU(),
                    nn.Conv1d(in_channels=nb_filter * (i + 1),
                              out_channels=nb_filter * (i + 1),
                              kernel_size=63,
                              padding="same"),
                    nn.ReLU(),
                    nn.MaxPool1d(kernel_size=2, stride=2)
                )
            )

        self.upsample_conv = nn.ConvTranspose1d(in_channels=nb_filter*num_nested_blocks,
                                           out_channels=nb_filter*num_nested_blocks, kernel_size=output_shape[1])
        self.downsample_conv = nn.Conv1d(in_channels=nb_filter*num_nested_blocks,
                                         out_channels=output_shape[0], kernel_size=1)

        # # Bilinear Upsampling (PyTorch Equivalent)
        # self.upsample = nn.Upsample(size=output_shape, mode="linear", align_corners=True)
        #
        # # Final 1D convolution
        # self.final_conv = nn.Conv1d(in_channels=nb_filter * num_nested_blocks,
        #                             out_channels=output_shape[-1],
        #                             kernel_size=1,
        #                             padding="same")

    def invert_min_max_scale(self,scaled, min_val, max_val, feature_range=(-1,1)):
        """Invert the scaling of a tensor scaled with min_max_scale()."""
        a, b = feature_range

        return (scaled - a) / (b - a) * (max_val - min_val) + min_val

    # def minmax_scale(self, tensor, min_max_training, feature_range=(-1, 1)):
    #     """
    #     Min-Max scaling for input data.
    #     Args:
    #         x: Input tensor
    #         feature_range: Tuple of min and max values
    #     Returns:
    #         Scaled tensor
    #     """
    #
    #     min_val,max_val = min_max_training
    #
    #     scaled_tensor = (tensor - min_val) / (max_val - min_val)
    #
    #     # Scale to the [a, b] range
    #     a, b = feature_range
    #     scaled_tensor = scaled_tensor * (b - a) + a
    #
    #     return scaled_tensor

    def forward(self, x):
        """
        Forward pass of the model.
        Args:
            x: Input tensor of shape (batch, seq_length)
        Returns:
            Output tensor of shape (batch, output_seq_length, num_channels)
        """
        # x = self.minmax_scale(x, min_max_in)
        # x = x.unsqueeze(1)  # Add channel dimension: (batch, 1, seq_length)

        for block in self.conv_layers:
            a = block[0](x)  # First Conv
            a = block[1](a)  # First ReLU
            d = a.clone()  # Store residual
            a = block[2](a)  # Second Conv
            a = block[3](a)  # Second ReLU

            x = d + a  # Skip connection (Element-wise Add)
            # max_pool
            x = block[4](x)

        # 1. Grow the last dimension (width: 1 → m)
        x = self.upsample_conv(x)

        # 2. Reduce the first dimension (features: 280 → n)
        x = self.downsample_conv(x)

        return x

