import torch

class MinMaxScaler:
    def __init__(self, scaler_file_path=None, x_values_orig=None, y_values_orig=None, device='cpu'):
        self._eps = 1e-5
        self.device = device

        if scaler_file_path is None:
            self.min_x, self.max_x = self._minmax_calc(x_values_orig)
            self.min_y, self.max_y = self._minmax_calc(y_values_orig)
        else:
            self._load_scalers(scaler_file_path)

    def to(self, device):
        """
        Move the scaler to the specified device.
        Args:
            device: Device to move the scaler to
        """
        self.device = device
        self.min_x = self.min_x.to(device)
        self.max_x = self.max_x.to(device)
        self.min_y = self.min_y.to(device)
        self.max_y = self.max_y.to(device)

    def min_max_scale_io(self, x_in, y_in):
        """
        Min-Max scaling for input data.
        Args:
            x_in: Input tensor
            y_in: Output tensor
        Returns:
            Scaled input and output tensors
        """
        x_scaled = self._min_max_scale(x_in, self.min_x, self.max_x)
        y_scaled = self._min_max_scale(y_in, self.min_y, self.max_y)

        return x_scaled, y_scaled

    def min_max_unscale_io(self, x_scaled, y_scaled):
        """
        Unscale input and output data.
        Args:
            x_scaled: Scaled input tensor
            y_scaled: Scaled output tensor
        returns:
            Unscaled input and output tensors
        """
        x_unscaled = self._min_max_unscale(x_scaled, self.min_x, self.max_x)
        y_unscaled = self._min_max_unscale(y_scaled, self.min_y, self.max_y)

        return x_unscaled, y_unscaled

    def scale_input(self, tensor):
        """
        Scale input data.
        Args:
            tensor: Input tensor
        Returns:
            Scaled tensor
        """
        x_scaled = self._min_max_scale(tensor, self.min_x, self.max_x)
        return x_scaled

    def unscale_output(self, tensor):
        """
        Unscale output data.
        Args:
            tensor: Scaled output tensor
        Returns:
            Unscaled tensor
        """
        y_unscaled = self._min_max_unscale(tensor, self.min_y, self.max_y)
        return y_unscaled


    def _load_scalers(self, file_path=None):
        self.min_x = torch.load(file_path + f"/min_x.pth", map_location=self.device)
        self.max_x = torch.load(file_path + f"/max_x.pth", map_location=self.device)
        self.min_y = torch.load(file_path + f"/min_y.pth", map_location=self.device)
        self.max_y = torch.load(file_path + f"/max_y.pth", map_location=self.device)

    def save_scalers(self, file_path=None):
        torch.save(self.min_x, file_path + f"/min_x.pth")
        torch.save(self.max_x, file_path + f"/max_x.pth")
        torch.save(self.min_y, file_path + f"/min_y.pth")
        torch.save(self.max_y, file_path + f"/max_y.pth")

    def _minmax_calc(self, tensor):
        """
        Min-Max scaling for input data.
        Args:
            tensor: Input tensor
        Returns:
            Scaled tensor
        """

        min_val, _ = torch.min(tensor, dim=0, keepdim=True)
        max_val, _ = torch.max(tensor, dim=0, keepdim=True)
        max_val = max_val + self._eps

        return min_val.to(self.device), max_val.to(self.device)

    @staticmethod
    def _min_max_scale(tensor, min_val, max_val):
        """Min-Max scaling for input data."""
        scaled_tensor = (tensor - min_val) / (max_val - min_val)

        return scaled_tensor

    @staticmethod
    def _min_max_unscale(scaled_tensor, min_val, max_val):
        """Unscale the tensor."""
        unscaled_tensor = scaled_tensor * (max_val - min_val) + min_val
        return unscaled_tensor