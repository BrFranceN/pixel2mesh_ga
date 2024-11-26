import torch
import torch.nn as nn
import torch.nn.functional as F

class HyperConv2D(nn.Module):
    def __init__(self, 
                 in_channels, 
                 out_channels, 
                 kernel_size, 
                 stride=1, 
                 padding=0, 
                 use_bias=True, 
                 activation=None, 
                 initializer=nn.init.xavier_normal_,
                 algebra=torch.tensor([[-1, 1, -1], [-1, -1, 1], [1, -1, -1]]) # Quaternion algebra
                ):
        super(HyperConv2D, self).__init__()
        
        if in_channels % 4 != 0:
            raise ValueError("The number of input channels must be divisible by 4.")
        
        self.in_channels = in_channels // 4
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.use_bias = use_bias
        self.activation = activation
        self.algebra = algebra
        
        # Define the kernels for each quaternion component
        self.kernel_r = nn.Parameter(torch.empty(out_channels, self.in_channels, kernel_size, kernel_size))
        self.kernel_i = nn.Parameter(torch.empty(out_channels, self.in_channels, kernel_size, kernel_size))
        self.kernel_j = nn.Parameter(torch.empty(out_channels, self.in_channels, kernel_size, kernel_size))
        self.kernel_k = nn.Parameter(torch.empty(out_channels, self.in_channels, kernel_size, kernel_size))
        
        # Initialize weights
        initializer(self.kernel_r)
        initializer(self.kernel_i)
        initializer(self.kernel_j)
        initializer(self.kernel_k)
        
        # Bias
        if self.use_bias:
            self.bias = nn.Parameter(torch.zeros(4 * out_channels))
        else:
            self.bias = None

    def forward(self, x):
        # Reshape input to split into quaternion components
        batch_size, height, width, channels = x.size()
        if channels % 4 != 0:
            raise ValueError("The number of input channels must be divisible by 4.")
        
        x_r, x_i, x_j, x_k = torch.chunk(x, chunks=4, dim=1)
        
        # Construct filter components using quaternion algebra
        F_r = torch.cat([
            self.kernel_r, 
            self.algebra[0, 0] * self.kernel_i, 
            self.algebra[1, 1] * self.kernel_j, 
            self.algebra[2, 2] * self.kernel_k
        ], dim=1)
        
        F_i = torch.cat([
            self.kernel_i, 
            self.kernel_r, 
            self.algebra[1, 2] * self.kernel_k, 
            self.algebra[2, 1] * self.kernel_j
        ], dim=1)
        
        F_j = torch.cat([
            self.kernel_j, 
            self.algebra[0, 2] * self.kernel_k, 
            self.kernel_r, 
            self.algebra[2, 0] * self.kernel_i
        ], dim=1)
        
        F_k = torch.cat([
            self.kernel_k, 
            self.algebra[0, 1] * self.kernel_j, 
            self.algebra[1, 0] * self.kernel_i, 
            self.kernel_r
        ], dim=1)
        
        # Perform convolution for each component
        y_r = F.conv2d(x, F_r, stride=self.stride, padding=self.padding)
        y_i = F.conv2d(x, F_i, stride=self.stride, padding=self.padding)
        y_j = F.conv2d(x, F_j, stride=self.stride, padding=self.padding)
        y_k = F.conv2d(x, F_k, stride=self.stride, padding=self.padding)
        
        # Concatenate the results
        outputs = torch.cat([y_r, y_i, y_j, y_k], dim=1)
        
        # Add bias if applicable
        if self.use_bias:
            outputs += self.bias.view(1, -1, 1, 1)
        
        # Apply activation if provided
        if self.activation:
            outputs = self.activation(outputs)
        
        return outputs


# Example input
batch_size = 8
height = 32
width = 32
channels = 8  # Must be divisible by 4
x = torch.randn(batch_size, channels, height, width)

print("input x: ",x.shape)

# Create layer
hyperconv = HyperConv2D(in_channels=8, out_channels=16, kernel_size=3, padding=1, activation=torch.relu)




# Forward pass
output = hyperconv(x)
print(output.shape)  # Expected output shape: (8, 64, 32, 32)