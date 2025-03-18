'''Use with GATR version
import torch
from gatr import GATr, SelfAttentionConfig, MLPConfig
from gatr.interface import embed_point, extract_scalar,embed_scalar


class GA_transformer(torch.nn.Module):
    """Example wrapper around a GATr model.
    
    Expects input data that consists of a point cloud: one 3D point for each item in the data.
    Returns outputs that consists of one scalar number for the whole dataset.
    
    Parameters
    ----------
    blocks : int
        Number of transformer blocks
    hidden_mv_channels : int
        Number of hidden multivector channels
    hidden_s_channels : int
        Number of hidden scalar channels
    """

    def __init__(self, blocks=4, hidden_mv_channels=16, hidden_s_channels=32,in_s_channels = None, out_s_channels = None ):
        super().__init__()
        self.gatr = GATr(
            in_mv_channels=1,
            out_mv_channels=1,
            hidden_mv_channels=hidden_mv_channels,
            in_s_channels=in_s_channels,
            out_s_channels=out_s_channels,
            hidden_s_channels=hidden_s_channels,
            num_blocks=blocks,
            attention=SelfAttentionConfig(),  # Use default parameters for attention
            mlp=MLPConfig(),  # Use default parameters for MLP
        )
        
    def forward(self, inputs,scalars,prev_ga_scalar=None):
        """Forward pass.
        
        Parameters
        ----------
        inputs : torch.Tensor with shape (*batch_dimensions, num_points, 3)
            Point cloud input data
        
        Returns
        -------
        outputs : torch.Tensor with shape (*batch_dimensions, 1)
            Model prediction: a single scalar for the whole point cloud.
        """
        
        # Embed point cloud in PGA
        embedded_inputs = embed_point(inputs).unsqueeze(-2)  # (..., num_points, 1, 16)

        if prev_ga_scalar is not None:
            embedded_scalar_input = embed_scalar(prev_ga_scalar)
            #the squeeze and unsqueeze is necessary because:
            # [batch_size,num_points,1,16] ->  [batch_size,num_points,16] -> [batch_size,num_points,1,16]
            embedded_inputs = (embedded_inputs.squeeze(-2) + embedded_scalar_input).unsqueeze(-2)

        
        # Pass data through GATr
        # print("embedded_inputs shape: ", embedded_inputs.shape)
        # print("scalars shape: ", scalars.shape)
        embedded_outputs, embedded_scalars = self.gatr(embedded_inputs, scalars=scalars)  # (..., num_points, 1, 16)


        #OLD VERSION
        # Extract scalar and aggregate outputs from point cloud
        nodewise_outputs = extract_scalar(embedded_outputs)  # (..., num_points, 1, 1)
        outputs = torch.mean(nodewise_outputs, dim=(-3, -2))  # (..., 1)

        return outputs,embedded_outputs,embedded_scalars,nodewise_outputs.squeeze(-2)
    
        #NEW VERSION TODO understand when I need to use

        embedded_outputs_squeezed =  embedded_outputs.squeeze(-2)
        return embedded_outputs_squeezed
        
        # return outputs
'''