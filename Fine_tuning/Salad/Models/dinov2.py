import torch
import torch.nn as nn

DINOV2_ARCHS = {
    'dinov2_vits14': 384,
    'dinov2_vitb14': 768,
    'dinov2_vitl14': 1024,
    'dinov2_vitg14': 1536,
}


class DINOv2(nn.Module):
    """
    DINOv2 model

    Args:
        model_name (str): The name of the model architecture
            should be one of ('dinov2_vits14', 'dinov2_vitb14', 'dinov2_vitl14', 'dinov2_vitg14')
        num_trainable_blocks (int): The number of last blocks in the model that are trainable.
        norm_layer (bool): If True, a normalization layer is applied in the forward pass.
        return_token (bool): If True, the forward pass returns both the feature map and the token.
    """

    def __init__(
            self,
            model_name='dinov2_vitb14',
            num_trainable_blocks=2,
            norm_layer=False,
            return_token=False
    ):
        super().__init__()

        assert model_name in DINOV2_ARCHS.keys(), f'Unknown model name {model_name}'
        #self.model = torch.hub.load('facebookresearch/dinov2', model_name)

        self.model = torch.hub.load('./Models/dinov2', model_name, source='local')

        #uncomment to integrate new fine-tuned model to salad

        # Unfreeze last 3 transformer blocks (just like during training)
        for block in self.model.blocks[-3:]:
            for param in block.parameters():
                param.requires_grad = True

        weights_path="./Weights/finetuned_dinov2_v3.pth"
        #weights_path = "./Weights/finetuned_dinov2_triplet.pth" # normal 10 epochs
            
        # Load the fine-tuned weights
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))  # or 'cuda' if needed

        self.model.load_state_dict(state_dict, strict=False)


        """
        #Load your own weights
        checkpoint_path = './path/to/your_weights.pth'  # update this path
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # Optional: If your checkpoint is a dict with 'model' key (common in training checkpoints)
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        # Step 3: Load the state dict into the model
        model.load_state_dict(state_dict, strict=False)  # strict=False allows partial loading if needed
        self.model = model
        """

        self.num_channels = DINOV2_ARCHS[model_name]
        self.num_trainable_blocks = num_trainable_blocks
        self.norm_layer = norm_layer
        self.return_token = return_token

    def forward(self, x):
        """
        The forward method for the DINOv2 class

        Parameters:
            x (torch.Tensor): The input tensor [B, 3, H, W]. H and W should be divisible by 14.

        Returns:
            f (torch.Tensor): The feature map [B, C, H // 14, W // 14].
            t (torch.Tensor): The token [B, C]. This is only returned if return_token is True.
        """

        B, C, H, W = x.shape

        x = self.model.prepare_tokens_with_masks(x)

        # First blocks are frozen
        with torch.no_grad():
            for blk in self.model.blocks[:-self.num_trainable_blocks]:
                x = blk(x)
        x = x.detach()

        # Last blocks are trained
        for blk in self.model.blocks[-self.num_trainable_blocks:]:
            x = blk(x)

        if self.norm_layer:
            x = self.model.norm(x)

        t = x[:, 0]
        f = x[:, 1:]

        # Reshape to (B, C, H, W)
        f = f.reshape((B, H // 14, W // 14, self.num_channels)).permute(0, 3, 1, 2)

        if self.return_token:
            return f, t
        return f

    def forward_last_3_layers(self, x):
        """
        Forward pass that returns a 4D feature map constructed from the patch embeddings
        of the last 3 transformer layers, optimized for inference.

        Parameters:
            x (torch.Tensor): Input tensor of shape [B, 3, H, W]

        Returns:
            torch.Tensor: Feature map of shape [B, C', H//14, W//14], where C' = 3 * hidden_dim
        """
        B, C, H, W = x.shape

        with torch.no_grad():  # Optional: remove if training with this
            outputs = self.model.get_intermediate_layers(
                x, n=3, reshape=False, return_class_token=False, norm=True
            )

        # Concatenate [B, N, D] x 3 → [B, N, 3D]
        x_cat = torch.cat(outputs, dim=-1)

        # Reshape to [B, 3D, H//14, W//14]
        f = x_cat.reshape(B, H // 14, W // 14, -1).permute(0, 3, 1, 2).contiguous()

        return f

    def extract_features_4d(self, image_tensor):
        """
        Extract features using get_intermediate_layers, and reshape to 4D tensor [B, C, H, W].
        This output is suitable for use with Conv2D-based aggregators.
        """
        features = self.model.get_intermediate_layers(image_tensor, n=3)

        # ⚠️ KEEP the CLS token to get 256 patches
        # features = [feat[:, 1:] for feat in features]  # ← this line removes CLS token

        features = torch.cat(features, dim=-1)  # [B, 256, total_feature_dim]
        B, N, D = features.shape

        H = W = int(N ** 0.5)
        assert H * W == N, f"Non-square patch grid: N={N}"

        features = features.permute(0, 2, 1).reshape(B, D, H, W)  # [B, D, H, W]
        features = torch.nn.functional.normalize(features, dim=1)
        return features