import inspect
import torch.nn as nn
import models as models

class cAItomorph(nn.Module):
    def __init__(self, class_count, arch, embedding_dim=768, dropout_p=0.0):
        super(cAItomorph, self).__init__()

        if arch not in models.__dict__:
            raise ValueError(f"Unknown model architecture '{arch}'")
        model_cls = models.__dict__[arch]
        kwargs = dict(input_dim=embedding_dim, num_classes=class_count)
        # Thread the MC-dropout rate only to models that accept it (Mean/Max/ABMIL
        # add a head Dropout(dropout_p); dropout_p=0.0 -> identity, no weight change,
        # so existing checkpoints still load). Other archs ignore it.
        try:
            if "dropout_p" in inspect.signature(model_cls).parameters:
                kwargs["dropout_p"] = dropout_p
        except (TypeError, ValueError):
            pass
        self.model = model_cls(**kwargs)
 
    def forward(self, embeddings, return_latent = False):
        """
        Forward pass through the cAItomorph model.
        Accepts embeddings of shape:
        - [embedding_dim]
        - [sequence_len, embedding_dim]
        - [batch_size, sequence_len, embedding_dim]

        Returns:
        - logits if return_latent=False
        - (latent, logits) if return_latent=True
        """
        
        if embeddings.dim() == 1:
            embeddings = embeddings.unsqueeze(0).unsqueeze(0)  # [D] -> [1, 1, D]
        elif embeddings.dim() == 2:
            embeddings = embeddings.unsqueeze(0)  # [N, D] -> [1, N, D]
        elif embeddings.dim() != 3:
            raise ValueError(f"Unexpected embedding shape: {embeddings.shape}")

        latent, logits = self.model(embeddings)

        if return_latent:
            return latent, logits
        return logits



    def __repr__(self):
        return f"cAItomorph(model={self.model})"
