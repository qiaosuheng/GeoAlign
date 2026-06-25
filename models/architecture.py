import torch
import torch.nn as nn
import torch.nn.functional as F



class AttentionPooling(nn.Module):
    """
      (Float32 ) +  
    """

    def __init__(self, in_dim):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(in_dim, in_dim // 2),
            nn.Tanh(),
            nn.Linear(in_dim // 2, 1)
        )

        self.attn_softmax = nn.Softmax(dim=1)

    def forward(self, x, mask=None):
        # x: [Batch, Seq, Dim] (BF16)


        x_fp32 = x.float()
        w = self.attention(x_fp32)  # [Batch, Seq, 1]

        if mask is not None:
            mask_expanded = mask.unsqueeze(-1)

            w = w.masked_fill(mask_expanded == 0, -1e9)


        alpha = self.attn_softmax(w)

        x_pooled = torch.sum(x_fp32 * alpha, dim=1)


        return x_pooled.to(dtype=x.dtype)


class MultiSampleDropoutHead(nn.Module):
    """
     : Multi-Sample Dropout Head
     4  Dropout Mask  Logits 
     (Ensemble)
    """

    def __init__(self, in_dim, num_classes, dropout_rate=0.15, num_samples=4):
        super().__init__()
        self.num_samples = num_samples

        self.dropouts = nn.ModuleList([nn.Dropout(dropout_rate) for _ in range(num_samples)])
        self.linear = nn.Linear(in_dim, num_classes)

    def forward(self, x):
        # x: [Batch, Dim]


        if self.training and self.num_samples > 1:
            logits_list = []
            for dropout_op in self.dropouts:

                logits_list.append(self.linear(dropout_op(x)))

            return torch.stack(logits_list, dim=0).mean(dim=0)


        else:
            return self.linear(x)



class GenericBioSiamese(nn.Module):
    def __init__(self, base_model, model_type, num_classes=2):
        super().__init__()
        self.model_type = model_type.lower()
        self.base_model = base_model
        print(f" [Siamese] : {self.model_type}")


        if self.model_type == "hyenadna":
            config = self.base_model.config
            self.hidden_dim = config.d_model if hasattr(config, 'd_model') else config.hidden_size
        elif self.model_type == "ntv3":

            config = self.base_model.config
            self.hidden_dim = config.embed_dim  # 256
        elif self.model_type == "caduceus":

            config = self.base_model.config
            self.hidden_dim = config.d_model  # 256
        else:
            raise ValueError(f" : {self.model_type}")





        head_dtype = torch.float32 if self.model_type in ["ntv3", "caduceus"] else torch.bfloat16
        self.norm = nn.LayerNorm(self.hidden_dim).to(dtype=head_dtype)
        self.pooler = AttentionPooling(self.hidden_dim).to(dtype=head_dtype)
        self.head = MultiSampleDropoutHead(
            in_dim=self.hidden_dim, num_classes=num_classes, dropout_rate=0.20, num_samples=4
        ).to(dtype=head_dtype)


    def _extract_features(self, input_ids):
        """
         
        """
        if self.model_type == "hyenadna":
            outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            return outputs.last_hidden_state

        elif self.model_type == "ntv3":
            # =====================================================================

            # =====================================================================

            NUM_CONV_LAYERS = 1
            DOWNSAMPLE = [False] * NUM_CONV_LAYERS
            # =====================================================================

            core = self.base_model.base_model.model.core


            x = core.embed_layer(input_ids)
            x = x.to(torch.float32)

            x = core.stem(x.permute(0, 2, 1))

            for i in range(NUM_CONV_LAYERS):
                block = core.conv_tower_blocks[i]
                x = block.conv(x)
                x = block.res_conv(x)  # Residual: x + ConvBlock(K=1)
                if DOWNSAMPLE[i]:
                    x = block.avg_pool(x)

            x = x.permute(0, 2, 1)  # [B, L', 256]
            for layer in core.transformer_blocks:
                out = layer(x)
                x = out["embeddings"]

            return x  # [B, L', 256]

        elif self.model_type == "caduceus":
            outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.last_hidden_state

            if hidden_states.shape[-1] == self.hidden_dim * 2:
                hidden_states = hidden_states[..., :self.hidden_dim]
            return hidden_states  # [B, L, d_model]

    def forward_one_branch(self, input_ids, attention_mask):

        x = self._extract_features(input_ids)
        x = self.norm(x)


        pooled = self.pooler(x, attention_mask)
        logits = self.head(pooled)
        return logits, pooled

    def forward(self, input_ids, input_ids_pair=None, attention_mask=None, attention_mask_pair=None,
                return_feats=False):
        if attention_mask is None: attention_mask = torch.ones_like(input_ids)

        logits_anchor, feat_anchor = self.forward_one_branch(input_ids, attention_mask)

        if input_ids_pair is None:
            return logits_anchor

        if attention_mask_pair is None: attention_mask_pair = torch.ones_like(input_ids_pair)
        logits_pair, feat_pair = self.forward_one_branch(input_ids_pair, attention_mask_pair)

        if return_feats:
            return logits_anchor, logits_pair, feat_anchor, feat_pair
        return (logits_anchor + logits_pair) / 2.0


class GenericVanillaBaseline(nn.Module):
    """
     
    """

    def __init__(self, base_model, model_type, num_classes=2):
        super().__init__()
        self.model_type = model_type.lower()
        self.base_model = base_model


        if self.model_type == "hyenadna":
            config = self.base_model.config
            self.hidden_dim = config.d_model if hasattr(config, 'd_model') else config.hidden_size
        elif self.model_type == "ntv3":

            config = self.base_model.config
            self.hidden_dim = config.embed_dim  # 256
        elif self.model_type == "caduceus":

            config = self.base_model.config
            self.hidden_dim = config.d_model  # 256
        else:
            raise ValueError(f" : {self.model_type}")


        head_dtype = torch.float32 if self.model_type in ["ntv3", "caduceus"] else torch.bfloat16
        self.norm = nn.LayerNorm(self.hidden_dim).to(dtype=head_dtype)
        self.pooler = AttentionPooling(self.hidden_dim).to(dtype=head_dtype)
        self.head = MultiSampleDropoutHead(
            in_dim=self.hidden_dim, num_classes=num_classes, dropout_rate=0.20, num_samples=4
        ).to(dtype=head_dtype)

    def _extract_features(self, input_ids):

        if self.model_type == "hyenadna":
            outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            return outputs.last_hidden_state

        elif self.model_type == "ntv3":
            # =====================================================================

            # =====================================================================
            NUM_CONV_LAYERS = 1
            DOWNSAMPLE = [False] * NUM_CONV_LAYERS
            # =====================================================================

            core = self.base_model.base_model.model.core


            x = core.embed_layer(input_ids)
            x = x.to(torch.float32)


            x = core.stem(x.permute(0, 2, 1))


            for i in range(NUM_CONV_LAYERS):
                block = core.conv_tower_blocks[i]
                x = block.conv(x)
                x = block.res_conv(x)
                if DOWNSAMPLE[i]:
                    x = block.avg_pool(x)


            x = x.permute(0, 2, 1)  # [B, L', 256]
            for layer in core.transformer_blocks:
                out = layer(x)
                x = out["embeddings"]

            return x  # [B, L', 256]

        elif self.model_type == "caduceus":

            outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.last_hidden_state

            if hidden_states.shape[-1] == self.hidden_dim * 2:
                hidden_states = hidden_states[..., :self.hidden_dim]
            return hidden_states  # [B, L, d_model]


    def forward(self, input_ids, attention_mask=None, **kwargs):
        if attention_mask is None: attention_mask = torch.ones_like(input_ids)
        x = self._extract_features(input_ids)
        x = self.norm(x)
        pooled = self.pooler(x, attention_mask)
        return self.head(pooled)


class GenericGenerativeLM(nn.Module):
    """
      ( TAPT )
     HyenaDNA ( Causal LM)  NTv3 ( Masked LM)
    """

    def __init__(self, base_model, model_type):
        super().__init__()
        self.model_type = model_type.lower()
        self.base_model = base_model


        mode_str = "Masked LM" if self.model_type in ["ntv3", "caduceus"] else "Causal LM"
        print(f" [Architecture]  ({mode_str}) | : {self.model_type.upper()}")

    def forward(self, input_ids, attention_mask=None, **kwargs):
        if self.model_type == "hyenadna":

            outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            x = outputs.last_hidden_state

            embed_weight = self.base_model.get_input_embeddings().weight
            logits = F.linear(x, embed_weight)
            return logits

        elif self.model_type == "ntv3":


            outputs = self.base_model(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)

            if hasattr(outputs, "logits"):
                logits = outputs.logits
            else:

                x = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs.hidden_states[-1]
                embed_weight = self.base_model.get_input_embeddings().weight
                logits = F.linear(x, embed_weight)

            return logits

        elif self.model_type == "caduceus":


            outputs = self.base_model(input_ids=input_ids, output_hidden_states=True, return_dict=True)
            hidden_states = outputs.last_hidden_state

            d_model = self.base_model.config.d_model
            if hidden_states.shape[-1] == d_model * 2:
                hidden_states = hidden_states[..., :d_model]
            embed_weight = self.base_model.get_input_embeddings().weight
            logits = F.linear(hidden_states, embed_weight)
            return logits