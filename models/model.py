import torch
import torch.nn as nn
from transformers import BartModel
import torchvision.models as models

class SimpleOutput:
    def __init__(self, loss, logits):
        self.loss = loss
        self.logits = logits

class SentimentClassifier(nn.Module):
    def __init__(self, args, tokenizer):
        super(SentimentClassifier, self).__init__()
        self.args = args
        
        self.bart = BartModel.from_pretrained("facebook/bart-base")
        
        resnet = models.resnet50(pretrained=True)
        self.resnet = nn.Sequential(*list(resnet.children())[:-1]) 
        
        self.visual_proj = nn.Linear(2048, 768)
        self.dropout = nn.Dropout(0.1)

        self.classifier = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(768, 7)
        )

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels=None, images=None, vis_mat=None):
        
        # 1. ENCODER (Text + Graph via Visible Matrix)
        if len(attention_mask.shape) == 3:
            extended_attention_mask = attention_mask.unsqueeze(1) 
        else:
            extended_attention_mask = attention_mask

        enc_out = self.bart.encoder(
            input_ids=input_ids, 
            attention_mask=extended_attention_mask
        )
        text_graph_feats = enc_out.last_hidden_state 

        # 2. VISUAL BRANCH
        img_feats = self.resnet(images).reshape(images.size(0), -1)
        img_embeds = self.visual_proj(img_feats).unsqueeze(1)
        img_embeds = self.dropout(img_embeds)

        # 3. FUSION
        fused_embeds = torch.cat((img_embeds, text_graph_feats), dim=1)
        
        pad_token_id = self.bart.config.pad_token_id
        text_padding_mask = (input_ids != pad_token_id).long() 
        img_mask = torch.ones(images.size(0), 1, device=images.device).long()
        fused_mask = torch.cat((img_mask, text_padding_mask), dim=1) 

        # 4. DECODER
        dec_out = self.bart.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=fused_embeds,
            encoder_attention_mask=fused_mask
        )
        dec_feats = dec_out.last_hidden_state

        # 5. CLASSIFY
        h_mask = torch.mean(dec_feats, dim=1) 
        logits = self.classifier(h_mask)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits, labels)

        return SimpleOutput(loss=loss, logits=logits)
