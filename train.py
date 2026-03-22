import os, json, torch
from torch.utils.data import DataLoader
from transformers import BartTokenizer, AdamW
import transformers.models.bart.modeling_bart as bart_modeling

_original_expand_mask = bart_modeling._expand_mask
def _patched_expand_mask(mask, dtype, tgt_len=None):
    return mask if mask.dim() == 4 else _original_expand_mask(mask, dtype, tgt_len)
bart_modeling._expand_mask = _patched_expand_mask

from config import Config
from models.dataloader_extract import HotelExtractDataset
from models.model import SentimentClassifier
from utils.utils_extract import Log, train_epoch, eval_model, EarlyStopping

def main():
    if not os.path.exists(Config.SAVE_DIR): os.makedirs(Config.SAVE_DIR)
    logger = Log(Config.SAVE_DIR, "mate_run").get_logger()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    with open(Config.TRAIN_JSON) as f: train_data = json.load(f)
    with open(Config.TEST_JSON) as f: test_data = json.load(f)

    train_ds = HotelExtractDataset(train_data, tokenizer, Config.MAX_LEN, Config.IMG_DIR)
    test_ds = HotelExtractDataset(test_data, tokenizer, Config.MAX_LEN, Config.IMG_DIR)
    
    train_loader = DataLoader(train_ds, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    test_loader = DataLoader(test_ds, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = SentimentClassifier(Config, tokenizer).to(device)
    optimizer = AdamW(model.parameters(), lr=Config.LEARNING_RATE)
    early_stopping = EarlyStopping(patience=7, path=Config.CHECKPOINT_PATH, trace_func=logger.info)

    # Training
    for epoch in range(Config.EPOCHS):
        train_acc, train_loss = train_epoch(model, train_loader, optimizer, device)
        val_f1, val_loss = eval_model(model, test_loader, device)
        logger.info(f"Epoch {epoch+1} | Loss: {train_loss:.4f} | Val Macro F1: {val_f1:.4f}")
        
        early_stopping(val_loss, model, epoch, optimizer)
        if early_stopping.early_stop: break

if __name__ == "__main__": main()
