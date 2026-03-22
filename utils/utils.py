import os
import torch
import numpy as np
import logging
from tqdm import tqdm
from sklearn.metrics import classification_report, f1_score

class EarlyStopping:
    def __init__(self, patience=3, delta=0, path='checkpoint.pth', trace_func=print):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.trace_func = trace_func
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model, epoch, optimizer):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
        elif score < self.best_score + self.delta:
            self.counter += 1
            self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, epoch, optimizer)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, epoch, optimizer):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_score': self.best_score,
            'val_loss': val_loss
        }
        torch.save(checkpoint, self.path)
        self.trace_func(f'Validation Loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        self.val_loss_min = val_loss

class Log(object):
    def __init__(self, log_dir, name):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        if not os.path.exists(log_dir): os.makedirs(log_dir)
        formatter = logging.Formatter('%(asctime)s | %(message)s', "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(log_dir, name + '.log'))
        fh.setLevel(logging.INFO); fh.setFormatter(formatter)
        sh = logging.StreamHandler(); sh.setLevel(logging.INFO); sh.setFormatter(formatter)
        self.logger.addHandler(fh); self.logger.addHandler(sh)
    def get_logger(self): return self.logger

def train_epoch(model, loader, optimizer, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    for batch in tqdm(loader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        mask = batch['attention_mask'].to(device)
        dec_input = batch['decoder_input_ids'].to(device)
        dec_mask = batch['decoder_attention_mask'].to(device)
        labels = batch['targets'].to(device)
        imgs = batch['image_pixels'].to(device)
        
        optimizer.zero_grad()
        outputs = model(input_ids, mask, dec_input, dec_mask, labels, imgs, None)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        preds = outputs.logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return correct/total, total_loss/len(loader)

def eval_model(model, loader, device):
    model.eval()
    total_loss = 0
    loss_fct = torch.nn.CrossEntropyLoss()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            dec_input = batch['decoder_input_ids'].to(device)
            dec_mask = batch['decoder_attention_mask'].to(device)
            labels = batch['targets'].to(device)
            imgs = batch['image_pixels'].to(device)

            outputs = model(input_ids, mask, dec_input, dec_mask, None, imgs, None)
            logits = outputs.logits
            loss = loss_fct(logits, labels)
            total_loss += loss.item()
            
            preds = logits.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    target_names = ["Facility", "Amenity", "Service", "Branding", "Experience", "Loyalty", "NOT_HOTEL"]
    
    report = classification_report(all_labels, all_preds, target_names=target_names, zero_division=0, digits=4)
    print("\n" + report)
    
    # Macro F1 and Loss 
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    avg_loss = total_loss / len(loader)
    
    return float(macro_f1), float(avg_loss)
