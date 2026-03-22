import torch
import os
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms

class HotelExtractDataset(Dataset):
    def __init__(self, data, tokenizer, max_len, image_dir):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.image_dir = image_dir
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        entry = self.data[item]
        review = entry['review_text']
        chunk = entry['candidate_chunk']
        label = entry['label']
        caption = entry['caption']
        triples_data = entry['triples'] # List of dicts {'text', 'sub', 'obj'}
        img_id = entry['image_id']

        # 1. TOKENIZE
        # S u C (Review + Caption)
        context_text = f"{caption}. {review}"
        ids_context = self.tokenizer.encode(context_text, add_special_tokens=False)
        
        # Triples
        triplets_ids = []
        for t in triples_data:
            t_ids = self.tokenizer.encode(t['text'], add_special_tokens=False)
            triplets_ids.append(t_ids)

        # 2. BUILD INPUT IDS
        bos = self.tokenizer.bos_token_id
        eos = self.tokenizer.eos_token_id
        
        # [BOS] Context [EOS]
        input_ids = [bos] + ids_context + [eos]
        
        # Ranges
        range_context = (0, len(input_ids)) 
        
        range_triples = []
        for t_ids in triplets_ids:
            start = len(input_ids)
            # Trip [EOS]
            input_ids.extend(t_ids + [eos])
            end = len(input_ids)
            range_triples.append((start, end))

        # Truncate
        if len(input_ids) > self.max_len:
            input_ids = input_ids[:self.max_len]
            # Adjust ranges
            range_context = (0, min(range_context[1], self.max_len))
            new_ranges = []
            for s, e in range_triples:
                if s < self.max_len:
                    new_ranges.append((s, min(e, self.max_len)))
            range_triples = new_ranges

        # Padding
        padding_len = self.max_len - len(input_ids)
        input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_len
        
        # 3. BUILD VISIBLE MATRIX 
        visible_matrix = np.full((self.max_len, self.max_len), -1e9, dtype=np.float32)
        
        # Helper: Set visible region
        def set_visible(r1, r2):
            # r1: (start, end), r2: (start, end)
            visible_matrix[r1[0]:r1[1], r2[0]:r2[1]] = 0.0

        # Equation in Adjacency matrix
        
        # Context (BOS/EOS/Caption/Review) see All and All see Context
        set_visible(range_context, (0, self.max_len)) 
        set_visible((0, self.max_len), range_context) 
        
        # w_i, w_j in same triple
        for r in range_triples:
            set_visible(r, r)
            
        # Connect 2 triples share same entity (sub/obj)
        for i in range(len(range_triples)):
            for j in range(i + 1, len(range_triples)):
                # Lấy entities của triple i và j
                ents_i = {triples_data[i]['sub'], triples_data[i]['obj']}
                ents_j = {triples_data[j]['sub'], triples_data[j]['obj']}
                
                if not ents_i.isdisjoint(ents_j):
                    # Có chung entity -> Visible lẫn nhau
                    set_visible(range_triples[i], range_triples[j])
                    set_visible(range_triples[j], range_triples[i])
                    
        # Masking Padding 
        real_len = len(input_ids) - padding_len
        visible_matrix[real_len:, :] = -1e9
        visible_matrix[:, real_len:] = -1e9
        
        # Diagonal
        np.fill_diagonal(visible_matrix, 0.0)

        # DECODER
        decoder_text = f"Aspect {chunk} is <mask>"
        dec = self.tokenizer.encode_plus(
            decoder_text, add_special_tokens=True, max_length=32,
            padding='max_length', return_attention_mask=True, return_tensors='pt', truncation=True
        )

        try:
            image = Image.open(os.path.join(self.image_dir, img_id)).convert("RGB")
            image = self.transform(image)
        except:
            image = torch.zeros(3, 224, 224)

        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(visible_matrix, dtype=torch.float),
            'decoder_input_ids': dec['input_ids'].flatten(),
            'decoder_attention_mask': dec['attention_mask'].flatten(),
            'targets': torch.tensor(label, dtype=torch.long),
            'image_pixels': image
        }
