import json, os, random, spacy, pandas as pd
from tqdm import tqdm
from config import Config

CAT_MAP = {
    "Facility": 0, "Amenity": 1, "Service": 2, 
    "Branding": 3, "Experience": 4, "Loyalty": 5,
    "NOT_HOTEL": 6
}

def get_text_safe(entry):
    for key in ['review', 'review_text', 'text', 'content']:
        if key in entry and entry[key]: return entry[key]
    return ""

def main():
    nlp = spacy.load("en_core_web_sm")
    os.makedirs(Config.HOTEL_DATA_DIR, exist_ok=True)

    # 1. Load Triplets
    triplet_map = {}
    try:
        df_trips = pd.read_csv(Config.TRIPLET_CSV)
        for _, row in df_trips.iterrows():
            img_name = str(row['image'])
            trip_data = {
                "text": f"{row['subject']} {row['relation']} {row['object']}",
                "sub": str(row['subject']).lower().strip(),
                "obj": str(row['object']).lower().strip()
            }
            if img_name not in triplet_map: triplet_map[img_name] = []
            if len(triplet_map[img_name]) < 5: triplet_map[img_name].append(trip_data)
    except Exception as e:
        print(f"⚠️ Warning: Triplets not found. {e}")

    # 2. Raw File
    with open(Config.RAW_FILE, 'r', encoding='utf-8') as f: 
        raw_data = json.load(f)
    
    positive_samples, negative_samples = [], []
    skipped_no_img = 0

    for entry in tqdm(raw_data, desc="Processing MATE"):
        img_id = entry.get('image_id')
        img_name = f"{img_id}.jpg"
        if not os.path.exists(os.path.join(Config.IMG_DIR, img_name)):
            skipped_no_img += 1
            continue

        text = get_text_safe(entry)
        if not text: continue
        caption = entry.get('photo_caption', "hotel view")
        triples = triplet_map.get(img_name, [])

        # Ground Truth Mapping
        ground_truth = {}
        aspects = entry.get('review_aspects', [])
        cats = entry.get('review_aspect_categories', [])
        for i, item in enumerate(aspects):
            term = item.get('term', '').lower().strip() if isinstance(item, dict) else str(item).lower().strip()
            cat = cats[i] if i < len(cats) else "NOT_HOTEL"
            
            mapped_cat = "NOT_HOTEL"
            c_low = cat.lower()
            if "facil" in c_low: mapped_cat = "Facility"
            elif "amen" in c_low: mapped_cat = "Amenity"
            elif "serv" in c_low: mapped_cat = "Service"
            elif "brand" in c_low: mapped_cat = "Branding"
            elif "exper" in c_low: mapped_cat = "Experience"
            elif "loyal" in c_low: mapped_cat = "Loyalty"
            if term and mapped_cat != "NOT_HOTEL": ground_truth[term] = mapped_cat

        # NLP Processing (Noun Chunks)
        doc = nlp(text)
        processed_chunks = set()
        for chunk in doc.noun_chunks:
            clean_chunk = chunk.text.lower().strip()
            for art in ["the ", "a ", "an "]:
                if clean_chunk.startswith(art): clean_chunk = clean_chunk[len(art):]
            if len(clean_chunk) < 2 or chunk.root.pos_ == "PRON": continue
            if clean_chunk in processed_chunks: continue
            processed_chunks.add(clean_chunk)

            label_str = "NOT_HOTEL"
            is_positive = False
            if clean_chunk in ground_truth:
                label_str = ground_truth[clean_chunk]
                is_positive = True
            else:
                for gt_term, gt_cat in ground_truth.items():
                    if gt_term in clean_chunk or clean_chunk in gt_term:
                        label_str = gt_cat
                        is_positive = True
                        break
            
            sample = {
                "review_text": text, "candidate_chunk": clean_chunk,
                "label": CAT_MAP.get(label_str, 6), "image_id": img_name,
                "caption": caption, "triples": triples 
            }
            if is_positive: positive_samples.append(sample)
            else: negative_samples.append(sample)

    # 3. Negative Sampling & Save
    if len(positive_samples) > 0:
        n_keep = max(1, int(len(positive_samples) * Config.NEGATIVE_RATIO))
        random.shuffle(negative_samples)
        all_samples = positive_samples + negative_samples[:n_keep]
        random.shuffle(all_samples)
        
        split = int(len(all_samples) * 0.8)
        with open(Config.TRAIN_JSON, 'w') as f: json.dump(all_samples[:split], f, indent=4)
        with open(Config.TEST_JSON, 'w') as f: json.dump(all_samples[split:], f, indent=4)
        print(f"🚀 Num of Samples: {len(all_samples)} (Lacked images: {skipped_no_img})")
    else:
        print("🛑 Can't find Positive samples.")

if __name__ == "__main__": main()
