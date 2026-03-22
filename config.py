import os

class Config:
    # Paths
    RAW_FILE = '/kaggle/input/datasets/tisdang/hotel-review-splitted/text_image_dataset.json'
    TRIPLET_CSV = '/kaggle/input/datasets/quanglapnguyen/top5-triples/top5_triples.csv'
    IMG_DIR = '/kaggle/working/hotel_data/images'
    
    PROCESSED_DIR = '/kaggle/working/hotel_data'
    TRAIN_JSON = os.path.join(PROCESSED_DIR, 'train_extract_graph.json')
    TEST_JSON = os.path.join(PROCESSED_DIR, 'test_extract_graph.json')
    
    SAVE_DIR = './log_extract_graph'
    CHECKPOINT_PATH = os.path.join(SAVE_DIR, 'best_model_graph.pth')

    # Hyperparameters
    MAX_LEN = 128
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 2e-6
    NUM_CLASSES = 7
    NEGATIVE_RATIO = 0.8
    NUM_WORKERS = 2
