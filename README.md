# 🏨 MATE-Hotel: Multimodal Aspect Term Extraction

This repository implements a **MATE** (Multimodal Aspect Term Extraction) model based on **BART-base**. The model is designed to identify and classify hotel-related aspects (Facility, Service, Amenity, Experience, Branding, Loyalty) from reviews by fusing textual context with visual knowledge triples via a **Visible Matrix (Graph Matrix)**.

## 🛠️ Installation

```bash
# 1. Clone the repository
git clone [https://github.com/quanglap100905/SeqCSG-MATE.git](https://github.com/quanglap100905/SeqCSG-MATE.git)
cd SeqCSG-MATE

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download the SpaCy model
python -m spacy download en_core_web_sm

# 4. Run
python prepare_data.py
python train.py
