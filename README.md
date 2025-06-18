# English to Urdu Neural Machine Translation

A neural machine translation system for English-Urdu language pairs built with Transformer architecture using Hugging Face's pre-trained models.

## 🎯 Overview

This project implements a fine-tuned neural machine translation model that translates English text to Urdu using the state-of-the-art Transformer architecture. The implementation leverages Helsinki-NLP's pre-trained OPUS-MT model and fine-tunes it on custom parallel corpus data.

## 🏗️ Architecture

### Model Components
- **Base Model**: Helsinki-NLP/opus-mt-en-ur (MarianMT)
- **Tokenizer**: Marian tokenizer with subword segmentation
- **Architecture**: Encoder-decoder Transformer with attention mechanism
- **Training**: Fine-tuning with AdamW optimizer and gradient clipping

### Key Features
- Pre-trained model fine-tuning for domain adaptation
- Beam search decoding for improved translation quality
- BLEU score evaluation for translation quality assessment
- Gradient clipping for stable training
- Custom dataset handling for parallel corpus

## 🚀 Features

- **Custom Dataset Support**: Load parallel English-Urdu corpus from CSV
- **Fine-tuning Pipeline**: Efficient fine-tuning of pre-trained models
- **Evaluation Metrics**: BLEU score calculation for translation quality
- **Beam Search**: Advanced decoding for better translation quality
- **GPU Acceleration**: CUDA support for faster training
- **Model Persistence**: Save and load trained models
- **Visualization**: Training loss plots and sample translations

## 📊 Performance

- **BLEU Score**: Achieves competitive translation quality
- **Training Efficiency**: Fast convergence with pre-trained initialization
- **Memory Optimization**: Efficient batch processing with attention masking

## 🛠️ Requirements

```bash
torch>=1.9.0
transformers>=4.15.0
sacrebleu>=2.0.0
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.5.0
scikit-learn>=1.0.0
tqdm>=4.62.0
```

## 💻 Usage

### Quick Start
```python
python english_urdu_translation.py
```

### Custom Dataset
```python
# Prepare your CSV file with 'english' and 'urdu' columns
df = pd.DataFrame({
    'english': ['Hello world', 'How are you?'],
    'urdu': ['ہیلو ورلڈ', 'آپ کیسے ہیں؟']
})
df.to_csv('data/your_corpus.csv', index=False)
```

### Translation Example
```python
from transformers import MarianMTModel, MarianTokenizer

model = MarianMTModel.from_pretrained('en_ur_transformer_model')
tokenizer = MarianTokenizer.from_pretrained('en_ur_transformer_model')

# Translate text
english_text = "Hello, how are you today?"
translation = translate_text(model, tokenizer, english_text)
print(f"Urdu: {translation}")
```

## 📈 Training Process

1. **Data Preprocessing**: Tokenization and padding of parallel sentences
2. **Fine-tuning**: Supervised learning on English-Urdu pairs
3. **Evaluation**: BLEU score calculation on test set
4. **Inference**: Beam search decoding for translation

## 🔧 Configuration

### Hyperparameters
- Learning Rate: 5e-5
- Batch Size: 16
- Max Sequence Length: 128
- Beam Size: 4
- Gradient Clipping: 1.0

### Training Settings
- Optimizer: AdamW
- Epochs: 3 (configurable)
- Device: Auto-detect GPU/CPU

## 📝 Dataset Format

Expected CSV format:
```csv
english,urdu
"Hello world","ہیلو ورلڈ"
"How are you?","آپ کیسے ہیں؟"
```

## 🔍 Evaluation

The model is evaluated using:
- **BLEU Score**: Standard MT evaluation metric
- **Sample Translations**: Qualitative assessment
- **Training Loss**: Convergence monitoring

## 🤝 Contributing

Contributions welcome! Areas for improvement:
- Additional language pairs
- Better evaluation metrics
- Data augmentation techniques
- Model architecture improvements

## 📄 License

MIT License - free to use and modify for research and commercial purposes.