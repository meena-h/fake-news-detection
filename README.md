# Fake News Detection using DistilBERT

A multi-class text classification model built using 
DistilBERT transformer to detect and classify fake news.

## Tech Stack
- Python, PyTorch, HuggingFace Transformers
- DistilBERT (fine-tuned)
- Pandas, NumPy
- Google Colab (GPU)

## Approach
- Fine-tuned DistilBERT on custom news dataset
- Custom PyTorch Dataset with attention masks
- 80/20 train-validation split
- Adam optimizer with linear learning rate scheduler
- Best model selection based on validation accuracy
