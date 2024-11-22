

# SynthMind Chatbot based on a Finetuned Qwen/Qwen2.5-1.5B Model

Qwen2.5 is the latest series of Qwen large language models. For Qwen2.5, we release a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters. Qwen2.5 brings the following improvements upon Qwen2:

Significantly more knowledge and has greatly improved capabilities in coding and mathematics, thanks to our specialized expert models in these domains.
Significant improvements in instruction following, generating long texts (over 8K tokens), understanding structured data (e.g, tables), and generating structured outputs especially JSON. More resilient to the diversity of system prompts, enhancing role-play implementation and condition-setting for chatbots.
Long-context Support up to 128K tokens and can generate up to 8K tokens.
Multilingual support for over 29 languages, including Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic, and more.

# Table of Contents
* Setup
* Fine-Tuning and Uploading to Hugging Face Hub
* Using the Fine-Tuned Model
* Troubleshooting

# Setup
Prerequisites
For installation of dependencies, run ```pip install requirements```
GPU (Optional but recommended for training)

# Fine-Tuning and Uploading to Hugging Face Hub
```Qwen2_5_1_5B Train Script.ipynb``` file uses ```synthetic-orders-data.csv``` and ```synthetic-product-data.csv``` under datasets folder to train and fine tune Qwen2.5-1.5B
and the finetuned model is uploaded to huggingface_hub

# Using the Fine-Tuned Model
The model can be used via StreamLit where Queries can be passed directly and an output is displayed
To start the application run ```streamlit run SynthMindchatbot.py```

# Troubleshooting
Out-of-Memory Errors:

Reduce the per_device_train_batch_size.
Use gradient accumulation:
python
Copy code
gradient_accumulation_steps = 4
Slow Training:

Use a GPU or enable mixed precision:
python
Copy code
fp16=True
Tokenizer Errors:

Ensure the tokenizer matches the model (Qwen/Qwen2.5-1.5B).
