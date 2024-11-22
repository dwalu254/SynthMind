# SynthMind Chatbot based on a Finetuned Qwen/Qwen2.5-1.5B Model
 
Qwen2.5 is the latest series of Qwen large language models. Qwen2.5 contains a number of base language models and instruction-tuned language models ranging from 0.5 to 72 billion parameters.
 
 
# Table of Contents
* Setup
* Running the Application
* Fine-Tuning and Uploading to Hugging Face Hub
* Troubleshooting
 
# Setup
Prerequisites
For installation of dependencies, run ```pip install -r ~/Requirements/requirements```
[Option on Google Colab] GPU (Optional but recommended for training)
 
 
# Running the Application
Important Note: Due to the commercialisation of popular LLM products, we opted to run the agent with local tensors. On the initial run, the application shall download the tensors locally. We kindly ask for your patience.
To start the application run ```streamlit run  ~/SynthMindchatbot.py```
 
 
# [Optional] Fine-Tuning and Uploading to Hugging Face Hub
```Qwen2_5_1_5B Train Script.ipynb``` file uses ```synthetic-orders-data.csv``` and ```synthetic-product-data.csv``` under datasets folder to train and fine tune Qwen2.5-1.5B and the finetuned model is uploaded to huggingface_hub
 
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
Tokenizer Errors:

Ensure the tokenizer matches the model (Qwen/Qwen2.5-1.5B).
