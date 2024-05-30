from langchain.embeddings.huggingface import HuggingFaceEmbeddings
from langchain import HuggingFaceHub
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.schema import retriever
import os
import pandas as pd


from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch


tokenizer = AutoTokenizer.from_pretrained("tiiuae/falcon-7b-instruct")