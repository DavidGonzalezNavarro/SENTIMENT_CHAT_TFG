from transformers import pipeline

pipe = pipeline("text2text-generation", model="facebook/m2m100_418M")