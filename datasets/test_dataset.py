from datasets import load_dataset

# If the dataset is gated/private, make sure you have run huggingface-cli login

#dataset = load_dataset("UBC-NLP/sparrow",'hate-2019-basile-spa',split="train",streaming=True)               #4499   Hate: 1857 Not:2642
#dataset = load_dataset("UBC-NLP/sparrow",'emotion-2018-mohammad-spa',split="train",streaming=True)          #2707   Anger: 691 Fear: 692 Joy: 649 Sadness:676
#dataset = load_dataset("UBC-NLP/sparrow",'emotion-2020-plaza-spa',split="train",streaming=True)             #6726   Supre
#dataset = load_dataset("UBC-NLP/sparrow",'humor-2021-chiruzzo-spa',split="train",streaming=True)            #23999
#dataset = load_dataset("UBC-NLP/sparrow",'irony-2016-barbieri-spa',split="train",streaming=True)            #6668
#dataset = load_dataset("UBC-NLP/sparrow",'sentiment-2016-mozetic-spa',split="train",streaming=True)       #122409
#dataset = load_dataset("UBC-NLP/sparrow",'irony-2019-ortega-spa',split="train",streaming=True)              #2159
#dataset = load_dataset("UBC-NLP/sparrow",'sentiment-2016-rei-spa',split="train",streaming=True)              #6098
dataset = load_dataset("UBC-NLP/sparrow",'subjective-2016-barbieri-spa',split="train",streaming=True)       #6098

print (dataset)

anger = 0
fear = 0
joy = 0
sadness = 0
surprise = 0
others = 0

labels = []
count = []
total = 0
for data in dataset:
    total += 1
    if(data['label'] not in labels):
        labels.append(data['label'])
        count.append(1)
    else:
        count[labels.index(data['label'])] += 1

for label in labels:
    print(label + ':')
    print(count[labels.index(label)] )

print(total)





