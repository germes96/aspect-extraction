# Extracting Aspects from Laptop Reviews with Tensorflow

Use of recurrent neural networks for the aspect extraction task using the `python` language and the `tensorflow` library (`keras` shell)

## Requiement
1. Having a basic knowledge of recurent neural networks (https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9 )
2. Understanding the mechanics of the LSTM and GRU variants (https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21)

## Task

Given a sentence, the task is to extract aspects. Here is an example

```
I like the battery life of this phone"

Converting this sentence to IOB would look like this -

I O
like O
the O
battery B-A
life I-A
of O
this O
phone O

```
The aspect here is: `battery life`

## Data Parsing
In order to remove this stain, it is necessary to transform the entrances by applying pre-treatment stains. The stains applied here are: 
- Tokenize — Encode the words
- convert to lower case
- Remove punctuation
- Tokenize — Create Vocab to Int mapping dictionary
- Tokenize — Encode the labels

## Features
The characters that interest us here are respectively:
- Word Embedding (WE)
- POS TAG

## Execute Data parser (optional)
This file allows you to generate a vectorial representation of the characteristics of each word of the sentence.
- WE: Vector size 300
- POS: Vector size 36

1. Install all project dependencies
```
pip install requirements.txt 
```

2. To execute this python code, you need the 3.6GB `GoogleNews-vectors-negative300.bin` document available at : (https://github.com/mmihaltz/word2vec-GoogleNews-vectors).

```
python parse.py -Train 1 -ds_name Laptop #for train data
python parse.py -Train 0 -ds_name Laptop #for test data

```

NB: stanford's server must be running to avoid errors during execution (https://stanfordnlp.github.io/CoreNLP/corenlp-server.html)


## Training Data


The training data must be in the following format (identical to the CoNLL2003 dataset).

A default test file is provided to help you getting started.


```
The	O
duck	B-A
confit	I-A
is	O
always	O
amazing	O
and	O
the	O
foie	B-A
gras	I-A
terrine	I-A
with	I-A
figs	I-A
was	O
out	O
of	O
this	O
world	O

The	O
wine	B-A
list	I-A
is	O
interesting	O
and	O
has	O
many	O
good	O
values	O
```


## Details
1. Install all project dependencies
```
pip install requirements.txt 
```

2. Train the model with

```
python main.py -ds_name Laptop -n_epoch 20 -train 1 -model_name BiGRU
```


3. Evaluate and interact with the model with
```
ython main.py -ds_name Latop -n_epoch 20 -train 0 -model_name BiGRU
```

The best models for each batch are stored in the `models` directory.



## Result

Chunk based evaluation

```
Laptop 2014 -> F1 - 0.67  P -  0.68  R - 0.60

```


## Others Reference

[Soufian Jebbara, Philipp Cimiano](https://arxiv.org/pdf/1709.06311.pdf)


[Ma and Hovy](https://arxiv.org/pdf/1603.01354.pdf).

- concatenate final states of a bi-lstm on character embeddings to get a character-based representation of each word
- concatenate this representation to a standard word vector representation (GloVe here)
- run a bi-lstm on each sentence to extract contextual representation of each word
- decode with a linear chain CRF

[Collobert et al.] (http://ronan.collobert.com/pub/matos/2011_nlp_jmlr.pdf)

- form a window around the word to tag
- apply MLP on that window
- obtain logits
- apply viterbi (CRF) for sequence tagging

[Poria et al.](https://www.sciencedirect.com/science/article/pii/S0950705116301721)

- form a window around the word to tag
- apply CNN on that window
- apply maxpool on that window (Caution: different from global maxpool)
- obtain logits
- apply CRF for sequence tagging



## License

This project is licensed under the terms of the apache 2.0 license (as Tensorflow and derivatives). If used for research, citation would be appreciated.

