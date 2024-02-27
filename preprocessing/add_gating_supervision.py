'''
Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
'''
import csv
import os
import string
from nltk import word_tokenize, ngrams
import numpy as np

punctuations = string.punctuation

"""
File che aggiunge informazioni di supervisione al meccanismo di gating.
Operazione volta a fornire al modello Hydrasum informazioni aggiuntive durante
la fase di addestramento, guidando il modello nella generazione di riassunti.


Lettura di un file tsv.
Restituisce una lista di dizionari, dove ogni dizionario
rappresenta una riga del file tsv: le chiavi del dizionario
corrispondono ai nomi delle colonne e i valori corrispondono
ai dati nella riga.
"""
def _read_tsv(input_file, quoting=csv.QUOTE_MINIMAL):
    """Reads a tab separated value file."""
    with open(input_file, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f, delimiter="\t", quoting=quoting)
        lines = []
        for line in reader:
            lines.append(line)
        return lines


"""
Dato in input un articolo ed un riassunto, restituisce
una lista di valori per il gate associati a ciascun token 
nel riassunto.
Se il token del riassunto è un segno di punteggiatura
--> valore del gate = -1
Se il token del riassunto è presente (come token) nell'articolo
--> valore del gate = 1
Altrimenti (se non presente) --> valore del gate = 0

Infine si verifica che la lunghezza dei gate corrisponda 
alla lunghezza dei token nel riassunto. Ciò viene fatto
tramite costrutto 'assert' che verifica che una condizione
specifica sia vera (se falsa, solleva un'eccezione).

OSS:
'set' converte la lista di token in un insieme, che è
una struttura dati che contiene solo elementi unici, ossia
non duplicati.
"""
# if token is seen, head1, if token is unseen head0
def get_gate_type0(article, summary):
    gate = []
    article_tokens = set(article.lower().split(' '))
    summary_tokens = summary.lower().split(' ')
    for token in summary_tokens:
        if token in punctuations:
            gate.append('-1')
        elif token in article_tokens:
            gate.append('1')
        else:
            gate.append('0')

    assert len(gate) == len(summary_tokens), 'gate length does not match summary length'
    return gate


"""
Simile alla funzione precedente ma con maggior "dettaglio".
Restituisce una lista di valori per il gate in base alla
presenza del token nel riassunto ed inoltre anche in base al 
contesto dei token precedenti.
Ciò avviene ovviamente con il ciclo for che itera sui token
del riassunto, restituendo sia l'indice (idx) che il valore
del token corrispondente in ogni iterazione.
Quindi, in ogni iterazione del ciclo:
- idx: rappresenta l'indice dell'elemento corrente
- token: rappresenta il token corrente

Se un token è nelle prime 3 posizioni del riassunto
--> valore del gate = -1
Questo perchè si ritiene che i primi token 
NON sono significativi o comunque troppo generici.
Analogamente a prima, se il token del riassunto è un segno di punteggiatura
--> valore del gate = -1
Terminati questi primi controlli:
Si crea una stringa 'prefix_wtoken' che rappresenta il contesto
di un token nelle ultime 2 posizioni del riassunto rispetto al numero
di iterazione attuale.
Si crea una seconda stringa 'prefix' che rappresenta il contesto
di un token nelle ultime 2 posizioni del riassunto rispetto al numero
di iterazione attuale ma ora SENZA il token stesso.
Tale passaggio è volto a catturare il contesto del token. 
Catturato tale contesto, si controlla se il token corrente 
e il suo contesto (nel riassunto) sono presenti nell'articolo.
Se il contesto con l'attuale parola è visto/presente nell'articolo 
--> valore del gate = 1
Se il contesto e il token non sono visti nell'articolo e il contesto non è visto
--> valore del gate = 0
Altrimenti: --> valore del gate = -1

Infine si verifica che la lunghezza dei gate corrisponda 
alla lunghezza dei token nel riassunto. Ciò viene fatto
tramite costrutto 'assert' che verifica che una condizione
specifica sia vera (se falsa, solleva un'eccezione).
"""
# if token is unseen and context is unseen, head1, if token is seen and context is seen head0
def get_gate_type2(article, summary):
    gate = []
    article_tokens = set(article.split(' '))
    summary_tokens = summary.split(' ')
    for idx, token in enumerate(summary_tokens):
        if idx < 3:
            gate.append('-1')
        elif token in punctuations:
            gate.append('-1')
        else:
            prefix_wtoken = ' '.join(summary_tokens[idx-2: idx + 1])
            prefix = ' '.join(summary_tokens[idx-2: idx])
            if prefix_wtoken in article:
                gate.append('1')
            elif prefix not in article and token not in article:
                gate.append('0')
            else:
                gate.append('-1')

    assert len(gate) == len(summary_tokens), 'gate length does not match summary length'
    return gate


"""
Ricorda: n-gram => sequenze contigue di n token
Funzione che calcola la sovrapposizone (overlap) di n-gram
tra due testi: l'input e l'output.
Calcola il rapporto tra il numero di n-gram comuni 
(che sono dati dall'intersezione tra gli n-gram in input e gli n-gram in output)
e il numero totale di n-gram nel testo di output.
Restituisce quindi un valore compreso tra 0 e 1 che indica
la percentuale di sovrapposizone tra n-gram.
Un valore più alto indica una maggiore similitudine tra
i due testi.

In particolare, grams_inp e grams_out sono un insieme degli n-gram, 
ossia non hanno duplicati e contengono gli n-gram (2-gram)
"""
def get_overlap(inp, out, ngram=2):
    grams_inp = set(ngrams(word_tokenize(inp.lower()), ngram))
    grams_out = set(ngrams(word_tokenize(out.lower()), ngram))

    total = len(grams_out)
    common = len(grams_inp.intersection(grams_out))
    if total == 0:
        return 0
    else:
        return float(common) / float(total)


"""
Funzione che utilizza la precedente 'get_overlap' per
calcolare la sovrapposizone tra un articolo ed un riassunto.
Quindi inp = articolo; out = riassunto.

Restituisce 1 se la sovrapposizione è maggiore della media,
altrimenti 0.

In sostanza questa funzione è utilizzata per assegnare un gate 
binario (0 o 1) in base alla percentuale di sovrapposizone tra 
gli n-gram dell'articolo (input) e del riassunto (output) rispetto
alla media, che viene calcolata nel main.
"""
def get_gate_type3(article, summary, mean_overlap):
    overlap = get_overlap(article, summary)
    if overlap < mean_overlap:
        return 0
    else:
        return 1



if __name__=='__main__':
    input_folder = '../../data/xsum/original_data'
    output_folder = '../../data/xsum/'

    split = 'train.tsv'
    input_file = os.path.join(input_folder, split)
    data = _read_tsv(input_file)

    outfile = open(os.path.join(output_folder, split), 'w')
    fieldnames = list(data[0].keys()) + ['gate', 'gate_sent']
    # creazione oggetto per scrivere dati
    writer = csv.DictWriter(outfile, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL, delimiter='\t')
    writer.writeheader()

    if split == 'train.tsv':
        mean_overlap = []
        for ex in data:
            article = ex['article']
            summary = ex['summary']
            overlap = get_overlap(article.lower(), summary.lower())
            mean_overlap.append(overlap)
        mean_overlap = np.mean(overlap)
    else:
        print('Please provide mean overlap based on the train file.')

    print(mean_overlap)


    # inizializzazione per diversi tipi di gate 
    # per poi calcolare i valori del gate tramite funzioni
    num_0s = 0.
    num_1s = 0.
    num_blank = 0.
    num_1sent = 0.

    for ex in data:
        article = ex['article']
        summary = ex['summary']

        gate = get_gate_type0(article.lower(), summary.lower())
        gate_sent = get_gate_type3(article.lower(), summary.lower(), mean_overlap)

        ex['gate'] = ' '.join(gate)
        ex['gate_sent'] = str(gate_sent)
        num_0s += gate.count('0')
        num_1s += gate.count('1')
        num_blank += gate.count('-1')
        num_1sent += gate_sent

        writer.writerow(ex)

"""
Stampa delle diverse statistiche
- percentuale di 1 nel gate
- percentuale di 0 nel gate
- percentuale di -1 nel gate
"""

print(num_1s/(num_0s + num_1s + num_blank))
print(num_0s/(num_0s + num_1s + num_blank))
print(num_blank/(num_0s + num_1s + num_blank))

print(num_1sent/len(data))