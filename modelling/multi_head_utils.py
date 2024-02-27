"""Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause"""

import torch
from transformers import PreTrainedModel, BartModel, BartConfig, BartPretrainedModel
from transformers.modeling_outputs import Seq2SeqLMOutput, Seq2SeqModelOutput, BaseModelOutput
from torch import nn
import train_seq2seq_utils
from transformers.models.bart.modeling_bart import BartEncoder, BartDecoder
import torch.nn.functional as F
import copy
from generation_utils_multi_heads import GenerationMixinCustom


"""
Definizione di una classe apposita per rappresentare modello bart con più teste: in particolare 2.
Quindi si analizza il caso in cui si ha un encoder e due decoder (= 2 teste = experts).
Condividono comunque layer condivisi.

        --> Gestisce la generazione di testo con 2 teste/decoder/experts. <--

Essendovi due teste/decoder/experts, ora esiste il concetto di "meccanismo di gating" poichè si ha
un input -> due decoder/teste/experts -> un output.
"""
class BartModelMultHeads(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # embedding condiviso tra encoder e due decoder
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # creazione unico encoder
        # creazione di due decoder (teste = 2): decoder0 e decoder1
        # l'embedding condiviso viene passato ad entrambi
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.decoder1 = BartDecoder(config, self.shared)

        self.num_decoder_layers_shared = None

        """
        Creazione di uno strato/layer lineare che proietta/mappa l'output
        di un layer del modello con dimensione config.d_model su un vettore di dimensione 2.
        Quindi il numero di output desiderato dal modulo lineare è 2.
        Questo layer lineare può essere utilizzato per assegnare pesi
        alle due teste del modello, per poi selezionare quale testa/decoder utilizzare durante 
        la generazione o l'inferenza del modello.
        
        Ricorda:
        Un layer lineare esegue una combinazione lineare degli input ponderati
        dai pesi, a cui viene aggiunto il bias. 
        Ciò permette di imparare relazioni lineari nei dati.
        Essendo però  'bias=False'  allora il layer lineare self.head_selector
        non includerà il termine di bias durante la trasformazione lineare.
        
        Tale layer lineare  self.head_selector  andrà applicato ad un input.
        Ciò fa ottenere un tensore di dimensione 2 come output, e tale output
        può essere interpretato come i pesi/probabilità associati alle due teste/decoder/experts.
        NB: un tensore di dimensione 2 non rappresenta direttamente la probabilità collegata
        a ciascuna testa/decoder. Infatti esso richiede un'ulteriore elaborazione attraverso
        una funzione di attivazione come la softmax per ottenere probabilità valide.
        Quindi si applicherà una softmax a tale output del layer lineare per ottenere
        le vere e proprie probabilità.
        
        OSS: 
        self.head_selector è uno strato/layer lineare che proietta l'output dei decoder 
        del modello sullo spazio dei logit.
        Infatti:
            output del layer lineare => logit
            allora si applica la softmax ai logit => probabilità
        
        Quindi  self.head_selector  è ora un layer lineare senza bias,
        che è in grado di trasformare un input di dimensione config.d_model
        in un output di dimensione 2.
        """
        self.head_selector = nn.Linear(config.d_model, 2, bias=False)

        # inizializzazione pesi del modello.
        # la funzione è definita in single_head_utils.py
        # inizializzazione dei pesi con distribuzione normale
        # con una deviazione standard std e media 0.
        self.init_weights()

    # restituisce embedding condiviso tra encoder e decoder/teste (2)
    # unchanged
    def get_input_embeddings(self):
        return self.shared

    # imposta l'eventuale nuovo embedding ad un nuovo valore 
    # fornito come argomento 'value'
    # unchanged
    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    # unchanged
    def get_encoder(self):
        return self.encoder

    # unchanged
    def get_decoder(self):
        return self.decoder

    """
    Funzione forward: 
    In un modello di machine learning, la funzione forward svolge un ruolo cruciale ed 
    è responsabile dell'esecuzione dell'avanzamento (forward pass) della rete neurale. 
    La forward pass è il processo attraverso il quale i dati di input vengono trasformati in un'uscita dal modello. 
    Durante questa fase, i parametri del modello vengono utilizzati per effettuare le trasformazioni necessarie 
    sui dati di input per ottenere l'output previsto.
    La funzione forward è solitamente definita come il metodo che prende in input i dati (input) 
    e restituisce l'output del modello. 
    
    -> è progettata per essere flessibile e adattarsi a diverse configurazioni 
    e opzioni di input del modello. 
    La sua logica si basa sulle impostazioni specifiche del modello e sugli 
    argomenti forniti durante la chiamata.
    """
    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            head_mask=None,
            decoder_head_mask=None,
            cross_attn_head_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            inputs_embeds=None,
            decoder_inputs_embeds=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            use_mixed=True,
            use_head=0,
    ):

        # setting dei parametri in base ai valori di configurazione del modello
        # verranno passati successivamente precisi valori 
        # quindi controlli sui valori di essi e successivo setting
                
        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        """
        Se decoder_input_ids non è fornito e neanche decoder_inputs_embeds, viene 
        creato automaticamente da input_ids usando la funzione shift_tokens_right.
        """
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = train_seq2seq_utils.shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        """
        output_attentions, output_hidden_states, use_cache, e return_dict, vengono impostati 
        ai valori di default specificati nella configurazione del modello se non sono forniti 
        durante la chiamata della funzione.
        """
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        """
        Se encoder_outputs non è fornito, la funzione chiama il modulo encoder 
        per ottenere gli output dell'encoder. 
        """
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                head_mask=head_mask,
                inputs_embeds=inputs_embeds,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
        # If the user passed a tuple for encoder_outputs, we wrap it in a BaseModelOutput when return_dict=True
        elif return_dict and not isinstance(encoder_outputs, BaseModelOutput):
            encoder_outputs = BaseModelOutput(
                last_hidden_state=encoder_outputs[0],
                hidden_states=encoder_outputs[1] if len(encoder_outputs) > 1 else None,
                attentions=encoder_outputs[2] if len(encoder_outputs) > 2 else None,
            )
            """
            Se l'opzione return_dict è abilitata e gli output dell'encoder non sono già di tipo BaseModelOutput, 
            vengono incapsulati (wrapping) in un oggetto BaseModelOutput.
            """

        # raggruppamento di tutti gli argomenti necessari al decoder tramite dizionario
        # così facendo sarà possibile passare tutti gli argomenti necessari
        # utilizzando la sintassi  **decoder_args (che tra l'altro semplifica e chiarisce il codice)
        decoder_args = {'input_ids': decoder_input_ids,
                        'attention_mask': decoder_attention_mask,
                        'encoder_hidden_states': encoder_outputs[0],
                        'encoder_attention_mask': attention_mask,
                        'head_mask': decoder_head_mask,
                        'cross_attn_head_mask': cross_attn_head_mask,
                        'past_key_values': past_key_values,
                        'inputs_embeds': decoder_inputs_embeds,
                        'use_cache': use_cache,
                        'output_attentions': output_attentions,
                        'output_hidden_states': True,
                        'return_dict': return_dict}

        # *** LOGICA DI OUTPUT DEL MODELLO ***
        # --> logica per modalità mista (importante 'use_mixed')
        # modalità mista = uso del meccanismo di gating: vengono generati due output dai due decoder ed inoltre
        #                  un selettore ('self.head_selector') decide quanto "peso" attribuire a ciascun decoder/testa.
        if use_mixed:
            # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
            # Passaggio di istanza precisa di input attraverso il modello
            # si ottengono così le previsioni/output del modello per le due teste
           
            # decoder_outputs -> output dalla prima testa/decoder
            # decoder_outputs1 -> output dalla seconda testa/decoder
            decoder_outputs = self.decoder(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args
            decoder_outputs1 = self.decoder1(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args

            # output comune ai due decoder, ottenuto selezionando l'output del layer condiviso
            # (h_m)^i sulla fig.2 del paper
            decoder_layer_common_output = decoder_outputs.hidden_states[self.num_decoder_layers_shared]
            # si ottengono i logit: valori "grezzi"/non normalizzati
            logits = self.head_selector(decoder_layer_common_output)
            # normalizzazione dei logit tramite softmax
            # otteniamo quindi la distribuzione di probabilità
            # ma solo una volta applicata la funzione softmax
            # la softmax converte i valori (logit) di un vettore in una probabilità
            # si ha quindi la probabilità di quanto ciascuna testa (decoder) contribuisce all'output finale
            prob_head_selector = nn.functional.softmax(logits, dim=-1)

            # oggetto risultato di un decoder (testa 0)
            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            ), Seq2SeqModelOutput( # oggetto risultato dell'altro decoder (testa 1)
                last_hidden_state=decoder_outputs1.last_hidden_state,
                past_key_values=decoder_outputs1.past_key_values,
                decoder_hidden_states=decoder_outputs1.hidden_states,
                decoder_attentions=decoder_outputs1.attentions,
                cross_attentions=decoder_outputs1.cross_attentions,
                encoder_last_hidden_state=None,
                encoder_hidden_states=None,
                encoder_attentions=None,
            ), prob_head_selector # specifica le probabilità associate a ciascuna testa/decoder
            # sono quindi le probabilità che indicano quanto ciascuna testa/decoder contribuisce all'ouput

        else:
            # --> logica per modalità singola testa (use_head)
            # modalità singola testa => non vi è il concetto di meccanismo gating
            # al dipendere dal valore di use_head viene eseguita la funzione forward,
            # ossia l'istanziazione di input attraverso il modello per ottenere le previsioni/output
            if use_head == 0:
                # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
                decoder_outputs = self.decoder(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args
            else:
                decoder_outputs = self.decoder1(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args

            if not return_dict:
                print('NEEDS TO BE IMPLEMENTED: Generation_mutlhead_utils. Use return_dict')
                exit()

            # restituzione oggetto in base al decoder selezionato
            # OSS: operando con una singola testa (decoder) non si ha una "pesatura" delle teste, 
            # essendo appunto una sola. Quindi non entra in gioco la probabilità (prob_head_selector)
            # associata alla testa/decoder
            return Seq2SeqModelOutput(
                last_hidden_state=decoder_outputs.last_hidden_state,
                past_key_values=decoder_outputs.past_key_values,
                decoder_hidden_states=decoder_outputs.hidden_states,
                decoder_attentions=decoder_outputs.attentions,
                cross_attentions=decoder_outputs.cross_attentions,
                encoder_last_hidden_state=encoder_outputs.last_hidden_state,
                encoder_hidden_states=encoder_outputs.hidden_states,
                encoder_attentions=encoder_outputs.attentions,
            )

# oss: essendovi qui la gestione per singola testa, non si parla di probabilità associata alla testa/decoder


"""
Tale classe che segue eredita la precedente (BartPretrainedModel).
Quindi si accede a tutti i metodi e attributi di BartModelMultHeads attraverso self.model.
"""
class ConditionalGenerationCustomBartMultHeads(GenerationMixinCustom, BartPretrainedModel):
    base_model_prefix = "model"
    authorized_missing_keys = [r"final_logits_bias", r"encoder\.version", r"decoder\.version"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModelMultHeads(config)
        self.model = base_model
        # tensore di dimensione  (1, n° token nel vocabolario del modello) contenente zeri 
        # viene utilizzato come termine di bias costante per i logit finali del modello.
        # Ricorda: I tensori sono una struttura dati che rappresenta un array multidimensionale e contengono dati numerici. 
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    """
    Setting dei layer condivisi tra i decoder/teste.
    In particolare si inizializzano pesi condivisi per i decoder, ossia tra decoder0 e decoder1
    """
    def initialize_correct_weights(self, config: BartConfig, num_decoder_layers_shared=6):
        num_layers = config.decoder_layers
        if num_decoder_layers_shared > num_layers:
            print(f'setting common decoder layers to max layers = {num_layers}')

        self.model.decoder1 = copy.deepcopy(self.model.decoder)

        # lego i pesi di due "modelli", ossia dei due decoder/teste: decoder1, decoder2.
        # si fa ciò con lo scopo di condividere pesi tra i decoder
        for k in range(num_decoder_layers_shared):
            _tie_decoder_weights(self.model.decoder.layers[k],
                                 self.model.decoder1.layers[k], f'decoder_layer{k}')

        self.model.num_decoder_layers_shared = num_decoder_layers_shared

    def freeze_weights(self):
        # con requires_grad = False --> i valori non saranno aggiornati
        # ciò viene fatto per l'encoder (ovviamente unico)
        # ma anche per i decoder (= teste -> 2)
        self.model.encoder.requires_grad_(False)
        for k in range(self.model.num_decoder_layers_shared):
            self.model.decoder.layers[k].requires_grad_(False)
            self.model.decoder1.layers[k].requires_grad_(False)

    def forward(
            self,
            input_ids,
            attention_mask=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            lm_labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            generate=True,
            use_mixed=True,
            use_head=None,
            gate=None,
            use_gate_supervision=False,
            gate_prob=None,
            use_sentence_gate_supervision=False,
            sent_gate=None, # dovrebbe essere un valore associato allo stile (preimpostato)
            **unused,
    ):
        # verifica su argomenti non utilizzati --> eventuale rimozione
        if "lm_labels" in unused:
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # raggruppamento di tutti gli argomenti necessari come input tramite dizionario
        # così facendo sarà possibile passare tutti gli argomenti necessari
        # utilizzando la sintassi  **input_args (che tra l'altro semplifica e chiarisce il codice)
        input_args = {'input_ids': input_ids,
                      'attention_mask': attention_mask,
                      'decoder_input_ids': decoder_input_ids,
                      'encoder_outputs': encoder_outputs,
                      'decoder_attention_mask': decoder_attention_mask,
                      'past_key_values': past_key_values,
                      'use_cache': use_cache,
                      'output_attentions': output_attentions,
                      'output_hidden_states': output_hidden_states,
                      'return_dict': return_dict,
                      'use_mixed': use_mixed,
                      'use_head': use_head}

        # se il modello è impostato in modalità mista
        # -> 2 output + 1 = tre output distinti
        # 'outputs' -> output della prima testa/decoder/expert (0)
        # 'outputs1' -> output della seconda testa/decoder/expert (1)
        # 'prob_head_selector' -> un ulteriore output corrispondente alla probabilità di scelta (= importanza) della testa/decoder/expert
        # Questo vettore di probabilità è ottenuto applicando la funzione softmax al risultato della linear transformation 
        # di un certo strato del decoder comune (decoder_layer_common_output), utilizzando il layer head_selector.
        if use_mixed:
            outputs, outputs1, prob_head_selector = self.model.forward(**input_args)
            # calcolo dei logit per ciascun output con utilizzo funzione lineare F
            # applicata al peso condiviso e al bias
            lm_logits0 = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
            lm_logits1 = F.linear(outputs1[0], self.model.shared.weight, bias=self.final_logits_bias)

            # calcolo delle softmax per ciascun set di logit 
            # -> si ottengono le probabilità normalizzate
            # softmax_0 -> probabilità della prima testa/decoder
            # softmax_1 -> probabilità della seconda testa/decoder
            softmax_0 = F.softmax(lm_logits0, dim=-1)
            softmax_1 = F.softmax(lm_logits1, dim=-1)

            # *** PESATURA DELLE PREVISIONI ***
            # suppongo non vi sia differenza tra gate_prob e gate_probability
            # se è fornita una "probabilità", essa viene utilizzata per pesare le softmax dei due output
            # non è proprio una probabilità, è un coefficiente di miscelazione nel gating 
            # una testa avrà coeff. di miscelazione '1-g' dove g = gate_prob = gate_probability
            # l'altra testa avrà coeff. di miscelazione 'g' dove g = gate_prob = gate_probability
            if gate_prob is not None:
                softmax_0 = softmax_0 * gate_prob
                softmax_1 = softmax_1 * (1 - gate_prob)
                
            # inoltre se attiva la supervisione in merito allo stile (guided training, addestramento guidato)
            # allora vengono applicate pesature basate sul valore sent_gate
            elif use_sentence_gate_supervision:
                # softmax_0 = torch.mul(softmax_0, (1 - sent_gate).unsqueeze(1).unsqueeze(2))
                # softmax_1 = torch.mul(softmax_1, sent_gate.unsqueeze(1).unsqueeze(2))
                softmax_0 = torch.mul(softmax_0, (1 - sent_gate).unsqueeze(2))
                softmax_1 = torch.mul(softmax_1, sent_gate.unsqueeze(2))
                #print(sent_gate)
                #print(softmax_0)
                #print(softmax_1)
                
            # altrimenti viene utilizzata la probabilità di selezione della testa calcolata
            # precedentemente, ossia utilizzando prob_head_selector
            else:
                # prob0 -> la probabilità associata alla prima testa
                # prob1 -> la probabilità associata alla seconda testa
                prob0 = prob_head_selector[:, :, 0].unsqueeze(2)
                prob1 = prob_head_selector[:, :, 1].unsqueeze(2)
                # softmax_0 -> le probabilità nella prima testa/decoder
                # softmax_1 -> le probabilità nella seconda testa/decoder
                softmax_0 = torch.mul(softmax_0, prob0)
                softmax_1 = torch.mul(softmax_1, prob1)
                # a seguito di tali risultati/probabilità, il modello dovrebbe esser anche stato 
                # capace di separare lo stile in modo automatico (unguided training).
                # Da verificare se ben compreso. ( verifica (?))

            # le softmax vengono combinate utilizzando pesi ottenuti dalla
            # probabilità di selezione della testa/decoder
            # si ha quindi una moltiplicazione elemento per elemento
            # sono quindi le probabilità logaritmiche normalizzate prodotte dal modello per ciascun token nel vocabolario
            
            lm_logits = torch.log(F.relu(softmax_0 + softmax_1) + 1e-6)  # TODO: This is not logits, rename

            """
            Questa riga di codice esegue due operazioni principali:
            1. `F.relu(softmax_0 + softmax_1)`: Somma i tensori `softmax_0` e `softmax_1` 
            elemento per elemento e applica poi la funzione di attivazione ReLU (Rectified Linear Unit) a questa somma. 
            La funzione ReLU restituisce 0 per tutti gli elementi negativi e mantiene invariati gli elementi positivi. 
            Questo è fatto per garantire che il logaritmo successivo non abbia argomenti negativi.
            2. `torch.log(... + 1e-6)`: Dopo aver applicato ReLU, si aggiunge un piccolo valore (1e-6) per evitare 
            problemi di logaritmo con argomenti nulli. Quindi, si calcola il logaritmo naturale del risultato ottenuto 
            dalla somma dei tensori.
            Questo risultato (`lm_logits`) viene chiamato "logits" anche se il termine "logits" in senso stretto 
            si riferisce a valori non normalizzati prima di applicare una funzione softmax. 
            In ogni caso, `lm_logits` rappresenta una sorta di punteggio associato a ciascun token 
            in base alle probabilità calcolate prima attraverso il meccanismo di softmax.
            
            --> ** ho un punteggio/valore/probabilità associata a ciascun token ** <--
            """
            
        # finito il concetto di gating e teste multiple. 
        # Ora controllo anche altro caso: se il modello è impostato per singola testa/decoder
        # -> singola testa/output -> = no meccanismo di gating.
        else:
            outputs = self.model.forward(**input_args)
            # calcolo dei logit
            lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
            # calcolo della softmax
            lm_logits = F.log_softmax(lm_logits, dim=-1)  # TODO: This is not logits, rename

        # gestione della loss del modello 
        # masked_lm_loss:   Viene inizializzata a None e conterrà la loss calcolata per la previsione della parola mascherata.
        # gate_loss:        Viene inizializzata a None e conterrà la loss calcolata per la supervisione sulla probabilità della testa/decoder.
        masked_lm_loss = None
        gate_loss = None
        if not generate: # ossia durante l'addestramento, cioè non si sta "generando"
            # si ha l'effettivo calcolo di masked_lm_loss
            lm_labels = train_seq2seq_utils.shift_tokens_left(decoder_input_ids, 1)

            # utilizzo della funzione di perdita NLLLoss ignorando indice 1 corrispondente al token di padding
            loss_fct = nn.NLLLoss(ignore_index=1)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            # la perdita viene calcolata confrontando lm_logits (ossia le probabilità logaritmiche normalizzate prodotte dal modello per ciascun token nel vocabolario) 
            # con lm_labels, ossia le parole target

            # se si è nella modalità mista e si fa uso del meccanismo di gating ossia si ha supervisione sulla probabilità
            # si ha l'effettivo calcolo di gate_loss
            # -> calcolo della perdita/loss associata al meccanismo di gating durante addestramento
            if use_mixed and use_gate_supervision:
                loss_fct_gate = nn.NLLLoss(ignore_index=-1)
                gate_loss = loss_fct_gate(torch.log(prob_head_selector.view(-1, 2)), gate.view(-1))
                # la perdita viene calcolata confrontando la probabilità di selezione della testa con gate
                 
        if not return_dict:
            # restituzione output come tupla
            output = (lm_logits,) + outputs[1:]
            return ((masked_lm_loss,) + output) if masked_lm_loss is not None else output

        if use_mixed: # modalità mista
            return_output = Seq2SeqLMOutput(
                            loss=masked_lm_loss,
                            logits=lm_logits,
                            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                            encoder_hidden_states=outputs.encoder_hidden_states,
                            encoder_attentions=outputs.encoder_attentions
                            )
        else: # modalità singola testa
            return_output = Seq2SeqLMOutput(
                            loss=masked_lm_loss,
                            logits=lm_logits,
                            past_key_values=outputs.past_key_values,
                            decoder_hidden_states=outputs.decoder_hidden_states,
                            decoder_attentions=outputs.decoder_attentions,
                            cross_attentions=outputs.cross_attentions,
                            encoder_last_hidden_state=outputs.encoder_last_hidden_state,
                            encoder_hidden_states=outputs.encoder_hidden_states,
                            encoder_attentions=outputs.encoder_attentions,
                            )

        # se si fa uso del meccanismo di gating ossia si ha supervisione sulla probabilità 
        # nell'output si include anche gate_loss e le probabilità della testa/decoder (prob_head_selector).
        if use_gate_supervision:
            return return_output, gate_loss, prob_head_selector
        else:
            return return_output

    # unchanged
    def prepare_inputs_for_generation(
            self,
            decoder_input_ids,
            past=None,
            attention_mask=None,
            head_mask=None,
            use_cache=None,
            encoder_outputs=None,
            **kwargs
    ):
        # cut decoder_input_ids if past is used
        if past is not None:
            decoder_input_ids = decoder_input_ids[:, -1:]

        return {
            "input_ids": None,  # encoder_outputs is defined. input_ids not needed
            "encoder_outputs": encoder_outputs,
            "past_key_values": past,
            "decoder_input_ids": decoder_input_ids,
            "attention_mask": attention_mask,
            "head_mask": head_mask,
            "use_cache": use_cache,  # change this to avoid caching (presumably for debugging)
        }

    # unchanged
    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

    # unchanged
    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    # unchanged
    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    # unchanged
    def get_decoder(self):
        return self.model.get_decoder()

    # unchanged
    def get_encoder(self):
        return self.model.get_encoder()

    # unchanged
    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return train_seq2seq_utils.shift_tokens_right(labels, self.config.pad_token_id,
                                                      self.config.decoder_start_token_id)


"""
Funzione utilizzata per collegare/legare i pesi di due "modelli", ossia
i due decoder/teste/experts: decoder0, decoder1.
Viene fatto ciò con lo scopo di condividere pesi tra i decoder.
Si definisce quindi di seguito la funzione
"""
def _tie_decoder_weights(decoder1: nn.Module, decoder2: nn.Module, module_name: str):
    def tie_decoder_recursively(
            decoder1_pointer: nn.Module,
            decoder2_pointer: nn.Module,
            module_name: str,
            depth=0,
    ):
        assert isinstance(decoder1_pointer, nn.Module) and isinstance(
            decoder2_pointer, nn.Module
        ), f"{decoder1_pointer} and {decoder2_pointer} have to be of type nn.Module"
        if hasattr(decoder1_pointer, "weight"):
            assert hasattr(decoder2_pointer, "weight")
            decoder1_pointer.weight = decoder2_pointer.weight
            if hasattr(decoder1_pointer, "bias"):
                assert hasattr(decoder2_pointer, "bias")
                decoder1_pointer.bias = decoder2_pointer.bias
            return

        decoder1_modules = decoder1_pointer._modules
        decoder2_modules = decoder2_pointer._modules
        if len(decoder2_modules) > 0:
            assert (
                    len(decoder1_modules) > 0
            ), f"Decoder modules do not match"

            all_decoder_weights = set([module_name + "/" + sub_name for sub_name in decoder1_modules.keys()])
            for name, module in decoder2_modules.items():
                tie_decoder_recursively(
                    decoder1_modules[name],
                    decoder2_modules[name],
                    module_name + "/" + name,
                    depth=depth + 1,
                )
                all_decoder_weights.remove(module_name + "/" + name)

            assert len(all_decoder_weights) == 0, 'There are some extra parameters in one of the decoders'

    # tie weights recursively
    tie_decoder_recursively(decoder1, decoder2, module_name)
