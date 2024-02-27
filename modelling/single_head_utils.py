"""Copyright (c) 2021, salesforce.com, inc.
All rights reserved.
SPDX-License-Identifier: BSD-3-Clause
For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause"""

import torch
from transformers import PreTrainedModel, BartModel, BartConfig, PegasusModel, PegasusConfig
from transformers.models.pegasus.modeling_pegasus import PegasusSinusoidalPositionalEmbedding
from transformers.modeling_outputs import Seq2SeqLMOutput
from torch import nn
import train_seq2seq_utils
from generation_utils import GenerationMixinCustom

"""
Definzione di una classe personalizzata che eredita le
funzionalità di un modello preaddestrato aggiungendo (quindi estendendo)
personalizzazioni specifiche.


        --> Gestisce la generazione di testo con un'unica testa/decoder. <--

Essendovi un'unica testa/decoder/expert, non esiste il concetto di "meccanismo di gating" poichè si ha
un input -> un decoder/testa/expert -> un output.
    
"""
class CustomPretrained(GenerationMixinCustom, PreTrainedModel):
    base_model_prefix = "model"
    _keys_to_ignore_on_load_unexpected = [r"encoder\.version", r"decoder\.version", r"final_logits_bias",
                                          r"lm_head\.weight"]

    # inizializzazione pesi in base al tipo di modulo
    def _init_weights(self, module):
        # deviazione standard specificata dalla configurazione del modello
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_() # azzera i bias
        elif isinstance(module, PegasusSinusoidalPositionalEmbedding):
            pass # non fa nulla: probabilmente esso non richiede inizializzazione dei pesi
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_() # azzera il peso corrispondente all'indice di padding

    # esempio di input fittizio per scopi di testing
    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]], device=self.device)
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens), device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    # getter
    def get_output_embeddings(self):
        return self.lm_head

    # setter
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    """
    In un modello di machine learning, la funzione forward svolge un ruolo cruciale ed 
    è responsabile dell'esecuzione dell'avanzamento (forward pass) della rete neurale. 
    La forward pass è il processo attraverso il quale i dati di input vengono trasformati in un'uscita dal modello. 
    Durante questa fase, i parametri del modello vengono utilizzati per effettuare le trasformazioni necessarie 
    sui dati di input per ottenere l'output richiesto (testo).
    --> La funzione forward è solitamente definita come il metodo che prende in input i dati (input) 
    e restituisce l'output del modello. 
    """

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        head_mask=None,
        decoder_head_mask=None,
        encoder_outputs=None,
        past_key_values=None,
        inputs_embeds=None,
        decoder_inputs_embeds=None,
        labels=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        generate=True,
        weights=None,
        # essendo modulo a singola testa --> riprende strategia di inferenza 1
        # Una singola testa non permette nè il concetto di gating
        #                                nè il concetto di separazione degli stili
        gate=None,
        sent_gate=None,
        use_sentence_gate_supervision=None,
        **kwargs,
    ):

        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids,
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=decoder_attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            decoder_inputs_embeds=decoder_inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        # calcolo dei logit per l'unico output (un decoder = una testa) grazie ad utilizzo di self.lm_head
        # self.lm_head è uno strato/layer lineare che proietta l'output del decoder del modello sullo spazio dei logit.
        # Ricorda: uno strato/layer lineare rappresenta una trasformazione lineare dei dati di input.
        #          Uno strato lineare trasforma gli input in output mediante una combinazione ponderata e l'aggiunta di un termine di bias.
        #          Viene quindi utilizzato per proiettare o trasformare gli input (che ora è l'output del decoder) in uno spazio di dimensioni diverse.
        lm_logits = self.lm_head(outputs[0]) + self.final_logits_bias

        masked_lm_loss = None
        if not generate: # se fase di addestramento, ossia non si sta "generando"
            # si ha l'effettivo calcolo di masked_lm_loss
            lm_labels = train_seq2seq_utils.shift_tokens_left(decoder_input_ids, 1)

            # calcolo della loss
            # utilizzo della funzione di perdita CrossEntropyLoss ignorando indice 1 corrispondente al token di padding
            loss_fct = nn.CrossEntropyLoss(ignore_index=1)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))
            """
            Si utilizza l'oggetto CrossEntropyLoss appena creato, ossia loss_fct, per calcolare la loss. 
            Gli argomenti passati a questa funzione sono i logit di output del modello (lm_logits) 
            e le etichette (lm_labels) dopo essere stati modificati utilizzando view(-1, self.config.vocab_size). 
            La funzione view viene utilizzata per modificare la forma dei tensori, in questo caso 
            per assicurarsi che abbiano due dimensioni: una per le previsioni del modello e una per le etichette.
            """
            # outputs = (masked_lm_loss,) + outputs

        # restituisce oggetto contenente informazioni necessare per addestramento o inferenza
        return Seq2SeqLMOutput(
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

    # Preparazione input necessari per la generazione (decodifica) durante inferenza
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

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return train_seq2seq_utils.shift_tokens_right(labels, self.config.pad_token_id,
                                                      self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past[:2]) + layer_past[2:],
            )
        return reordered_past


# definizione classe per implementazione modello basato su bart
# potrà esser utilzzata per generare sequenze di testo con appunto il modello BART personalizzato
class ConditionalGenerationCustomBart(CustomPretrained):
    config_class = BartConfig

    def __init__(self, config):
        super().__init__(config)
        base_model = BartModel(config)
        self.model = base_model
        # tensore di dimensione  (1, n° token nel vocabolario del modello) contenente zeri 
        # viene utilizzato come termine di bias costante per i logit finali del modello.
        # Ricorda: I tensori sono una struttura dati che rappresenta un array multidimensionale e contengono dati numerici. 
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        """
        self.lm_head è uno strato/layer lineare che proietta l'output del modello (config.d_model dimensioni) 
        su uno spazio di logit con dimensioni pari al numero di embeddings condivisi (self.model.shared.num_embeddings).
        Questo strato lineare è progettato per trasformare l'output del decoder (unico) del modello 
        (di dimensione config.d_model) in logit, dove ogni elemento dei logit è associato a un token nel vocabolario. 
        Il parametro bias=False indica che non viene utilizzato un termine di bias durante questa trasformazione lineare.
        """
        # creazione layer lineare che proietta l'output del decoder del modello nel vocabolario per la generazione dei token successivi
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        # richiama il metodo che inizializza i pesi del modello
        self.init_weights()


# definizione classe per implementazione modello basato su pegasus 
# (altro modello simile a BART ma sviluppato da google research)
# potrà esser utilzzata per generare sequenze di testo con appunto il modello Pegasus personalizzato
class ConditionalGenerationCustomPegasus(CustomPretrained):
    config_class = PegasusConfig
    _keys_to_ignore_on_load_missing = [r"embed_positions\.weight",]

    def __init__(self, config):
        super().__init__(config)
        base_model = PegasusModel(config)
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))
        """
        Come prima (vedi sopra):
        self.lm_head è uno strato/layer lineare che proietta l'output del modello (config.d_model dimensioni) 
        su uno spazio di logit con dimensioni pari al numero di embeddings condivisi (self.model.shared.num_embeddings).
        Quindi questo strato lineare è progettato per trasformare l'output del decoder del modello 
        (di dimensione config.d_model) in logit, dove ogni elemento dei logit è associato a un token nel vocabolario. 
        Il parametro bias=False indica che non viene utilizzato un termine di bias durante questa trasformazione lineare.
        """
        # creazione layer lineare che proietta l'output del modello nel vocabolario per la generazione dei token successivi
        self.lm_head = nn.Linear(config.d_model, self.model.shared.num_embeddings, bias=False)

        self.init_weights()