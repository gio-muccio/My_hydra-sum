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
Versione estesa di multi_head_utils.py
Infatti ora si supportano 3 teste/decoder/experts.
Gran parte del codice riprende quanto visto in multi_head_utils.py, 
ma ovviamente ora vi sono dettagli riguardanti il 3° decoder.

        --> Gestisce la generazione di testo con 3 teste/decoder/experts. <--

Essendovi tre teste/decoder/experts, ora esiste il concetto di "meccanismo di gating" poichè si ha
un input -> tre decoder/teste/experts -> un output.
Vedi eventuali commenti utili in multi_head_utils.py.
"""
class BartModelMultHeads(BartPretrainedModel):
    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        # embedding condiviso tra encoder e tre decoder
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        # creazione unico encoder
        # creazione di tre decoder (teste = 3)
        # l'embedding condiviso viene passato ad entrambi
        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.decoder1 = BartDecoder(config, self.shared)
        self.decoder2 = BartDecoder(config, self.shared)

        self.num_decoder_layers_shared = None

        """
        self.head_selector è un layer/strato lineare che prende un vettore di input 
        con dimensione config.d_model e restituisce un vettore di output con dimensione 3, 
        che verrà utilizzato per selezionare quale testa o decoder utilizzare durante 
        la generazione o l'inferenza del modello.
        Tale layer lineare  self.head_selector  andrà applicato ad un input.
        Quindi: tensore di dimensione 3 come output, e tale output
        può essere interpretato come i pesi/probabilità associati alle tre teste/decoder/experts.
        In altri termini più semplici, il head_selector funziona come una sorta di "interruttore/rubinetto" 
        che determina quale tra le tre teste deve essere attivata e quanto in base alle informazioni 
        estratte dal vettore di input.
        """
        self.head_selector = nn.Linear(config.d_model, 3, bias=False)

        self.init_weights()

    # restituisce embedding condiviso tra encoder e decoders
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

        # different to other models, Bart automatically creates decoder_input_ids from
        # input_ids if no decoder_input_ids are provided
        if decoder_input_ids is None and decoder_inputs_embeds is None:
            decoder_input_ids = train_seq2seq_utils.shift_tokens_right(
                input_ids, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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
        # --> logica per modalità mista (use_mixed) = uso del meccanismo di gating
        # modalità mista, ora con 3 decoder/teste/experts.
        if use_mixed:
            # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
            # Passaggio di istanza precisa di input attraverso il modello
            # si ottengono così le previsioni/output del modello con le due teste
            decoder_outputs = self.decoder(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args
            decoder_outputs1 = self.decoder1(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args
            decoder_outputs2 = self.decoder2(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args

            # output del layer condiviso tra tre decoder
            decoder_layer_common_output = decoder_outputs.hidden_states[self.num_decoder_layers_shared]
            # si ottengono i logit: valori "grezzi"/ non normalizzati
            logits = self.head_selector(decoder_layer_common_output)
            # normalizzazione dei logit tramite softmax
            # otteniamo quindi la distribuzione di probabilità
            # una volta applicata la funzione softmax
            # la softmax converte i valori di un vettore in una probabilità
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
            ), Seq2SeqModelOutput( # oggetto risultato di un altro decoder (testa 1)
                last_hidden_state=decoder_outputs1.last_hidden_state,
                past_key_values=decoder_outputs1.past_key_values,
                decoder_hidden_states=decoder_outputs1.hidden_states,
                decoder_attentions=decoder_outputs1.attentions,
                cross_attentions=decoder_outputs1.cross_attentions,
                encoder_last_hidden_state=None,
                encoder_hidden_states=None,
                encoder_attentions=None,
            ), Seq2SeqModelOutput( # oggetto risultato di un altro decoder ancora (testa 2)
                last_hidden_state=decoder_outputs2.last_hidden_state,
                past_key_values=decoder_outputs2.past_key_values,
                decoder_hidden_states=decoder_outputs2.hidden_states,
                decoder_attentions=decoder_outputs2.attentions,
                cross_attentions=decoder_outputs2.cross_attentions,
                encoder_last_hidden_state=None,
                encoder_hidden_states=None,
                encoder_attentions=None,
            ), prob_head_selector # specifica le probabilità associate a ciascuna testa/decoder
            # sono quindi le probabilità che indicano quanto ciascun decoder
            # contribuisce all'ouput

        else:
            # --> logica per modalità singola testa (use_head)
            # modalità singola testa = non vi è il concetto di meccanismo gating
            # rispetto a prima, ora un'opzione in più avendo 3 teste
            # al dipendere dal valore di use_head viene eseguita la forward pass
            # ossia l'istanziazione di input attraverso il modello per ottenere le 
            # previsioni/output
            if use_head == 0:
                # decoder outputs consists of (dec_features, past_key_value, dec_hidden, dec_attn)
                decoder_outputs = self.decoder(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args
            elif use_head == 1:
                decoder_outputs = self.decoder1(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args
            else:
                decoder_outputs = self.decoder2(**decoder_args) # passa in input tutti gli elementi del dizionario decoder_args

            if not return_dict:
                print('NEEDS TO BE IMPLEMENTED: Generation_mutlhead_utils. Use return_dict')
                exit()

            # restituzione oggetto in base al decoder selezionato
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

class ConditionalGenerationCustomBartMultHeads(GenerationMixinCustom, BartPretrainedModel):
    base_model_prefix = "model"
    authorized_missing_keys = [r"final_logits_bias", r"encoder\.version", r"decoder\.version"]

    def __init__(self, config: BartConfig):
        super().__init__(config)
        base_model = BartModelMultHeads(config)
        self.model = base_model
        self.register_buffer("final_logits_bias", torch.zeros((1, self.model.shared.num_embeddings)))

    """
    Setting dei layer condivisi tra i decoder/teste.
    In particolare si inizializzano pesi condivisi per i decoder, ossia tra decoder, decoder1 e decoder2
    """
    def initialize_correct_weights(self, config: BartConfig, num_decoder_layers_shared=6):
        num_layers = config.decoder_layers
        if num_decoder_layers_shared > num_layers:
            print(f'setting common decoder layers to max layers = {num_layers}')

        self.model.decoder1 = copy.deepcopy(self.model.decoder)
        self.model.decoder2 = copy.deepcopy(self.model.decoder)

        # lego i pesi di tre "modelli", ossia dei tre decoder/teste: decoder, decoder1, decoder2.
        # si fa ciò con lo scopo di condividere pesi tra i decoder
        for k in range(num_decoder_layers_shared):
            _tie_decoder_weights(self.model.decoder.layers[k],
                                 self.model.decoder1.layers[k], f'decoder_layer{k}')
            _tie_decoder_weights(self.model.decoder.layers[k],
                                 self.model.decoder2.layers[k], f'decoder_layer{k}')

        self.model.num_decoder_layers_shared = num_decoder_layers_shared

    def freeze_weights(self):
        self.model.encoder.requires_grad_(False)
        for k in range(self.model.num_decoder_layers_shared):
            self.model.decoder.layers[k].requires_grad_(False)
            self.model.decoder1.layers[k].requires_grad_(False)
            # perchè il freeze del decoder2 non avviene ?

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
            sent_gate=None,
            **unused,
    ):
        # verifica su argomenti non utilizzati
        if "lm_labels" in unused:
            labels = unused.pop("lm_labels")
        if "decoder_cached_states" in unused:
            past_key_values = unused.pop("decoder_cached_states")
        if "decoder_past_key_values" in unused:
            past_key_values = unused.pop("decoder_past_key_values")
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

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

        # *** PESATURA DELLE PREVISIONI ***
        """
        OSS: con 3 teste vi è un approccio più "diretto", nel senso che vi sono meno controlli. (perchè?)
        Si estraggono direttamente le probabilità associate a ciascun decoder da prob_head_selector 
        e vengono utilizzate per ponderare le rispettive distribuzioni di probabilità. 
        Fin qui come prima.
        Ora non c'è alcun controllo del "peso" del gate specifico come nel caso a due teste, dove 
        veniva utilizzato gate_prob.
        Infatti prima (a due teste) veniva fornita una "probabilità" preimpostata, ed essa veniva utilizzata 
        per pesare le softmax dei due output.
        Si precisa come non sia proprio una probabilità, bensì è un coefficiente di miscelazione nel gating 
        Con due teste, una testa aveva coeff. di miscelazione '1-g' dove g = gate_prob = gate_probability
        l'altra testa aveva coeff. di miscelazione 'g' dove g = gate_prob = gate_probability
        Ora probabilmente questo coeff. di miscelazione non viene trattato con 3 teste perchè, a logica,
        tale concetto non si può estendere ad una terza testa.
        
        Invece, ora non si pone alcun controllo in merito allo stile. 
        Quindi non si ha use_sentence_gate_supervision per la separazione degli stili.
        Perchè?
        Questo ovviamente comporta a NON aver un addestramento guidato in merito allo stile,
        e di conseguenza si parla di "unguided training". 
        -> Ciò comporta ad avere solo l'opzione di separazione automatica degli stili.
        """
        # se il modello è impostato in modalità mista
        # -> tre output corrispondenti a tre decoder/teste/experts distinti
        if use_mixed:
            outputs, outputs1, outputs2, prob_head_selector = self.model.forward(**input_args)
            # calcolo dei logit per ciascun output con utilizzo funzione lineare F
            lm_logits0 = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
            lm_logits1 = F.linear(outputs1[0], self.model.shared.weight, bias=self.final_logits_bias)
            lm_logits2 = F.linear(outputs2[0], self.model.shared.weight, bias=self.final_logits_bias)

            # calcolo delle softmax per ciascun set di logit
            softmax_0 = F.softmax(lm_logits0, dim=-1)
            softmax_1 = F.softmax(lm_logits1, dim=-1)
            softmax_2 = F.softmax(lm_logits2, dim=-1)

            # ora viene utilizzata la probabilità di selezione della testa calcolata
            # precedentemente, ossia utilizzando prob_head_selector
            prob0 = prob_head_selector[:, :, 0].unsqueeze(2)
            prob1 = prob_head_selector[:, :, 1].unsqueeze(2)
            prob2 = prob_head_selector[:, :, 2].unsqueeze(2)
            softmax_0 = torch.mul(softmax_0, prob0)
            softmax_1 = torch.mul(softmax_1, prob1)
            softmax_2 = torch.mul(softmax_2, prob2)

            # le softmax vengono combinate utilizzando pesi ottenuti dalla
            # probabilità di selezione della testa/decoder
            lm_logits = torch.log(softmax_0 + softmax_1 + softmax_2 + 1e-6)  # TODO: This is not logits, rename
        else:
            # se il modello è impostato per singola testa/decoder
            # -> singola testa/output
            outputs = self.model.forward(**input_args)
            lm_logits = F.linear(outputs[0], self.model.shared.weight, bias=self.final_logits_bias)
            lm_logits = F.log_softmax(lm_logits, dim=-1)  # TODO: This is not logits, rename

        # gestione della loss del modello 
        # masked_lm_loss:   Viene inizializzata a None e conterrà la loss calcolata per la previsione della parola mascherata.
        masked_lm_loss = None
        if not generate: # ossia durante l'addestramento, cioè se non si "genera"
            # si ha l'effettivo calcolo di masked_lm_loss
            lm_labels = train_seq2seq_utils.shift_tokens_left(decoder_input_ids, 1)

            loss_fct = nn.NLLLoss(ignore_index=1)
            masked_lm_loss = loss_fct(lm_logits.view(-1, self.config.vocab_size), lm_labels.view(-1))

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


# perchè i pesi legati/collegati sono solo tra due decoder e non fra tre decoder
# essendovi ora 3 decorder?
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
