from transformers import (
    DistilBertConfig,
    DistilBertForTokenClassification,
    DistilBertTokenizer,
)
from transformers import WEIGHTS_NAME
from ucla263brah.si.span_identification.ner.bert_lstm_crf import BertLstmCrf
from ucla263brah.si.span_identification.ner.utils_ner import (
    convert_examples_to_features,
    InputExample,
)
import os
import torch
from torch.nn import CrossEntropyLoss

# MIGHT HAVE TO CHANGE THIS
output_dir = "/home/hd10/si_model/si_distilbert_single"


def predict_spans(sentences):
    tokenizer = DistilBertTokenizer.from_pretrained(
        "distilbert-base-uncased", do_lower_case=True
    )

    # Load configuration
    config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=3)

    # Load model
    bert_model = DistilBertForTokenClassification.from_pretrained(
        "distilbert-base-uncased", config=config
    )

    checkpoints = [output_dir]
    for checkp in checkpoints:
        model = BertLstmCrf(
            bert_model,
            config,
            num_labels=3,
            embedding_dim=config.hidden_size,
            hidden_dim=int(config.hidden_size / 2),
            rnn_layers=0,
            # rnn_dropout=config.hidden_dropout_prob,
            # output_dropout=config.hidden_dropout_prob,
            use_cuda=True,
        )
        checkpoint = os.path.join(checkp, WEIGHTS_NAME)
        state_dict = torch.load(checkpoint)
        model.load_state_dict(state_dict, strict=False)

    del bert_model

    examples = [
        InputExample(
            words=sentence.split(), guid=[], labels=["O" for x in sentence.split()]
        )
        for sentence in sentences
    ]
    model_type = "distilbert"
    max_seq_length = 256
    pad_token_label_id = CrossEntropyLoss().ignore_index
    label_list = ["O", "B-PROP", "I-PROP"]
    features = convert_examples_to_features(
        examples,
        label_list,
        max_seq_length,
        tokenizer,
        cls_token_at_end=bool(model_type in ["xlnet"]),
        # xlnet has a cls token at the end
        cls_token=tokenizer.cls_token,
        cls_token_segment_id=2 if model_type in ["xlnet"] else 0,
        sep_token=tokenizer.sep_token,
        sep_token_extra=bool(model_type in ["roberta"]),
        # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
        pad_on_left=bool(model_type in ["xlnet"]),
        # pad on the left for xlnet
        pad_token=tokenizer.convert_tokens_to_ids([tokenizer.pad_token])[0],
        pad_token_segment_id=4 if model_type in ["xlnet"] else 0,
        pad_token_label_id=pad_token_label_id,
    )

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)

    del features

    model.eval()
    with torch.no_grad():
        inputs = {
            "input_ids": all_input_ids,
            "attention_mask": all_input_mask,
            "labels": all_label_ids,
        }
        outputs = model(**inputs)
        _, _, predicted_tags = outputs

    preds = []
    # Iterate through each line of text
    for x in range(all_input_ids.shape[0]):
        p = []
        tokens = tokenizer.convert_ids_to_tokens(all_input_ids[x])

        # JUST A REMINDER:
        #  label "1" means start of propaganda, and "2" means the propaganda ends here.
        for i in range(len(tokens)):
            if tokens[i] == "[SEP]":
                break
            if tokens[i] == "[CLS]":
                continue
            p.append((tokens[i], predicted_tags[x][i]))

        preds.append(p)

    return preds


def get_span_indices_for_sentence(s, p):
    spans = []
    current_span = ""
    for span in p:
        if span[1] == 0:
            if len(current_span) > 0:
                spans.append(current_span.strip())
            current_span = ""
        else:
            current_span += span[0] + " "

    indices = []
    for span in spans:
        start = s.lower().find(span)
        end = start + len(span)
        assert span == s[start:end].lower()
        indices.append((start, end))

    return indices


def get_span_indices(sentences, preds):
    indices = []
    for i, p in enumerate(preds):
        index = get_span_indices_for_sentence(sentences[i], p)
        indices.append(index)

    return indices


if __name__ == "__main__":
    sentences = [
        "The American heroes found themselves in a disastrous situation.",
        "The American heroes found themselves in a disastrous situation.",
    ]
    preds = predict_spans(sentences)
    indices = get_span_indices(sentences, preds)
    print(preds)