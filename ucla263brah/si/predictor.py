
from transformers import RobertaConfig, RobertaForTokenClassification, RobertaTokenizer
from transformers import DistilBertConfig, DistilBertForTokenClassification, DistilBertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from transformers import WEIGHTS_NAME, AutoModel
from span_identification.ner.bert_lstm_crf import BertLstmCrf
from span_identification.ner.utils_ner import convert_examples_to_features, InputExample
import os
import torch
from torch.nn import CrossEntropyLoss
import spacy
from torch.utils.data import TensorDataset
import numpy as np

# nlp = spacy.load("en_core_web_sm")

# MIGHT HAVE TO CHANGE THIS
output_dir = "model_checkpoints/si_distilbert_single"

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased", do_lower_case=True)
checkpoints = [output_dir]

# Load configuration
config = DistilBertConfig.from_pretrained("distilbert-base-uncased", num_labels=3)
print(config)
# asdf
# Load model
bert_model = DistilBertForTokenClassification.from_pretrained("distilbert-base-uncased",
                                    config=config)


# Give some dummy data
# example_sentence = "The American heroes found themselves in a disastrous situation."

# Open up the file
text_examples = []
with open("article813494037.txt", "r") as f:
    for line in f.readlines():
        line = line.replace("\n", "")
        if line:
            # print("1 : " + line)
            example_sentence = line.split()
            example_converted = InputExample(words=example_sentence, guid=[], labels=["O" for x in example_sentence])
            text_examples.append(example_converted)
# example_sentence = example_sentence.split()
# example_converted = InputExample(words=example_sentence, guid=[], labels=["O" for x in example_sentence])

model_type = "distilbert"
max_seq_length = 256
pad_token_label_id = CrossEntropyLoss().ignore_index
label_list = ["O", "B-PROP", "I-PROP"]
features = convert_examples_to_features(text_examples, label_list, max_seq_length, tokenizer,
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
                                        pad_token_label_id=pad_token_label_id
                                        )

all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
# print(all_input_ids)  # ROBERTA HAS 2 SEP TOKENS
all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
# all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
all_label_ids = torch.tensor([f.label_ids for f in features], dtype=torch.long)
print(all_input_ids.shape)
print(all_input_mask.shape)
print(all_label_ids.shape)

# dataset = TensorDataset(all_input_ids, all_input_mask)

# LOAD MODEL
# if args.eval_all_checkpoints:
#   checkpoints = list(os.path.dirname(c) for c in sorted(glob.glob(args.output_dir + "/**/" + WEIGHTS_NAME, recursive=True)))
#   logging.getLogger("pytorch_transformers.modeling_utils").setLevel(logging.WARN)  # Reduce logging
# logger.info("Evaluate the following checkpoints: %s", checkpoints)
for checkp in checkpoints:
  model = BertLstmCrf(
    bert_model,
    config,
    num_labels=3,
    embedding_dim=config.hidden_size,
    hidden_dim=int(config.hidden_size / 2),
    rnn_layers=0,
    #rnn_dropout=config.hidden_dropout_prob,
    #output_dropout=config.hidden_dropout_prob,
    use_cuda=True
  )
  checkpoint = os.path.join(checkp, WEIGHTS_NAME)
  print(checkpoint)
  state_dict = torch.load(checkpoint)
  model.load_state_dict(state_dict, strict=False)
#
# print("Saving to Huggingface")
# model.save_pretrained("model_checkpoints/semeval2020_task11_si")
# tokenizer.save_pretrained("model_checkpoints/semeval2020_task11_si")

#

# model = AutoModel.from_pretrained("tuscan-chicken-wrap/semeval2020_task11_si")
# model.load_state_dict(torch.load("model_checkpoints/ner_deberta_large_crf/pytorch_model.bin"))

model.eval()
with torch.no_grad():
    inputs = {"input_ids": all_input_ids,
              "attention_mask": all_input_mask,
              "labels": all_label_ids}
    # if args.model_type != "distilbert":
        # inputs["token_type_ids"] = batch[2] if args.model_type in ["bert", "xlnet"] else None  # XLM and RoBERTa don"t use segment_ids
    outputs = model(**inputs)

    # print(outputs['last_hidden_state'].shape)
    # print(outputs["pooler_output"].shape)
    _, _, predicted_tags = outputs
    # asdf

    # print(predicted_tags[0][1])

    # Iterate through each line of text
    for x in range(all_input_ids.shape[0]):

        tokens = tokenizer.convert_ids_to_tokens(all_input_ids[x])

        # JUST A REMINDER:
        #  label "1" means start of propaganda, and "2" means the propaganda ends here.
        for i in range(len(tokens)):
            print(tokens[i] + " : " + str(predicted_tags[x][i]))
            if tokens[i] == "[SEP]":
                break
