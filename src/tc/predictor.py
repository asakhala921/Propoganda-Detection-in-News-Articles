import string
from collections import defaultdict

import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn import metrics
from unidecode import unidecode

import nltk

nltk.download("stopwords")
nltk.download("punkt")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize


def compute_metrics(preds, labels):
    p = F.softmax(torch.from_numpy(preds.predictions), dim=-1)
    pf = torch.argmax(p, dim=-1).squeeze().detach().cpu().numpy()
    return metrics.classification_report(
        preds.label_ids, pf, target_names=labels, output_dict=True
    )


def predict_and_print_results(trainer, dev_dataset, labels):
    preds = trainer.predict(dev_dataset)
    p = F.softmax(torch.from_numpy(preds.predictions), dim=-1)
    pf = torch.argmax(p, dim=-1).squeeze().cpu().detach().numpy()
    print(
        metrics.classification_report(
            preds.label_ids, pf, target_names=labels, digits=4
        )
    )

    return preds


# Adapted from https://github.com/aschern/semeval2020_task11/
def postprocess(x, probs, labels):
    stop_words = set(stopwords.words("english"))
    ps = PorterStemmer()
    inverse_mapping = {b: a for a, b in enumerate(labels)}

    spans_coords = list(zip(x["start"].values, x["end"].values))
    spans_source = x["span"].values
    spans = [
        " ".join(
            [
                ps.stem(word)
                for word in word_tokenize(unidecode(span.lower()))
                if word not in stop_words and word not in string.punctuation
            ]
        )
        for span in spans_source
    ]

    counts = defaultdict(set)
    for i in range(len(spans)):
        counts[spans[i]].add(spans_coords[i][0])
    for el in counts:
        counts[el] = len(counts[el])
    article_preds = probs[x.index]

    for i in range(len(article_preds)):
        log = article_preds[i]

        if (
            counts[spans[i]] >= 3
        ):  # or (counts[spans[i]] >= 2 and log[inverse_mapping["Repetition"]] >= 0.001):
            log[inverse_mapping["Repetition"]] = 1

        # if counts[spans[i]] == 1 and (log[inverse_mapping["Repetition"]] < 0.99 or len(spans[i].split()) <= 1):
        #     log[inverse_mapping["Repetition"]] = 0

        article_preds[i] = log

    probs[x.index] = article_preds
    return x, probs


def postprocess_predictions(dev_data, preds, labels):
    article_ids = [d["article_id"] for d in dev_data]
    spans = [d["span"] for d in dev_data]
    start = [d["span_start"] for d in dev_data]
    end = [d["span_end"] for d in dev_data]
    p = F.softmax(torch.from_numpy(preds.predictions), dim=-1).numpy()

    df = pd.DataFrame(
        {
            "article_ids": article_ids,
            "span": spans,
            "start": start,
            "end": end,
            "label": preds.label_ids,
        }
    )

    for i, g in enumerate(df.groupby("article_ids", as_index=False)):
        _, p = postprocess(g[1], p)

    pf = np.argmax(p, axis=-1)
    print(
        metrics.classification_report(
            preds.label_ids, pf, target_names=labels, digits=4
        )
    )
    df["predictions"] = pf
    return df
