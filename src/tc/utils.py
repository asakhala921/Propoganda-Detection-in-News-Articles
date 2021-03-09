from spacy import displacy
import pandas as pd
from pandas_profiling import ProfileReport

DEFAULT_COLORS = [
    "#7aecec",
    "#bfeeb7",
    "#feca74",
    "#ff9561",
    "#aa9cfc",
    "#c887fb",
    "#9cc9cc",
    "#ffeb80",
    "#ff8197",
    "#ff8197",
    "#f0d0ff",
    "#bfe1d9",
    "#bfe1d9",
    "#e4e7d2",
    "#e4e7d2",
    "#e4e7d2",
    "#e4e7d2",
    "#e4e7d2",
]


def get_labels_and_colors(dataset):
    labels = (
        dataset["train"].features["technique_classification"].feature["technique"].names
    )
    colors = {labels[i]: DEFAULT_COLORS[i] for i in range(len(labels))}

    return labels, colors


def visualize_example(index, dataset, labels, colors):
    text = dataset["train"][index]["text"]
    end_offsets = dataset["train"][index]["technique_classification"]["end_char_offset"]
    start_offsets = dataset["train"][index]["technique_classification"][
        "start_char_offset"
    ]
    technique = [
        labels[i]
        for i in dataset["train"][index]["technique_classification"]["technique"]
    ]

    ex = [
        {
            "text": text,
            "ents": [
                {
                    "start": start_offsets[i],
                    "end": end_offsets[i],
                    "label": technique[i],
                }
                for i in range(len(end_offsets))
            ],
            "title": None,
        }
    ]

    html = displacy.render(ex, style="ent", manual=True, options={"colors": colors})
    return html


def generate_span_lengths(d, label="span_identification"):
    spans = d[label]
    span_lengths = [
        spans["end_char_offset"][i] - spans["start_char_offset"][i]
        for i in range(len(spans["end_char_offset"]))
    ]
    return span_lengths


def prepare_df(dataset, tokenizer, labels):
    data = [article for article in dataset["train"]] + [
        article for article in dataset["validation"]
    ]  # + [article for article in dataset["test"]]
    articles = [d["text"] for d in data]
    article_lengths = [len(tokenizer(d["text"])["input_ids"]) for d in data]
    span_lengths = []
    techniques = []
    mismatch = 0
    for d in data:
        lengths = generate_span_lengths(d)
        lengths2 = generate_span_lengths(d, label="technique_classification")
        if lengths != lengths2:
            mismatch += 1
        span_lengths.extend(lengths2)
        techniques.extend(
            [labels[i] for i in d["technique_classification"]["technique"]]
        )

    return (
        pd.DataFrame({"articles": articles, "length": article_lengths}),
        pd.DataFrame({"span_lengths": span_lengths, "techniques": techniques}),
        mismatch,
    )


def generate_report(df, title="EDA for the PTC corpus"):
    return ProfileReport(
        df,
        title=title,
        dark_mode=True,
        explorative=True,
        correlations=None,
        interactions=None,
        missing_diagrams=None,
        samples=None,
        duplicates=None,
    )