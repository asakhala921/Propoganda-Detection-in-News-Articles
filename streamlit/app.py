from typing import Sequence, List, Optional, Dict
import re
import gc
import copy

from spacy_streamlit.util import get_html
import streamlit as st
from spacy import displacy
import pandas as pd
import numpy as np

from ucla263brah.tc.dataset import get_spans_in_context
from ucla263brah.tc.predictor import predict_tc
from ucla263brah.si.predictor import predict_spans, get_span_indices

DEFAULT_EXAMPLE = """
Humanity’s WIPEOUT Foreshadowed?
World Health Chief: Global Pandemic Imminent

According to a World Health Organization doctor, a global pandemic is imminent, and no one will be prepared for it when it hits.
Dr. Tedros Adhanom, director-general for WHO, has said that the next outbreak that will hit us will be a “terrible” one, causing a large death all over the world.
“Humanity is more vulnerable in the face of epidemics because we are much more connected and we travel around much more quickly than before,” said WHO specialist in infectious diseases Dr. Sylvie Brand.
“We know that it is coming, but we have no way of stopping it,” said Brand.
According to Dr. Tedros, the flu is extremely dangerous to everyone living on the planet.
This fear was also promoted by experts at the World Economic Forum in Davos, Switzerland last month.
The claims came exactly 100 years after the 1918 Spanish flu that claimed 50 million lives and killed three times as many people as World War I.
A mutated strain is the most likely contender to wipe out millions because it can join together with other strains to become deadlier.
“This is not some future nightmare scenario.
A devastating epidemic could start in any country at any time and kill millions of people because we are still not prepared.
The world remains vulnerable.
We do not know where and when the next global pandemic will occur, but we know it will take a terrible toll both on human life and on the economy,” said Dr. Tedros.
“Hidden underneath this fear-mongering message of a global pandemic is a far more sinister W.H.O.
agenda,” warns Mike Adams, the Health Ranger, publisher of Medicine.news.
“The real agenda is a global push for blind, fear-based acceptance of unsafe, unproven vaccines that will be rolled out alongside the next global pandemic,” Adams warns.
“Fear circumvents rational thinking, which is why the vaccine-pharma cartels routinely turn to irrational fear propaganda to demand absolute and unquestioning acceptance of risky medical interventions that should always be scrutinized for safety and efficacy.” –Natural News
Dr. Tedros’ comments come on the heels of the plague outbreak in Madagascar, which was the most recent epidemic to receive international aid attention amid fears it would spread.
More than 200 people were killed during the outbreak that ravaged the island over the winter, which prompted 10 nearby African countries to be placed on high alert.
"""

NER_ATTRS = ["label", "start", "end"]
LABELS = [
    "Appeal_to_Authority",
    "Appeal_to_fear-prejudice",
    "Bandwagon,Reductio_ad_hitlerum",
    "Black-and-White_Fallacy",
    "Causal_Oversimplification",
    "Doubt",
    "Exaggeration,Minimisation",
    "Flag-Waving",
    "Loaded_Language",
    "Name_Calling,Labeling",
    "Repetition",
    "Slogans",
    "Thought-terminating_Cliches",
    "Whataboutism,Straw_Men,Red_Herring",
]
COLORS = {
    "Appeal_to_Authority": "#7aecec",
    "Appeal_to_fear-prejudice": "#bfeeb7",
    "Bandwagon,Reductio_ad_hitlerum": "#feca74",
    "Black-and-White_Fallacy": "#ff9561",
    "Causal_Oversimplification": "#aa9cfc",
    "Doubt": "#c887fb",
    "Exaggeration,Minimisation": "#9cc9cc",
    "Flag-Waving": "#ffeb80",
    "Loaded_Language": "#ff8197",
    "Name_Calling,Labeling": "#ff8197",
    "Repetition": "#f0d0ff",
    "Slogans": "#bfe1d9",
    "Thought-terminating_Cliches": "#bfe1d9",
    "Whataboutism,Straw_Men,Red_Herring": "#e4e7d2",
    "Propoganda": "#7aecec",
}


def get_spacy_example(sentences, indices):
    template = {
        "ents": [],
        "text": "",
        "title": None,
    }

    current_index = 0
    for i, index in enumerate(indices):
        for start, end in index:
            if end - start < 2:
                continue

            template["ents"].append(
                {
                    "start": start + current_index,
                    "end": end + current_index,
                    "label": "Propoganda",
                }
            )

        template["text"] += sentences[i] + "\n"
        current_index = len(template["text"])

    return [template]


def visualize_ner(
    example,
    labels: Sequence[str] = LABELS,
    attrs: List[str] = NER_ATTRS,
    show_table: bool = True,
    title: Optional[str] = None,
    colors: Dict[str, str] = COLORS,
    key: Optional[str] = None,
) -> None:
    """Visualizer for named entities."""
    if title:
        st.header(title)
    exp = st.beta_expander("Select entity labels")
    label_select = exp.multiselect(
        "Entity labels",
        options=labels,
        default=list(labels),
        key=f"{key}_ner_label_select",
    )
    html = displacy.render(
        example,
        style="ent",
        manual=True,
        options={"ents": label_select, "colors": colors},
    )
    style = "<style>mark.entity { display: inline-block }</style>"
    st.write(f"{style}{get_html(html)}", unsafe_allow_html=True)
    if show_table:
        data = []
        for ent in example[0]["ents"]:
            d = [ent[attr] for attr in attrs]
            d.append(example[0]["text"][ent["start"] : ent["end"]])
            data.append(d)

        df = pd.DataFrame(data, columns=attrs + ["text"])
        st.dataframe(df)


st.title("Propoganda Detection")
input_text = re.sub(r"\n+", "\n", DEFAULT_EXAMPLE).strip()
text = st.text_area(
    "Text to analyze",
    input_text,
    height=200,
)


@st.cache(max_entries=10)
def propoganda_spans(text):
    if len(text) < 1:
        return None

    text = re.sub(r"\n+", "\n", text).strip()
    sentences = text.split("\n")
    preds = predict_spans(sentences)
    indices = get_span_indices(sentences, preds)
    ex = get_spacy_example(sentences, indices)
    gc.collect()
    return ex


ex = propoganda_spans(text)
if ex is not None:
    visualize_ner(
        ex,
        labels=["Propoganda"],
        show_table=False,
    )

    st.sidebar.header("Propoganda Spans Quality")
    st.sidebar.slider("", key="si")


@st.cache(max_entries=10)
def propoganda_techniques(ex):
    p = ex[0]
    p["technique_classification"] = {}
    starts = []
    ends = []
    for ent in p["ents"]:
        starts.append(ent["start"])
        ends.append(ent["end"])

    p["technique_classification"]["start_char_offset"] = starts
    p["technique_classification"]["end_char_offset"] = ends
    p["article_id"] = "dummy"

    data = get_spans_in_context(p, max_length=128)
    data = [d["context"] for d in data]

    if len(data) > 0:
        probs = predict_tc(data)
        preds = np.argmax(probs, axis=-1)

        del data, p["technique_classification"]

        for i, ent in enumerate(p["ents"]):
            ent["label"] = LABELS[preds[i]]

    return ex


if ex is not None:
    example_tc = propoganda_techniques(copy.deepcopy(ex))
    gc.collect()

    if len(example_tc[0]["ents"]) > 0:
        st.title("Technique Classification")
        visualize_ner(example_tc, show_table=False)
        st.sidebar.header("Propoganda Techniques Quality")
        st.sidebar.slider("", key="tc")
