from typing import Sequence, List, Optional, Dict
from functools import partial

import torch
from spacy_streamlit.util import get_html
import streamlit as st
from spacy import displacy
import pandas as pd

from transformers import AutoModelForSequenceClassification

import ucla263brah.tc as tc

DEFAULT_EXAMPLE = [
    {
        "ents": [
            {"end": 323, "label": "Appeal_to_Authority", "start": 265},
            {"end": 1935, "label": "Appeal_to_Authority", "start": 1795},
            {"end": 157, "label": "Doubt", "start": 149},
            {"end": 1091, "label": "Repetition", "start": 1069},
            {"end": 1462, "label": "Appeal_to_fear-prejudice", "start": 1334},
            {"end": 1616, "label": "Appeal_to_fear-prejudice", "start": 1577},
            {"end": 1910, "label": "Appeal_to_fear-prejudice", "start": 1856},
            {"end": 2086, "label": "Appeal_to_fear-prejudice", "start": 2023},
        ],
        "text": 'Next plague outbreak in Madagascar could be \'stronger\': WHO\n\nGeneva - The World Health Organisation chief on Wednesday said a deadly plague epidemic appeared to have been brought under control in Madagascar, but warned the next outbreak would likely be stronger.\n\n"The next transmission could be more pronounced or stronger," WHO Director-General Tedros Adhanom Ghebreyesus told reporters in Geneva, insisting that "the issue is serious."\n\nAn outbreak of both bubonic plague, which is spread by infected rats via flea bites, and pneumonic plague, spread person to person, has killed more than 200 people in the Indian Ocean island nation since August.\n\nMadagascar has suffered bubonic plague outbreaks almost every year since 1980, often caused by rats fleeing forest fires.\n\nThe disease tends to make a comeback each hot rainy season, from September to April.\nOn average, between 300 and 600 infections are recorded every year among a population approaching 25 million people, according to a UN estimate.\n\nBut Tedros voiced alarm that "plague in Madagascar behaved in a very, very different way this year."\n\nCases sprang up far earlier than usual and, instead of being confined to the countryside, the disease infiltrated towns.\nThe authorities recorded more than 2 000 cases, and Tedros said Wednesday the death toll stood at 207.\n\nHe also pointed to the presence of the pneumonic version, which spreads more easily and is more virulent, in the latest outbreak.\n\nHe praised the rapid response from WHO and Madagascar authorities that helped bring the outbreak under control, but warned that the danger was not over.\n\nThe larger-than-usual outbreak had helped spread the bacteria that causes the plague more widely.\n\nThis along with poor sanitation and vector control on Madagascar meant that "when (the plague) comes again it starts from more stock, and the magnitude in the next transmission could be higher than the one that we saw," Tedros said.\n\n"That means that Madagascar could be affected more, and not only that, it could even spill over into neighbouring countries and beyond," he warned.\n\nComplicating vector control is the fact that the fleas that carry the Yersinia pestis bacteria that causes the plague have proven to be widely resistant to chemicals and insecticides.\n\n"That\'s a dangerous combination," Tedros said.\n',
        "title": None,
    }
]

DEFAULT_DATA = {
    "article_id": "article813452859",
    "context": "January?\nMichael Swadling: I guess her only chance is if Labour decides that they want to dishonour democracy and effectively keep us in the EU.\n© AP Photo / Pablo Martinez Monsivais UK 'In Need of Leadership', May's Brexit Deal Unwelcome to Trump - US Ambassador\nThere is a chance; as unfortunately there are many MPs who don't respect the vote and may just turn on it, but short of that I don't see any way the Conservatives would vote for it, and the majority is slender as it is, as the DUP is bitterly against it, and I can't see the Lib Dems voting for it, so it will only be if there are enough, what I can describe as remoaner MPs, that the deal won't be [BOP] dead in the water [EOP] \nSputnik: What could be a solution to the political chaos if the Prime Minister's deal is not approved?\nMichael Swadling: The EU withdrawal act is in place; we'll leave and revert to WTO terms and that works, that's fine.\nI often use the example of an iPhone to people; that's a piece of technology which is manufactured in China, uses American technology and these are two countries we deal with on WTO terms, this isn't a fantasy, stuck in a port somewhere, there isn't a massive tariff, this is the world that really exists today.\nWhen we exit the EU on WTO terms; that will be fine for whatever trading we do with the EU, just as well as it does for our trade in China.\nREAD MORE: UK",
    "span": "dead in the water",
    "span_start": 1293,
    "span_end": 1310,
    "text": "EU Profits From Trading With UK While London Loses Money – Political Campaigner\n\nWith the Parliamentary vote on British Prime Minister Theresa May’s Brexit plan set to be held next month; President of the European Commission Jean Claude Juncker has criticised the UK’s preparations for their departure from the EU.\nBut is there any chance that May's deal will make it through parliament and if it fails, how could this ongoing political deadlock finally come to an end?\nSputnik spoke with political campaigner Michael Swadling for more…\nSputnik: Does Theresa May have any chance of getting her deal through Parliament on the 14th January?\nMichael Swadling: I guess her only chance is if Labour decides that they want to dishonour democracy and effectively keep us in the EU.\n© AP Photo / Pablo Martinez Monsivais UK 'In Need of Leadership', May's Brexit Deal Unwelcome to Trump - US Ambassador\nThere is a chance; as unfortunately there are many MPs who don't respect the vote and may just turn on it, but short of that I don't see any way the Conservatives would vote for it, and the majority is slender as it is, as the DUP is bitterly against it, and I can't see the Lib Dems voting for it, so it will only be if there are enough, what I can describe as remoaner MPs, that the deal won't be dead in the water.\nSputnik: What could be a solution to the political chaos if the Prime Minister's deal is not approved?\nMichael Swadling: The EU withdrawal act is in place; we'll leave and revert to WTO terms and that works, that's fine.\nI often use the example of an iPhone to people; that's a piece of technology which is manufactured in China, uses American technology and these are two countries we deal with on WTO terms, this isn't a fantasy, stuck in a port somewhere, there isn't a massive tariff, this is the world that really exists today.\nWhen we exit the EU on WTO terms; that will be fine for whatever trading we do with the EU, just as well as it does for our trade in China.\nREAD MORE: UK Finance Chief Bashed for Failing to Unlock Money for No-Deal Brexit — Reports\nSputnik: Do you think that the EU needs the UK more than the UK needs the EU?\nMichael Swadling: The EU makes a profit on its trade with the UK; the UK makes a loss on its trade with the EU.\nThey have a financial incentive to ensure that good trading relations continue far more than we do.\n© REUTERS / Toby Melville UK Trade Minister Says '50-50' Chance Brexit Will Not Happen – Reports\nThe lifeblood and cash flow that keeps manufacturing in Europe going, comes from the city of London.\nIf someone in a city in Germany wants to do a deal with someone in Japan; the financial services of that are probably going through the city of London, they're not going through Frankfurt and Paris.\nViews and opinions, expressed in the article are those of Michael Swadling and do not necessarily reflect those of Sputnik\n\n",
    "technique": -1,
}


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
}

TC_TOKENIZER = tc.dataset.get_tokenizer("microsoft/deberta-large")
TC_MODEL = AutoModelForSequenceClassification.from_pretrained(
    "hd10/semeval2020_task11_tc"
)
TC_MODEL.eval()


def visualize_ner(
    example,
    labels: Sequence[str] = LABELS,
    attrs: List[str] = NER_ATTRS,
    show_table: bool = True,
    title: Optional[str] = "Named Entities",
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
# text = st.text_area("Text to analyze", DEFAULT_EXAMPLE[0]["text"], height=200)

visualize_ner(
    DEFAULT_EXAMPLE,
    show_table=True,
)

collate_fn = partial(tc.trainer.collate_fn_with_tokenizer, tokenizer=TC_TOKENIZER)
data = TC_TOKENIZER(DEFAULT_DATA["context"], truncation=True)
TC_MODEL.to("cuda:0")
with torch.no_grad():
    output = TC_MODEL(**collate_fn([data]).to("cuda:0"))

probs = torch.nn.functional.softmax(output.logits.cpu(), dim=-1).numpy()
st.text(DEFAULT_DATA["context"])
outputs = []
for i, p in enumerate(probs[0]):
    outputs.append([LABELS[i], f"{p*100:.2f}"])

df = pd.DataFrame(outputs, columns=["Label", "Probability"])
st.dataframe(df)
