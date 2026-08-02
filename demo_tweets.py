import gradio as gr
import torch.nn as nn
import torch
from transformers import AutoTokenizer, BertModel


MODEL_NAME = "google-bert/bert-base-cased"
MAX_LENGTH = 128
LABELS = {0: "negative", 1: "neutral", 2: "positive"}

CHECKPOINT_PATH = r"C:\Users\DORIANE\Desktop\NLP pour la finance\checkpoints_bert\model.pth"


class Model(nn.Module):
    def __init__(self, model_name=MODEL_NAME, num_classes=3):
        super().__init__()
        self.model = BertModel.from_pretrained(model_name)
        self.hidden_dim = self.model.config.hidden_size
        self.proj_lin = nn.Linear(self.hidden_dim, num_classes)

    def forward(self, input_ids):
        x = self.model(input_ids)
        x = x.last_hidden_state[:, 0]
        x = self.proj_lin(x)

        return x


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = Model(model_name=MODEL_NAME, num_classes=3)
model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location="cpu"))
model.eval()


def prediction(text):
    ids = tokenizer(
        text,
        padding="max_length",
        truncation=True,
        max_length=MAX_LENGTH,
        return_tensors="pt"
    )

    with torch.no_grad():
        outputs = model(ids["input_ids"])
        pred = outputs.argmax(dim=1)

    return "Sentiment prédit : " + LABELS[pred.item()]


demo = gr.Interface(
    fn=prediction,
    inputs=gr.Textbox(
        label="Texte du tweet",
        placeholder="Écris ou colle un tweet ici...",
        lines=3,
    ),
    outputs=gr.Textbox(label="Résultat"),
    title="Analyse de sentiment de tweets (BERT)",
    description=(
        "Modèle BERT (bert-base-cased) entraîné pour classifier un tweet en "
        "negative, neutral ou positive."
    ),
    examples=[
        ["I really loved this, it made my day!"],
        ["This is the worst service I have ever experienced."],
        ["I'm going to the store later."],
    ],
)

if __name__ == "__main__":
    demo.launch()