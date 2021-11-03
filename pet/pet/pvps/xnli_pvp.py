# coding: utf-8
from typing import Tuple, List, Union, Dict
from .pvp import PVP, FilledPattern
from .base import Verbalizer
from pet.utils import InputExample, get_verbalization_ids
import string


XNLIVerbalizer = {
    "en": {"contradiction": ["wrong"], "entailment": ["right"], "neutral": ["maybe"]},
    "zh": {"contradiction": ["否"], "entailment": ["是"], "neutral": ["也许"]},
    "de": {
        "contradiction": ["Nein"],
        "entailment": ["Ja"],
        "neutral": ["vielleicht"],
    },
    "fr": {"contradiction": ["non"], "entailment": ["oui"], "neutral": ["possible"]},
    "ar": {"contradiction": ["لا"], "entailment": ["نعم"], "neutral": ["يمكن"]},
    "bg": {"contradiction": ["не"], "entailment": ["да"], "neutral": ["можеби"]},
    "el": {"contradiction": ["όχι"], "entailment": ["Nαί"], "neutral": ["μπορεί"]},
    "es": {"contradiction": ["no"], "entailment": ["sí"], "neutral": ["talvez"]},
    "hi": {"contradiction": ["झूठ"], "entailment": ["सच"], "neutral": ["शायद"]},
    "ru": {"contradiction": ["нет"], "entailment": ["да"], "neutral": ["возможно"]},
    "sw": {"contradiction": ["hasi"], "entailment": ["ndio"], "neutral": ["labda"]},
    "th": {"contradiction": ["ไม่"], "entailment": ["ใช่"], "neutral": ["อาจจะ"]},
    "tr": {"contradiction": ["hiçbir"], "entailment": ["Evet"], "neutral": ["belki"]},
    "ur": {"contradiction": ["نہیں"], "entailment": ["ہاں"], "neutral": ["شاید"]},
    "vi": {"contradiction": ["Không"], "entailment": ["dạ"], "neutral": ["lẽ"]},
}


QAMAPPING = {
    "en": {"Q": "Question:", "A": "Answer:"},
    "zh": {"Q": "问题：", "A": "答案："},
    "de": {"Q": "Frage:", "A": "Antworten:"},
    "fr": {"Q": "Question:", "A": "Répondre:"},
    "ar": {"Q": "سؤال:", "A": "إجابه:"},
    "bg": {"Q": "Въпрос:", "A": "Отговор:"},
    "el": {"Q": "Ερώτηση:", "A": "Απάντηση:"},
    "es": {"Q": "Pregunta:", "A": "Respuesta:"},
    "hi": {"Q": "सवाल:", "A": "उत्तर:"},
    "ru": {"Q": "Вопрос:", "A": "Отвечать:"},
    "sw": {"Q": "Swali:", "A": "Jibu:"},
    "th": {"Q": "คำถาม:", "A": "ตอบ:"},
    "tr": {"Q": "Soru:", "A": "Cevap:"},
    "ur": {"Q": "سوال:", "A": "جواب:"},
    "vi": {"Q": "Câu hỏi:", "A": "Câu trả lời:"},
}


def _get_pattern(text_a, text_b, mask, pattern_lang, pattern_id):
    XNLIPatterns = {
        0: (
            [
                text_a,
                ".",
                QAMAPPING[pattern_lang]["Q"],
                text_b,
                "?",
                QAMAPPING[pattern_lang]["A"],
                mask,
                ".",
            ],
            None,
        ),
    }
    return XNLIPatterns[pattern_id]


class XNLIPVP(PVP):
    VERBALIZER = XNLIVerbalizer

    def __init__(self, wrapper, pattern_lang: str, pattern_id: int):
        self.pattern_lang = pattern_lang
        super(XNLIPVP, self).__init__(wrapper, pattern_id)

    def verbalize(self, label: str) -> List:
        return self.VERBALIZER[self.pattern_lang][label]

    def get_parts(self, example: InputExample) -> FilledPattern:
        if len(example.text_a.rstrip(string.punctuation)) == 0:
            text_a = self.shortenable(example.text_a)
        else:
            text_a = self.shortenable(example.text_a.rstrip(string.punctuation))
        if len(example.text_b.rstrip(string.punctuation)) == 0:
            text_b = self.shortenable(example.text_b)
        else:
            text_b = self.shortenable(example.text_b.rstrip(string.punctuation))
        return _get_pattern(
            text_a, text_b, self.mask, self.pattern_lang, self.pattern_id
        )
