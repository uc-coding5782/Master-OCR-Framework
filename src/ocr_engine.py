"""
Core OCR engine. Wraps PaddleOCR for text detection + recognition,
with multilingual support and confidence scoring.
"""

import os
os.environ.setdefault("PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT", "0")

from typing import Any
import numpy as np
from paddleocr import PaddleOCR

# PaddleOCR language codes you'll commonly want:
#   'en'  - English
#   'ch'  - Chinese + English
#   'french', 'german', 'korean', 'japan', 'hindi', 'ar' (Arabic), etc.
# Full list: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md
SUPPORTED_LANGS = {
    "en": "en",
    "chinese": "ch",
    "french": "french",
    "german": "german",
    "korean": "korean",
    "japanese": "japan",
    "hindi": "hi",
    "arabic": "ar",
    "spanish": "es",
    "russian": "ru",
}


class OCREngine:
    def __init__(self, lang: str = "en", use_angle_cls: bool = True, use_gpu: bool = False):
        """
        lang: language code (see SUPPORTED_LANGS or PaddleOCR docs for more)
        use_angle_cls: auto-detect and correct text orientation (90/180/270 deg)
        use_gpu: set True if you have a CUDA GPU available
        """
        self.lang = lang
        try:
            self.model = PaddleOCR(use_angle_cls=use_angle_cls, lang=lang, use_gpu=use_gpu, show_log=False)
        except (TypeError, ValueError):
            v3_kwargs: dict[str, Any] = {"lang": lang}
            if use_angle_cls:
                v3_kwargs["use_textline_orientation"] = True
            if use_gpu:
                v3_kwargs["device"] = "gpu"
            try:
                self.model = PaddleOCR(**v3_kwargs)
            except (TypeError, ValueError):
                self.model = PaddleOCR(lang=lang)

    def run(self, img: np.ndarray) -> list[dict]:
        """
        Run OCR on a preprocessed image.
        Returns a list of dicts: {"text": str, "confidence": float, "box": [[x,y],...]}
        """
        try:
            result = self.model.ocr(img, cls=True)
        except TypeError:
            result = self.model.ocr(img)

        extracted = []
        if not result or result[0] is None:
            return extracted

        if isinstance(result[0], dict):
            res_dict = result[0]
            texts = res_dict.get("rec_texts") or []
            scores = res_dict.get("rec_scores") or []
            polys = res_dict.get("rec_polys") or res_dict.get("dt_polys") or []
            for i in range(len(texts)):
                box = polys[i].tolist() if i < len(polys) and hasattr(polys[i], "tolist") else []
                extracted.append({
                    "text": texts[i],
                    "confidence": float(scores[i]) if i < len(scores) else 1.0,
                    "box": box,
                })
        else:
            for line in result[0]:
                box, (text, confidence) = line
                extracted.append({
                    "text": text,
                    "confidence": float(confidence),
                    "box": box,
                })
        return extracted

    def run_and_join(self, img: np.ndarray, min_confidence: float = 0.5) -> str:
        """Convenience method: return plain concatenated text above a confidence threshold."""
        results = self.run(img)
        lines = [r["text"] for r in results if r["confidence"] >= min_confidence]
        return "\n".join(lines)
