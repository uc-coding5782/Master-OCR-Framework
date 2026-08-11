"""
High-accuracy multilingual OCR script.

Usage:
    python main.py --image path/to/image.jpg --lang en
    python main.py --image path/to/image.jpg --lang hindi --no-preprocess
    python main.py --image path/to/image.jpg --lang en --min-confidence 0.6

Run `python main.py --list-langs` to see supported language codes.
"""

import argparse
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from preprocessing import preprocess, load_image
from ocr_engine import OCREngine, SUPPORTED_LANGS
from postprocessing import clean_text


def main():
    parser = argparse.ArgumentParser(description="High-accuracy OCR (image to text)")
    parser.add_argument("--image", type=str, help="Path to input image")
    parser.add_argument("--lang", type=str, default="en", help="Language code (see --list-langs)")
    parser.add_argument("--min-confidence", type=float, default=0.5, help="Minimum OCR confidence to keep a line")
    parser.add_argument("--no-preprocess", action="store_true", help="Skip preprocessing (denoise/deskew/contrast)")
    parser.add_argument("--no-spellcheck", action="store_true", help="Skip spell correction step")
    parser.add_argument("--gpu", action="store_true", help="Use GPU if available")
    parser.add_argument("--list-langs", action="store_true", help="List supported language codes and exit")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save extracted text")

    args = parser.parse_args()

    if args.list_langs:
        print("Supported language shortcuts:")
        for name, code in SUPPORTED_LANGS.items():
            print(f"  {name:10s} -> {code}")
        print("\nPaddleOCR supports many more languages directly by code.")
        print("Full list: https://github.com/PaddlePaddle/PaddleOCR/blob/main/doc/doc_en/multi_languages_en.md")
        return

    if not args.image:
        parser.error("--image is required (unless using --list-langs)")

    lang_code = SUPPORTED_LANGS.get(args.lang, args.lang)  # allow raw codes too

    print(f"[1/4] Loading image: {args.image}")
    if args.no_preprocess:
        img = load_image(args.image)
    else:
        print("[2/4] Preprocessing (denoise, deskew, contrast, upscale)...")
        img = preprocess(args.image)

    print(f"[3/4] Running OCR (lang={lang_code})...")
    engine = OCREngine(lang=lang_code, use_gpu=args.gpu)
    results = engine.run(img)

    print("[4/4] Postprocessing...")
    final_text = clean_text(
        results,
        min_confidence=args.min_confidence,
        lang=lang_code,
        spell_correct=not args.no_spellcheck,
    )

    print("\n--- Extracted Text ---\n")
    print(final_text if final_text.strip() else "(no text detected)")

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            f.write(final_text)
        print(f"\nSaved output to: {args.output}")


if __name__ == "__main__":
    main()
