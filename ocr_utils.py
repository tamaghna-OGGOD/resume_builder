"""
OCR helper – PDF ➜ Pillow images ➜ Tesseract ➜ plain text (via image_to_string)
---------------------------------------------------------------------------
pip install pdf2image pytesseract pillow beautifulsoup4  # (bs4 only for cleanup)
"""

import re
from typing import List, Optional

from pdf2image import convert_from_path          # Poppler wrapper
import pytesseract                               # Tesseract wrapper

# --------------------------------------------------------------------------- #
# Internal helpers                                                            #
# --------------------------------------------------------------------------- #
def _pdf_to_text_pages(
    pdf_path: str,
    poppler_path: Optional[str] = None,
    dpi: int = 300,
    lang: str = "eng",
) -> List[str]:
    """
    Convert each PDF page to a Pillow Image, then run Tesseract OCR
    (image_to_string) and return **plain-text** strings—one per page.
    """
    pages = convert_from_path(pdf_path, dpi=dpi, poppler_path=poppler_path)

    return [
        pytesseract.image_to_string(page, lang=lang)
        for page in pages
    ]

def _clean_whitespace(text: str) -> str:
    """Collapse runs of whitespace into single spaces and trim ends."""
    return re.sub(r"\s+", " ", text).strip()

# --------------------------------------------------------------------------- #
# Public function                                                             #
# --------------------------------------------------------------------------- #
def pdf_to_text(
    pdf_path: str,
    poppler_path: Optional[str] = None,
    dpi: int = 300,
    lang: str = "eng",
    keep_linebreaks: bool = False,
) -> str:
    """
    Full pipeline → returns a single cleaned string ready for the LLM.

    Parameters
    ----------
    keep_linebreaks : bool
        If True, preserves a blank line between original pages.
        If False (default), concatenates everything into one paragraph.
    """
    page_texts = _pdf_to_text_pages(
        pdf_path,
        poppler_path=poppler_path,
        dpi=dpi,
        lang=lang,
    )
    cleaned = [_clean_whitespace(t) for t in page_texts]
    return ("\n\n" if keep_linebreaks else " ").join(cleaned)
