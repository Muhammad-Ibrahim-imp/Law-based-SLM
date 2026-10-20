"""Text cleaning for legal documents.

``clean_legal_text(text)`` is used for every legal source. ``layout=True``
adds the rules for court opinions that came from PDFs / OCR (CourtListener,
HFforLegal): re-flowing hard-wrapped and double-spaced lines, joining words
hyphenated across lines, dropping repeated running headers and short OCR junk
lines. Statutes keep their line structure (list items like "(a)" are lines).

``quality_issue(text)`` returns the reason a cleaned document should be dropped
from pretraining, or None. The rules come from inspecting ~30 raw documents
per source (results/legal_prep/REPORT.md).
"""

import html
import re
import unicodedata

# Mojibake seen in Pile of Law / CourtListener text (UTF-8 read as cp1252).
_MOJIBAKE = {"â€™": "’", "â€˜": "‘", "â€œ": "“", "â€\x9d": "”", "â€”": "—", "â€“": "–",
             "Â§": "§", "Â¶": "¶", "Â ": " "}
_FFFD_SECTION = re.compile("�+(?=\\s*\\d)")        # "� 1983" was "§ 1983"
_FFFD_LETTERS = re.compile("(?<=[A-Za-z])�(?=[A-Za-z])")   # O�Neil -> O'Neil
_CONTROL = re.compile(r"[\x00-\x08\x0b\x0e-\x1f\x7f-\x9f]")
_HTML_TAG = re.compile(r"</?(?:p|div|span|br|a|i|b|u|em|strong|pre|blockquote|sup|sub|center|font|html|body)\b[^>]*>",
                       re.I)
_HTML_ENTITY = re.compile(r"&(?:[a-z]{2,8}|#\d{1,5}|#x[0-9a-f]{1,4});", re.I)
_STAR_PAGE = re.compile(r"\[\*+\d+\]|(?<!at )\*{1,3}\d+")   # *580, **12, [*5]; "at *2" is a Westlaw pin cite
_FILLER = re.compile(r"(?:l{6,}|_{4,}|={4,}|~{4,}|\.{6,})")  # "llllll" layout filler, rules
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_PHONE = re.compile(r"(?<![\w-])(?:\(\d{3}\)\s?|\d{3}[-.])\d{3}[-.]\d{4}(?![\w-])")
_SSN = re.compile(r"(?<![\w-])\d{3}-\d{2}-\d{4}(?![\w-])")


def clean_legal_text(text: str, layout: bool = False) -> str:
    # Unicode: NFC, known mojibake, U+FFFD replacement characters, control chars.
    text = unicodedata.normalize("NFC", text)
    for bad, good in _MOJIBAKE.items():
        if bad in text:
            text = text.replace(bad, good)
    text = _FFFD_SECTION.sub("§", text)
    text = _FFFD_LETTERS.sub("'", text)
    text = text.replace("�", "")
    text = text.replace("\x0c", "\n").replace("\r\n", "\n").replace("\r", "\n")
    text = _CONTROL.sub("", text)

    # Stray markup.
    if "<" in text:
        text = _HTML_TAG.sub("", text)
    if "&" in text and _HTML_ENTITY.search(text):
        text = html.unescape(text)

    # Replace non-breaking spaces with normal spaces
    text = text.replace("\xa0", " ").replace(" ", " ").replace(" ", " ")

    # Remove page-break separator lines
    text = re.sub(r"[-–—]{10,}", "", text)
    text = _FILLER.sub(" ", text)

    # Remove standalone page-number lines
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Remove inline page-break markers like *580 (asterisk + digits, no space)
    text = _STAR_PAGE.sub("", text)

    # Light PII scrub; party names are public record and stay.
    text = _EMAIL.sub("[EMAIL]", text)
    text = _SSN.sub("[SSN]", text)
    text = _PHONE.sub("[PHONE]", text)

    if layout:
        text = _fix_layout(text)

    # Collapse repeated whitespace
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)

    # normalizing smart quotes to straight quotes
    # text = re.sub(r"[“”]", '"', text) # double
    # text = re.sub(r"[‘’]", "'", text) # single

    # em-dash — / en-dash – vs plain hyphen -
    # ellipsis … vs three periods ...
    # apostrophe-like punctuation such as ′

    return text.strip()


_DIGITS = re.compile(r"\d+")
_END_OF_SENTENCE = re.compile(r"[.:;?!\"”’)\]]$")


def _fix_layout(text: str) -> str:
    """PDF/OCR layout repair for court opinions (see module docstring)."""
    lines = [ln.strip() for ln in text.split("\n")]
    nonblank = [ln for ln in lines if ln]
    if not nonblank:
        return ""

    # Running headers/footers: a line (digits ignored) that repeats 3+ times
    # is kept once. Short lines (<10 chars) are left alone.
    counts: dict[str, int] = {}
    for ln in nonblank:
        if len(ln) >= 10:
            key = _DIGITS.sub("#", ln.lower())
            counts[key] = counts.get(key, 0) + 1
    seen: set[str] = set()
    kept = []
    for ln in lines:
        if ln and len(ln) >= 10:
            key = _DIGITS.sub("#", ln.lower())
            if counts[key] >= 3:
                if key in seen:
                    continue
                seen.add(key)
        # OCR junk fragments: short lines without a real word ("E |", "a3", "=--000~;").
        # Citation lines ("128 So. 2d 92 (1961)") mix letters and multi-digit numbers and stay.
        if (ln and len(ln) <= 20 and not re.search(r"[A-Za-z]{3,}|\bNo\.", ln)
                and not re.search(r"[A-Za-z].*\d{2,}|\d{2,}.*[A-Za-z]", ln)
                and not re.fullmatch(r"\(?[\dA-Za-z]{1,4}[.)]?", ln)):
            continue
        kept.append(ln)

    # Double-spaced PDF text: almost every text line is followed by a blank line.
    blanks = sum(1 for ln in kept if not ln)
    texts = sum(1 for ln in kept if ln)
    if texts > 20 and blanks >= 0.6 * texts:
        kept = [ln for ln in kept if ln]

    # Re-flow hard-wrapped lines: join when the line does not end a sentence and
    # the next starts in lower case. A line-end hyphen is dropped only if the merged
    # word occurs elsewhere in the document ("dis-\nmissed" + "dismissed" seen);
    # otherwise it is a true compound and keeps it ("well-\nreasoned" -> "well-reasoned").
    words = set(_WORD.findall("\n".join(kept).lower()))
    out: list[str] = []
    for ln in kept:
        if out and out[-1] and ln and ln[0].islower():
            prev = out[-1]
            if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha():
                head = re.search(r"[A-Za-z]+$", prev[:-1]).group(0)
                tail = re.match(r"[a-z]+", ln).group(0)
                out[-1] = prev[:-1] + ln if (head + tail).lower() in words else prev + ln
                continue
            if not _END_OF_SENTENCE.search(prev):
                out[-1] = prev + " " + ln
                continue
        out.append(ln)
    return "\n".join(out)


# ---------------------------------------------------------------- quality filters
_WORD = re.compile(r"[A-Za-z]+")
_STOPWORDS = frozenset("the of and to a in that is for on by be as with was this it or not are at from "
                       "which an any shall have has his her he she court such may were".split())


def quality_issue(text: str, min_words: int = 150) -> str | None:
    """Why a cleaned document should be dropped, or None to keep it.

    too_short       fewer than ``min_words`` words (one-line orders, "Affirmed.")
    low_alpha       under 60% of non-space characters are letters (tables, OCR junk)
    repeated_lines  over 30% of the longer lines are repeats
    not_english     under 8% of words are common English function words
    """
    words = _WORD.findall(text)
    if len(words) < min_words:
        return "too_short"
    nonspace = sum(1 for ch in text if not ch.isspace())
    if nonspace and sum(1 for ch in text if ch.isalpha()) / nonspace < 0.6:
        return "low_alpha"
    long_lines = [ln for ln in text.split("\n") if len(ln) >= 20]
    if len(long_lines) >= 10 and 1 - len(set(long_lines)) / len(long_lines) > 0.3:
        return "repeated_lines"
    if sum(1 for w in words if w.lower() in _STOPWORDS) / len(words) < 0.08:
        return "not_english"
    return None
