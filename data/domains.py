"""Shared FAMILY / CRIMINAL / TAX domain tagger.

One keyword definition used by the pretraining yield measurement
(``scripts/measure_domain_yield.py``) and by SFT prep (``reddit_domain``
source, ``scripts/sft_domain_report.py``).

Profiles:

* ``strict``: >= 3 distinct domain terms and >= 1.0 term hits per 10k chars.
* ``loose``:  >= 2 distinct terms and >= 0.3 hits per 10k chars.
  Both are for long documents (court opinions, statutes); precision measured
  on CourtListener: strict 92-96%, loose near-misses 47-93%
  (results/domain_measurement/REPORT.md).
* ``short``:  for short texts such as SFT questions (a ~400-token Reddit post
  rarely has 3 distinct legal terms). A domain matches with one STRONG term
  (e.g. divorce, child support, arrested, DUI) or >= 2 distinct terms from the
  long list plus a lay-phrasing list (my ex, custody, police, jail, ...).
  No density rule. Tuned once on RedditLegalQA / StackExchange; the precision
  check is in results/sft_domains/REPORT.md.

``classify`` is multi-label: a text can match several domains or none
("other").
"""

from __future__ import annotations

import re

import numpy as np

# Each entry: canonical term -> regex (applied to lowercased text).
TERMS = {
    "family": {
        "divorce": r"divorc(?:e|es|ed)", "dissolution of marriage": r"dissolution of (?:the )?marriage",
        "child custody": r"child custody", "parenting time": r"parenting time", "visitation": r"visitation",
        "child support": r"child support", "alimony": r"alimony", "spousal support": r"spousal support",
        "marital property": r"marital property", "equitable distribution": r"equitable distribution",
        "community property": r"community property", "prenuptial": r"prenuptial",
        "paternity": r"paternity", "adoption": r"adoptions?",
        "termination of parental rights": r"termination of (?:the )?parental rights",
        "guardianship of the minor": r"guardianship of the minors?", "domestic relations": r"domestic relations",
    },
    "criminal": {
        "indictment": r"indictments?", "sentencing": r"sentencing", "sentenced": r"sentenced",
        "conviction": r"convictions?", "convicted": r"convicted", "guilty plea": r"guilty pleas?",
        "mens rea": r"mens rea", "prosecutor": r"prosecutors?", "felony": r"felon(?:y|ies)",
        "misdemeanor": r"misdemeanors?", "probation": r"probation", "parole": r"parole",
        "miranda": r"miranda", "motion to suppress": r"motions? to suppress",
        "the jury found the defendant guilty": r"the jury found (?:the )?defendant guilty",
    },
    "tax": {
        "internal revenue code": r"internal revenue code", "i.r.c.": r"i\.\s?r\.\s?c\.",
        "26 u.s.c.": r"26 u\.\s?s\.\s?c\.", "commissioner of internal revenue": r"commissioner of internal revenue",
        "notice of deficiency": r"notices? of deficiency", "taxable income": r"taxable income",
        "gross income": r"gross income", "income tax": r"income tax(?:es)?",
        "estate tax": r"estate tax(?:es)?", "gift tax": r"gift tax(?:es)?",
        "tax refund": r"tax refunds?", "depreciation": r"depreciation", "capital gain": r"capital gains?",
        "accuracy-related penalty": r"accuracy-related penalt(?:y|ies)",
        "property tax": r"property tax(?:es)?", "sales tax": r"sales tax(?:es)?",
    },
}
DOMAINS = list(TERMS)
TERM_LIST = [(d, t) for d in DOMAINS for t in TERMS[d]]
DOMAIN_COLS = {d: [i for i, (dd, _) in enumerate(TERM_LIST) if dd == d] for d in DOMAINS}
# One literal-prefixed pattern per term: Python's re scans literal prefixes fast,
# a single 48-branch alternation with a leading lookbehind ran at ~1 MB/s.
TERM_RES = [re.compile(TERMS[d][t] + r"(?![a-z0-9])") for d, t in TERM_LIST]
NOT_AFTER = {"estate tax": "real "}  # "real estate tax" is property tax, not estate tax

PROFILES = {"strict": (3, 1.0), "loose": (2, 0.3)}  # (min distinct terms, min hits per 10k chars)

# ---- short profile: lay phrasing. STRONG terms tag on their own; WEAK terms
# need a second distinct term (weak or from TERMS) of the same domain.
SHORT_STRONG = {
    "family": [
        r"divorc(?:e|es|ed|ing)", r"child support", r"alimony", r"spousal (?:support|maintenance)",
        r"(?:child|full|joint|sole|primary|physical|legal|shared|50/50|50-50) custody",
        r"custody (?:battle|agreement|order|arrangement|schedule|case|hearing|rights|evaluation)",
        r"custody of (?:my |our |the |his |her |their |both )?(?:\w+ )?(?:kids?|child|children|son|daughter|baby|grandchild(?:ren)?|niece|nephew|sister|brother)",
        r"adoption(?! of (?:a |the )?(?:dog|cat|pet|puppy|kitten|rule|policy|resolution|ordinance))",
        r"adopt(?:ed|ing)? (?:a |my |our |the |his |her )?(?:\w+ )?(?:kids?|child|children|son|daughter|baby|stepchild|step-?son|step-?daughter|niece|nephew|grandchild)",
        r"(?:termination of |terminate (?:my |his |her |their )?)parental rights",
        r"parenting (?:time|plan)", r"prenup(?:tial)?s?", r"paternity", r"family court",
    ],
    "criminal": [
        r"(?<!cardiac )arrest(?:ed|s)?", r"charged with(?! (?:a |an |the )?(?:late )?(?:fees?|interest|rent|cleaning))", r"indict(?:ed|ment|ments)", r"felon(?:y|ies)",
        r"misdemeanors?", r"dui", r"dwi", r"owi", r"probation", r"parole",
        r"plea (?:deal|bargain|agreement)", r"plead(?:ed)? (?:guilty|no contest)", r"public defender",
        r"expunge(?:d|ment)?", r"expungement", r"sentenced", r"convicted", r"criminal (?:charges?|record|case|defense)",
    ],
    "tax": [
        r"irs", r"tax returns?", r"income tax(?:es)?", r"property tax(?:es)?", r"capital gains?",
        r"1099s?", r"w-?2s?",
    ],
}
SHORT_WEAK = {
    "family": [
        r"my ex", r"ex-?(?:wife|husband|girlfriend|boyfriend|partner)", r"custody", r"visitation",
        r"co-?parent(?:ing)?", r"baby (?:mama|momma|daddy)", r"dna test", r"marriage", r"married",
        r"separat(?:ed|ion)", r"spouse", r"guardian ad litem", r"step-?(?:son|daughter|father|mother|kids?|child)",
        r"child protective services", r"cps", r"restraining order", r"protective order", r"domestic violence",
        r"adopt(?:ed|ing|ion)?", r"parental rights", r"(?:father|mother)'?s rights", r"birth certificate",
    ],
    "criminal": [
        r"police|cops?|officers?", r"court date", r"criminal", r"jail", r"prison",
        r"warrant", r"theft", r"assault", r"shoplift(?:ing|ed)?", r"possession",
        r"district attorney", r"prosecut(?:or|ors|ion|e|ed)", r"bail", r"court-?appointed",
        r"speeding ticket", r"domestic violence", r"restraining order",
    ],
    "tax": [
        r"taxes", r"deduct(?:ion|ions|ible)?", r"audit(?:ed)?", r"tax refund", r"write[- ]off", r"taxable",
    ],
}


def _compile(patterns):
    return [re.compile(r"(?<![a-z0-9])(?:" + p + r")(?![a-z0-9])") for p in patterns]


_STRONG_RES = {d: _compile(v) for d, v in SHORT_STRONG.items()}
_WEAK_RES = {d: _compile(v) + [TERM_RES[i] for i in DOMAIN_COLS[d]] for d, v in SHORT_WEAK.items()}

SHORT_MAX_CHARS = 4000  # SFT texts longer than this are tagged with the "loose" profile


def term_counts(lower_text: str) -> np.ndarray:
    """Hits per TERM_LIST entry in already-lowercased text (uint16, saturating)."""
    counts = np.zeros(len(TERM_LIST), dtype=np.uint16)
    for i, rx in enumerate(TERM_RES):
        bad = NOT_AFTER.get(TERM_LIST[i][1])
        n = 0
        for m in rx.finditer(lower_text):
            s = m.start()
            if s and lower_text[s - 1].isalnum():
                continue
            if bad and lower_text[max(0, s - len(bad)):s] == bad:
                continue
            n += 1
        counts[i] = min(n, 65535)
    return counts


def classify_counts(counts: np.ndarray, lens: np.ndarray, domain: str, profile: str) -> np.ndarray:
    """Vectorised strict/loose test over many documents: counts[n, len(TERM_LIST)], lens[n]."""
    min_distinct, min_density = PROFILES[profile]
    c = counts[:, DOMAIN_COLS[domain]].astype(np.int64)
    distinct = (c > 0).sum(1)
    density = c.sum(1) / np.maximum(lens, 1) * 1e4
    return (distinct >= min_distinct) & (density >= min_density)


def _short_match(lower: str, domain: str) -> bool:
    if any(rx.search(lower) for rx in _STRONG_RES[domain]):
        return True
    return sum(1 for rx in _WEAK_RES[domain] if rx.search(lower)) >= 2


def short_matches(text: str) -> dict[str, dict[str, list[str]]]:
    """Which strong / weak short-profile terms matched, per domain (for spot checks)."""
    lower = text.lower()
    return {d: {"strong": sorted({m.group(0) for rx in _STRONG_RES[d] for m in [rx.search(lower)] if m}),
                "weak": sorted({m.group(0) for rx in _WEAK_RES[d] for m in [rx.search(lower)] if m})}
            for d in DOMAINS}


def classify(text: str, profile: str = "short") -> set[str]:
    """Domains ("family", "criminal", "tax") the text belongs to; empty set = other."""
    lower = text.lower()
    if profile == "short":
        return {d for d in DOMAINS if _short_match(lower, d)}
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; choose from short, {', '.join(PROFILES)}")
    counts = term_counts(lower)[None]
    lens = np.array([len(text)])
    return {d for d in DOMAINS if classify_counts(counts, lens, d, profile)[0]}


def profile_for(text: str) -> str:
    """The profile to use for one SFT text: short below SHORT_MAX_CHARS, loose above."""
    return "short" if len(text) <= SHORT_MAX_CHARS else "loose"


# ---------------------------------------------------------------- statutes by heading
# State codes have no domain field: a heading (Title/Chapter/...) whose text matches a
# domain list marks everything under it. Lists reviewed on the top headings by size
# (results/domain_measurement/REPORT.md for family/tax, results/legal_prep/REPORT.md
# for criminal).
SECTION_RE = re.compile(r"\n(?=[ \t]*(?:§|�)+\s*\d)")  # § is sometimes mojibake (U+FFFD)

HEADING_RE = re.compile(
    r"^[ \t]*(title|division|book|subtitle|chapter|part|article)\s+([0-9]+[\w.\-]*|[ivxlc]+\b)[^\n]{0,200}$",
    re.I | re.M)
LEVEL = {"title": 0, "division": 0, "book": 0, "subtitle": 1, "chapter": 2, "part": 3, "article": 4}
FAMILY_HEADING = re.compile(
    r"domestic relations|marriage|marital|married persons|husband and wife|divorce|dissolution of marriage"
    r"|annulment|separate maintenance|alimony|spousal (?:support|maintenance)|child support|support of (?:minor )?children"
    r"|custody of (?:minor )?child|child custody|custody and visitation|parenting|time-sharing|visitation"
    r"|parent and child|parentage|paternity|adoption of (?:minors|children|a child|adults)|adoption code|adoptions?\s*$"
    r"|termination of parental|parental rights|premarital|prenuptial|family law|family code|domestic partner"
    r"|guardianship of minors|uniform interstate family|uniform child custody", re.I)
FAMILY_HEADING_EXCLUDE = re.compile(
    r"therap|counsel|famil(?:y|ies) (?:farm|services|planning|investment|relief|child care|support services|resource)"
    r"|single-family|department of|license|tax|offenses against"
    # added after the heading review: education, time-share property, prison/DUI/nursing visitation, gift deduction
    r"|student|educat|real estate time|vacation time|drunk|censorship|home visitation|deduction", re.I)


def split_state_code(text, heading=FAMILY_HEADING, exclude=FAMILY_HEADING_EXCLUDE):
    """Yield (chunk_text, in_domain, setting_heading). A heading that matches
    `heading` (and not `exclude`) marks everything up to the next heading of the
    same or a higher level (title > subtitle > chapter > part > article)."""
    heads = list(HEADING_RE.finditer(text))
    stack = {}  # level -> heading text that set the family flag
    for i, m in enumerate(heads):
        lvl = LEVEL[m.group(1).lower()]
        for k in [k for k in stack if k >= lvl]:
            del stack[k]
        h = m.group(0).strip()
        if heading.search(h) and not exclude.search(h):
            stack[lvl] = h
        end = heads[i + 1].start() if i + 1 < len(heads) else len(text)
        chunk = text[m.start():end]
        setter = stack[min(stack)] if stack else None
        yield chunk, setter is not None, setter


TAX_HEADING = re.compile(
    r"taxation|\btax(?:es)?\b|revenue|assessment of property|ad valorem|tax court|tax appeals|tax tribunal", re.I)
TAX_HEADING_EXCLUDE = re.compile(
    r"revenue bonds?|tax increment|tax-exempt bond|lottery|gaming revenue|famil|marriage|divorce", re.I)


CRIMINAL_HEADING = re.compile(
    r"\bcrim(?:e|es|inal)\b|penal (?:code|law)|\boffen[cs]es\b|sentenc|punishment|corrections?\b"
    r"|probation|parole|controlled substances?|homicide|assault|larceny|theft|burglary|robbery"
    r"|sex(?:ual)? (?:offen|assault|abuse)|weapons|firearms|narcotic|indictment|arrest|bail\b"
    r"|extradition|habeas corpus|juvenile delinquen|driving under the influence|search warrant"
    r"|victims? of crime|crime victims?|prosecution|criminal procedure|jury trial", re.I)
CRIMINAL_HEADING_EXCLUDE = re.compile(
    r"\bcivil\b|commission|department of|board of|licens|\btax|insurance|unemployment|workers'? comp"
    r"|retirement|pension|appropriation|\bfund\b|\bgrants?\b|food|cosmetic|pharmac|fish|game|wildlife"
    r"|hunting|boating|alcoholic beverage control|criminal justice information|uniform commercial"
    # added once after the top-heading review (results/legal_prep/REPORT.md): body sentences parsed
    # as headings ("Article 5 of the ... Act shall ...", "Title 16, Section ..."), vehicle titling
    r"|^\s*\w+\s+[\w.\-]+,?\s+(?:of|section)\b|\bshall\b|may be cited|\(relating to"
    r"|technical corrections|certificate of title|anti-?theft|abandoned vehicles|accidents", re.I)
