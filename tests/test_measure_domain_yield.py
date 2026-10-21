import numpy as np

from scripts.measure_domain_yield import TERM_LIST, classify, split_state_code, term_counts


def test_term_counts_and_classify():
    text = ("The divorce decree set child support and alimony; real estate taxes and "
            "the Internal Revenue Code (I.R.C. s. 61) define gross income.").lower()
    c = term_counts(text)
    hits = {TERM_LIST[i][1] for i in np.flatnonzero(c)}
    # "real estate taxes" must not count as estate tax
    assert hits == {"divorce", "child support", "alimony", "internal revenue code", "i.r.c.", "gross income"}
    lens = np.array([len(text)])
    assert classify(c[None], lens, "family", "strict")[0]
    assert classify(c[None], lens, "tax", "strict")[0]
    assert not classify(c[None], lens, "criminal", "loose")[0]


def test_split_state_code_inherits_family_flag():
    text = ("Title 19 - Domestic Relations\nChapter 1 - Marriage\nx\nChapter 2 - General Provisions\ny\n"
            "Title 20 - Corporations\nChapter 1 - Dissolution\nz\n")
    flags = [(chunk.split("\n")[0], fam) for chunk, fam, _ in split_state_code(text)]
    assert flags == [("Title 19 - Domestic Relations", True), ("Chapter 1 - Marriage", True),
                     ("Chapter 2 - General Provisions", True), ("Title 20 - Corporations", False),
                     ("Chapter 1 - Dissolution", False)]


def test_split_state_code_tax_headings():
    from scripts.measure_domain_yield import TAX_HEADING, TAX_HEADING_EXCLUDE
    text = ("Title 12 - Revenue and Taxation\nChapter 1 - Income Tax\nx\nChapter 9 - Revenue Bonds\ny\n"
            "Title 13 - Highways\nChapter 1 - Toll Revenue Bonds\nz\n")
    flags = [fam for _, fam, _ in split_state_code(text, TAX_HEADING, TAX_HEADING_EXCLUDE)]
    assert flags == [True, True, True, False, False]
