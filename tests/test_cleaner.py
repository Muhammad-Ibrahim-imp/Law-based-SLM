from data.cleaner import clean_legal_text


def test_removes_nonbreaking_spaces():
    text = "Exhibit\xa010.2\nINDEMNIFICATION AGREEMENT"
    result = clean_legal_text(text)
    assert "\xa0" not in result
    assert "Exhibit 10.2" in result


def test_removes_page_break_dashes():
    text = (
        "end of clause one\n\n"
        "--------------------------------------------------------------------------------\n\n"
        "start of clause two"
    )
    result = clean_legal_text(text)
    assert "-" * 10 not in result
    assert "end of clause one" in result
    assert "start of clause two" in result


def test_removes_standalone_page_numbers():
    text = "end of page seven\n\n8\n\nstart of page eight"
    result = clean_legal_text(text)
    assert "\n8\n" not in result
    assert "end of page seven" in result
    assert "start of page eight" in result


def test_removes_inline_page_break_markers():
    # Real artifact from a court opinion sample: "Califor*580nia"
    text = "Anders v Califor*580nia, 386 US 738"
    result = clean_legal_text(text)
    assert "*580" not in result
    assert "California" in result


def test_keeps_real_legal_citations_intact():
    # Citations like these are legitimate content and must NOT be stripped
    text = "see Anders v California, 386 US 738; People v Paige, 54 AD2d 631"
    result = clean_legal_text(text)
    assert "386 US 738" in result
    assert "54 AD2d 631" in result


def test_collapses_excess_blank_lines():
    text = "paragraph one\n\n\n\n\nparagraph two"
    result = clean_legal_text(text)
    assert "\n\n\n" not in result
    assert "paragraph one" in result
    assert "paragraph two" in result


def test_collapses_repeated_spaces():
    text = "word1     word2"
    result = clean_legal_text(text)
    assert "  " not in result
    assert "word1 word2" in result


def test_real_contract_excerpt_is_cleaned():
    # Shortened real excerpt from the Atticus contracts sample
    text = (
        "Exhibit\xa010.2\nINDEMNIFICATION AGREEMENT\n"
        "\xa0\xa0\xa0\xa0\xa0This Indemnification Agreement (this \u201cAgreement\u201d) is made "
        "as of the 3rd day\nof February, 2009, by and between Magellan Petroleum "
        "Corporation, a Delaware\ncorporation (the \u201cCompany\u201d).\n\n"
        "\xa0\n\n\n\n--------------------------------------------------------------------------------\n\n\n\n"
        "\xa0\n\n\n\nIndemnitee or on his behalf in connection with such action."
    )
    result = clean_legal_text(text)
    assert "\xa0" not in result
    assert "-" * 10 not in result
    assert "INDEMNIFICATION AGREEMENT" in result
    assert "Indemnitee or on his behalf" in result


def test_real_court_opinion_excerpt_is_cleaned():
    # Shortened real excerpt from the courtlistener_opinions sample
    text = (
        "\u2014Appeal by the defendant from a judgment of the Supreme Court, "
        "Queens County (Brennan, J.), rendered December 1, 1983.\n"
        "Counsel\u2019s application for leave to withdraw as counsel is granted "
        "(see, Anders v Califor*580nia, 386 US 738; People v Paige, 54 AD2d 631)."
    )
    result = clean_legal_text(text)
    assert "*580" not in result
    assert "California" in result
    assert "386 US 738" in result
    assert "54 AD2d 631" in result

# ---------------------------------------------------------------- legal pretraining rules
from data.cleaner import quality_issue  # noqa: E402


def test_unicode_mojibake_and_control_chars():
    text = "42 U.S.C. � 1983 and O�Neil\x0cnext page â€™s"
    out = clean_legal_text(text)
    assert "§ 1983" in out and "O'Neil" in out and "\x0c" not in out and "�" not in out
    assert "’s" in out


def test_star_pagination_but_not_westlaw_pin_cites():
    out = clean_legal_text("held so. [*5] See 2013 WL 8351981, at *2-3; **12 Next")
    assert "[*5]" not in out and "**12" not in out and "at *2-3" in out


def test_html_and_pii_scrub_keep_names():
    out = clean_legal_text("<p>Call John Smith at (555) 123-4567 or jsmith@example.com, SSN 123-45-6789.</p> &amp;")
    assert "<p>" not in out and "&" in out and "John Smith" in out
    assert "[PHONE]" in out and "[EMAIL]" in out and "[SSN]" in out


def test_layout_reflow_hyphenation_headers_and_junk():
    page = "Case: 15-10084 Document: 0051 Page: {} Date Filed: 06/09/2017"
    text = "\n\n".join([page.format(1), "The district court imposed a", "sentence that was well-", "reasoned and fair.",
                       page.format(2), "E |", "No. 16-1317", page.format(3), "lllllllll Plaintiff"]
                      + ["Filler line {} of the opinion.".format("abcdefghijklmnopqrstu"[i] * 3) for i in range(20)])
    out = clean_legal_text(text, layout=True)
    # double-spaced lines re-flowed; a true compound keeps its line-end hyphen
    assert "The district court imposed a sentence that was well-reasoned and fair." in out
    assert out.count("Case: 15-10084") == 1           # running header kept once
    assert "E |" not in out and "No. 16-1317" in out and "lllll" not in out
    assert "128 So. 2d 92 (1961)" in clean_legal_text("128 So. 2d 92 (1961)\nHenry WHITTLE\nv.\nLillian WHITTLE.",
                                                      layout=True)


def test_layout_off_keeps_statute_lines():
    text = "(a) In general.\n(b) Exception.\n(1) first"
    assert clean_legal_text(text) == text


def test_quality_issue_reasons():
    assert quality_issue("Affirmed.") == "too_short"
    assert quality_issue(" ".join(["12 34 56 78"] * 200) + " the of and") in ("low_alpha", "too_short")
    english = "The court held that the defendant was entitled to a hearing on the motion. " * 30
    assert quality_issue(english) is None
    assert quality_issue("Der Beklagte wurde verurteilt weil seine Klage abgewiesen wurde heute " * 30) == "not_english"
    rep = "\n".join(["This line repeats itself again and again."] * 40 + [english])
    assert quality_issue(rep) == "repeated_lines"


def test_line_end_hyphen_kept_for_compounds_dropped_for_split_words():
    assert "well-reasoned" in clean_legal_text("The opinion was well-\nreasoned and short.", layout=True)
    out = clean_legal_text("The claim was dis-\nmissed. Later the appeal was dismissed too.", layout=True)
    assert "The claim was dismissed." in out and "dis-" not in out
