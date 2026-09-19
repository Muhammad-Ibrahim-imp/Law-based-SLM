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