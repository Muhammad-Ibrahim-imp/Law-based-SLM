import numpy as np
import pytest

from data.domains import TERM_LIST, classify, profile_for, term_counts

OPINION_FAMILY = ("In this dissolution of marriage action the trial court awarded the wife alimony, "
                  "divided the marital property and set child support. ") * 3
OPINION_CRIMINAL = ("Defendant was convicted of a felony after a guilty plea; the prosecutor sought "
                    "a longer sentence and the court denied his motion to suppress. ") * 3
OPINION_TAX = ("The Commissioner of Internal Revenue issued a notice of deficiency for unreported "
               "gross income under the Internal Revenue Code. ") * 3


@pytest.mark.parametrize("text,expected", [
    (OPINION_FAMILY, {"family"}),
    (OPINION_CRIMINAL, {"criminal"}),
    (OPINION_TAX, {"tax"}),
    ("The lease required thirty days notice before the landlord could enter. " * 3, set()),
    (OPINION_FAMILY + OPINION_CRIMINAL, {"family", "criminal"}),
])
def test_long_profiles(text, expected):
    assert classify(text, "strict") == expected
    assert classify(text, "loose") == expected


def test_strict_needs_three_distinct_terms():
    text = "The divorce decree mentioned child support. " + "Unrelated contract text. " * 10
    assert classify(text, "strict") == set()
    assert classify(text, "loose") == {"family"}


def test_real_estate_tax_is_not_estate_tax():
    hits = lambda t: {TERM_LIST[i][1] for i in np.flatnonzero(term_counts(t))}  # noqa: E731
    assert hits("real estate taxes were assessed") == set()
    assert hits("the federal estate tax was due") == {"estate tax"}


@pytest.mark.parametrize("text,expected", [
    ("My ex won't let me see my kids even though we have joint custody. What can I do?", {"family"}),
    ("Going through a divorce, does she get half my 401k?", {"family"}),
    ("I got arrested for shoplifting at Walmart, first offense. Will I go to jail?", {"criminal"}),
    ("Got a DUI last weekend, my court date is next month. Do I need a lawyer?", {"criminal"}),
    ("The IRS says I owe money on my tax return from 2019.", {"tax"}),
    ("My landlord kept my security deposit for normal wear and tear.", set()),
    # one weak term alone is not enough
    ("My spouse and I want to buy a house together.", set()),
    ("The police towed my car from my own driveway.", set()),
    ("My father had a cardiac arrest while driving and hit a car.", set()),
    # multi-label
    ("My ex was arrested for domestic violence and now wants full custody.", {"family", "criminal"}),
])
def test_short_profile(text, expected):
    assert classify(text, "short") == expected


def test_profile_for_and_bad_profile():
    assert profile_for("short question") == "short"
    assert profile_for("x" * 5000) == "loose"
    with pytest.raises(ValueError):
        classify("text", "medium")
