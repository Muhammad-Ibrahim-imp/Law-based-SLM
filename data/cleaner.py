import re

def clean_legal_text(text: str) -> str:
    # Replace non-breaking spaces with normal spaces
    text = text.replace("\xa0", " ")

    # Remove page-break separator lines
    text = re.sub(r"[-–—]{10,}", "", text)

    # Remove standalone page-number lines
    text = re.sub(r"\n\s*\d+\s*\n", "\n", text)

    # Remove inline page-break markers like *580 (asterisk + digits, no space)
    text = re.sub(r"\*\d+", "", text)

    # Collapse repeated whitespace
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)

    # normalizing smart quotes to straight quotes
    # text = re.sub(r"[\u201C\u201D]", '"', text) # double
    # text = re.sub(r"[\u2018\u2019]", "'", text) # single

    # em-dash — / en-dash – vs plain hyphen -
    # ellipsis … vs three periods ...
    # apostrophe-like punctuation such as ′
    
    return text.strip()