import re


def normalize_paragraphs(text: str) -> str:
    if text is None:
        return ""
    text = str(text)

    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

