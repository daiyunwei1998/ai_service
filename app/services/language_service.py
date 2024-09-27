import langid


def detect_language(text: str) -> str:
    try:
        return langid.classify(text)
    except:
        return "zh-tw"