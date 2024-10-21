import langid

# Define your allowed languages
ALLOW_LIST = ['en', 'zh-tw']


def detect_language(text: str) -> str:
    try:
        # Classify the text
        detected_lang, confidence = langid.classify(text)

        # Map detected 'zh' to 'zh-tw' if necessary
        if detected_lang == "zh":
            detected_lang = "zh-tw"

        # Return the language if it's in the allowed list, else default to 'zh-tw'
        if detected_lang in ALLOW_LIST:
            return detected_lang
        else:
            return "zh-tw"
    except Exception as e:
        # Fallback to default language
        return "zh-tw"
