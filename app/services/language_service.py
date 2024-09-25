from langdetect import detect


def detect_language(text: str) -> str:
    try:
        return "zh-tw"#detect(text)
    except:
        return "zh-tw"