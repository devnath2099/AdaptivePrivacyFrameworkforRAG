import re


def regex_predict(rows, labels):
    patterns = {'EMAIL': r'[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}',
                'PHONE_NUMBER': r'(?<!\w)\+?\d[\d ()-]{7,}\d(?!\w)'}
    allowed = {label[2:] for label in labels if label != 'O'}
    return [[{'start': m.start(), 'end': m.end(), 'type': kind, 'confidence': 1.}
             for kind, pattern in patterns.items() if kind in allowed for m in re.finditer(pattern, row['text'])] for row in rows]
