import re 

def is_frag(word):
    if re.search(r'===', word):
        return True
    else:
        return False

def is_punct(word):
    if re.search(r'\.|,|:|·', word):
        return True
    else:
        return False

def is_digit(word):
    if re.match(r'~.*~', word):
        return True
    else:
        return False
