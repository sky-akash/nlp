SOS_token = 0
EOS_token = 1


def messy_text(text: str):
    """Generates a messy version of a text
    """
    vocabulary = sorted(list(set(text)))
    substitutions = reversed(vocabulary)
    charmap = dict(zip(vocabulary, substitutions))
    return "".join([charmap[x] for x in text])


class Lang:
    def __init__(self, name):
        self.name = name
        self.letter2index = {}
        self.letter2count = {}
        self.index2letter = {0: "SOS", 1: "EOS"}
        self.n_letters = 2

    def add_word(self, word):
        for c in word:
            self.add_letter(c)

    def add_letter(self, letter):
        if letter not in self.letter2index:
            self.letter2index[letter] = self.n_letters
            self.letter2count[letter] = 1
            self.index2letter[self.n_letters] = letter
            self.n_letters += 1
        else:
            self.letter2count[letter] += 1


