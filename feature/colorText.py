
import sys
import json
import misc

# class WC: word color
# opinion word: +1:green, -1: red
# opinion target: yellow
class WC:
    YELLOW = '\033[1;33m'
    GREEN = '\033[0;32m'
    RED = '\033[0;31m'
    LIGHT_BLUE = '\033[1;34m'
    NC = '\033[0m' # no color

# ct: coloring text for the words in dictionary
def ct(text, wordColor, removeNewLine=True):
    for word, color in wordColor.items():
        text = text.replace(word, color + '[' + word + ']' + WC.NC) 
    if removeNewLine:
        text = text.replace('\n', ' ')
    return text

# ct2: coloring text of whole text
def ct2(text, color):
    return color + text + WC.NC

# sentiment lexicon to word -> color mapping
def dictToWordColorMapping(sentiDict):
    colorMap = { 1: WC.GREEN, -1: WC.RED }
    wordColorForSent = { word: colorMap[score] for word,score in sentiDict.items() }
    return wordColorForSent

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage:', sys.argv[0], 'sentiDictFile textFile', file=sys.stderr)
        exit(-1)
    
    wordLexiconFile = sys.argv[1]
    textFile = sys.argv[2]

    sentiDict = readSentiDict(wordLexiconFile)
    wordColor = dictToWordColorMapping(sentiDict)

    with open(textFile, 'r') as f:
        for line in f:
            print(ct(line.strip(), wordColor))

