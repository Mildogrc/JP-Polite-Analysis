from transformers import AutoTokenizer

t1 = AutoTokenizer.from_pretrained('cl-tohoku/bert-base-japanese')
t2 = AutoTokenizer.from_pretrained('cl-tohoku/bert-large-japanese')

vocab1 = t1.get_vocab()
vocab2 = t2.get_vocab()

if vocab1 == vocab2:
    print("Vocabs are identical")
else:
    print("Vocabs are DIFFERENT")
