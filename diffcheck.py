import ipdb

noterm = open('data_2/train.tsv').readlines()
terms = open('data_2_terminology/train.tsv').readlines()

diff = list(set(terms) - set(noterm))

ipdb.set_trace()
