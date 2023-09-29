from collections import defaultdict


d = defaultdict(list)
with open('/home/user/purepython/cpython-3.9/pydyna/store_attr_flowc_spacy.txt') as f:
    for i, line in enumerate(f):
        dic = line.split(':')[0].strip()
        act = line.split(':')[1].strip()
        d[dic].append(act)
        if len(d[dic])%2 == 1:
            assert act == '<add-dict>', f"{i}"
        else:
            assert act == '<take-dict>', f"{i}"

# for dic in d:
#     for i, act in enumerate(d[dic]):
#         if i % 2 == 0:
#             assert act == '<add-dict>'
#         else:
#             assert act == '<take-dict>'
42443739
4805725
32038701
16914990

1915834
784463


2226069
31645980

8693813
25482059

7070082
5036824

13493068
2226683 

1412422
2770404

1432918
2790621