import os

all_words = []
all_poses = []
all_labels = []

DIR = 'data/vlsp/raw_data'
for file_name in os.listdir(DIR):
    with open(DIR + '/' + file_name) as f:
        lines = f.readlines()

        for l in lines[4:]:
            try:
                sl = l.strip().split('\t')
                if len(sl) > 0 and sl[0] not in {'<s>', '</s>', ''}:
                    all_words.append(sl[0])
                    all_poses.append(sl[1])
                    all_labels.append(sl[3])
                    if sl[0] == '' or sl[1] == '' or sl[3] == '':
                        print('line', l)
            except:
                print('error with line', l)

vocab = list(set(all_words))
vocab.sort()
poses = list(set(all_poses))
poses.sort()
labels = list(set(all_labels))
labels.sort()

print(labels)

with open('data/vlsp/parsed_data/all_words.txt', 'w') as f:
    for w in vocab:
        f.write('{}\n'.format(w))

    f.write('$UNK$')

with open('data/vlsp/parsed_data/all_poses.txt', 'w') as f:
    for w in poses:
        f.write('{}\n'.format(w))

alphabet = list(set(''.join(vocab)))
alphabet.sort()

with open('data/vlsp/parsed_data/all_chars.txt', 'w') as f:
    for c in alphabet:
        f.write('{}\n'.format(c))

    f.write('$UNK$')

