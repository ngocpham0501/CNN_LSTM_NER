import argparse

ALL_LABELS = ['O', 'B-LOC', 'I-LOC', 'B-MISC', 'I-MISC', 'B-ORG', 'I-ORG', 'B-PER', 'I-PER']

parser = argparse.ArgumentParser(description='Hybrid LSTM-CNN with CRF for Vietnamese NER')
parser.add_argument('-i', help='Job identity', type=int, default=0)

parser.add_argument('-e', help='Number of epochs', type=int, default=20)
parser.add_argument('-p', help='Patience of early stop (0 for ignore early stop)', type=int, default=5)

parser.add_argument('-w2v', help='Use word2vec embeddings', type=int, default=1)
parser.add_argument('-char', help='Use char embeddings (0|cnn|lstm)', type=str, default='0')
parser.add_argument('-pos', help='POS tag embedding dimensions', type=int, default='0')

parser.add_argument('-cnn', help='CNN configurations', type=str, default='1:32,3:64,5:64,7:32')
parser.add_argument('-cnnh', help='CNN hidden layers', type=int, default='0')

parser.add_argument('-lstm', help='Number of output LSTM dimension', type=str, default='256,256')

parser.add_argument('-hd', help='Hidden layer configurations', type=str, default='256,128')

parser.add_argument('-el', help='Use extra loss', type=int, default=1)
parser.add_argument('-crf', help='Use CRF', type=int, default=0)

opt = parser.parse_args()
print('Running opt: {}'.format(opt))

# PARSED CONFIGS
JOB_IDENTITY = opt.i

EPOCHS = opt.e
EARLY_STOPPING = False if opt.p == 0 else True
PATIENCE = opt.p

USE_W2V = False if opt.w2v == 0 else True
INPUT_W2V_DIM = 300

CHAR_EMBEDDING = opt.char
NCHARS = 200

USE_POS = False if opt.pos == 0 else True
POS_EMBEDDING_DIM = opt.pos
NPOSES = 39

USE_LSTM = False if opt.lstm == '0' else True
OUTPUT_LSTM_DIMS = list(map(int, opt.lstm.split(','))) if opt.lstm != '0' else []

USE_CNN = False if opt.cnn == '0' else True
CNN_FILTERS = {
    int(k): int(f) for k, f in [i.split(':') for i in opt.cnn.split(',')]
} if opt.cnn != '0' else {}
CNN_HIDDEN_LAYERS = opt.cnnh

HIDDEN_LAYERS = list(map(int, opt.hd.split(','))) if opt.hd != '0' else []

USE_CRF = False if opt.crf == 0 else True
USE_EXTRA_LOSS = False if opt.el == 0 else True

# DATA CONSTANT
DATA = 'data/vlsp/'
RAW_DATA = DATA + 'raw_data/'
PARSED_DATA = DATA + 'parsed_data/'
PICKLE_DATA = DATA + 'pickle/'

ALL_WORDS = PARSED_DATA + 'all_words.txt'
ALL_CHARS = PARSED_DATA + 'all_chars.txt'
ALL_POSES = PARSED_DATA + 'all_poses.txt'

W2V_DATA = DATA + 'w2v_model/'
TRIMMED_W2V = W2V_DATA + 'trimmed_w2v.npz'

TRAINED_MODELS = DATA + 'trained_models/'
MODEL_NAMES = TRAINED_MODELS + '{}_{}'
