import pickle

import data.vlsp.constants as constants
from utils import get_trimmed_w2v_vectors, Timer
from model import CnnLstmCrfModel
from sklearn.metrics import precision_recall_fscore_support
from dataset import Dataset, parse_data


def main():
    t = Timer()
    t.start('Load data')
    raw_train = Dataset(*parse_data())
    validation, train = raw_train.one_vs_nine()
    test = validation
    t.stop('Load data')

    # get pre trained embeddings
    embeddings = get_trimmed_w2v_vectors(constants.TRIMMED_W2V)

    model = CnnLstmCrfModel(
        model_name=constants.MODEL_NAMES.format('ner', constants.JOB_IDENTITY),
        embeddings=embeddings,
        batch_size=128,
        constants=constants,
    )

    # train, evaluate and interact
    model.build()
    model.load_data(train=train, validation=validation)
    # model.load_data(train=raw_train, validation=test)
    model.run_train(epochs=constants.EPOCHS, early_stopping=constants.EARLY_STOPPING, patience=constants.PATIENCE)
    # model.run_train(epochs=0, early_stopping=False, patience=0)

    t.start('testing')
    y_pred = model.predict_on_test(test)
    preds = []
    labels = []
    for pred, label in zip(y_pred, test.labels):
        labels.extend(label)
        preds.extend(pred[:len(label)])

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average=None)
    print("precision of each class:\t{}\t{}".format(constants.JOB_IDENTITY, p))
    print("recall of each class:\t{}\t{}".format(constants.JOB_IDENTITY, r))
    print("f1 of each class:\t{}\t{}".format(constants.JOB_IDENTITY, f1))

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro')
    print("macro average result:\t{}\t{}\t{}\t{}".format(constants.JOB_IDENTITY, p, r, f1))

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='macro', labels=list(range(1, len(constants.ALL_LABELS))))
    print("macro average (exclude 0) result:\t{}\t{}\t{}\t{}".format(constants.JOB_IDENTITY, p, r, f1))

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='micro')
    print("micro average result:\t{}\t{}\t{}\t{}".format(constants.JOB_IDENTITY, p, r, f1))

    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average='micro', labels=list(range(1, len(constants.ALL_LABELS))))
    print("micro average (exclude 0) result:\t{}\t{}\t{}\t{}".format(constants.JOB_IDENTITY, p, r, f1))
    t.stop('testing')


if __name__ == '__main__':
    main()
