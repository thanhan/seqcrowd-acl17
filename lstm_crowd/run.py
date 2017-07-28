# run LSTM-Crowd
# Example: python run.py -T train.pkl -v val.txt -t test.txt -m 0
import sys
#sys.path.insert(0, 'libs/mod_deep')
sys.path.insert(0, '../../')

import os
import numpy as np
import optparse
import itertools
from collections import OrderedDict
from utils import create_input
import loader

from utils import models_path, evaluate, eval_script, eval_temp
from loader import word_mapping, char_mapping, tag_mapping
from loader import update_tag_scheme, prepare_dataset
from loader import augment_with_pretrained



optparser = optparse.OptionParser()
optparser.add_option(
    "-T", "--train", default="",
    help="Train set location"
)
optparser.add_option(
    "-v", "--dev", default="",
    help="Dev set location"
)
optparser.add_option(
    "-t", "--test", default="",
    help="Test set location"
)
optparser.add_option(
    "-c", "--crowdreg", default="1.0",
    help="regularization for crowd embeddings parameters"
)

optparser.add_option(
    "-r", "--reload", default="0",
    type='int', help="Reload the last saved model"
)


optparser.add_option(
    "-m", "--model", default="0",
    type='int', help="type of model to use 0/2"
)

optparser.add_option(
    "-d", "--cdim", default="10",
    type='int', help="dimension of crowd embedding"
)


opts = optparser.parse_args()[0]

if opts.model == 0:
    from model import Model
else:
    from model2 import Model

parameters = OrderedDict([('tag_scheme', 'iobes'), ('lower', False), ('zeros', False), ('char_dim', 25), ('char_lstm_dim', 25), ('char_bidirect', True), ('word_dim', 100),
                      ('word_lstm_dim', 100), ('word_bidirect', True), ('pre_emb', ''), ('all_emb', False), ('cap_dim', 0), ('crf', True), ('dropout', 0.5),
                      ('lr_method', 'sgd-lr_.005'), ('crowd_dim', int(opts.cdim)), ('n_crowds', 47), ('crowd_reg', float(opts.crowdreg))])

model = Model(parameters=parameters, models_path='./models')
print "Model location: %s" % model.model_path

#opts_train = 'gt_sen.txt'
#opts_dev = '../../data_origin/eng.testb'
#opts_test = '../../gt_sen.txt'
#opts_test = '../../data_origin/eng.testa'

opts_train = opts.train
opts_dev = opts.dev
opts_test = opts.test

lower = parameters['lower']
zeros = parameters['zeros']
tag_scheme = parameters['tag_scheme']

#train_sentences = loader.load_sentences(opts_train, lower, zeros)
#dev_sentences = loader.load_sentences(opts_dev, lower, zeros)
#test_sentences = loader.load_sentences(opts_test, lower, zeros)

#import hmm, util, copy
#hs, hc, all_sen, features, labels = hmm.run_rod()
#res, wid = util.to_lample(hs.data, features, labels)
import pickle, copy
#res, wid = pickle.load(open('lample_train.pkl', 'r'))
res, wid = pickle.load(open(opts_train, 'r'))
train_sentences = copy.deepcopy(res)
dev_sentences = loader.load_sentences(opts_dev, lower, zeros)
test_sentences = loader.load_sentences(opts_test, lower, zeros)



# Use selected tagging scheme (IOB / IOBES)
update_tag_scheme(train_sentences, tag_scheme)
update_tag_scheme(dev_sentences, tag_scheme)
update_tag_scheme(test_sentences, tag_scheme)

dico_words, word_to_id, id_to_word = word_mapping(train_sentences, lower)
dico_words_train = dico_words

# Create a dictionary and a mapping for words / POS tags / tags
dico_chars, char_to_id, id_to_char = char_mapping(train_sentences)
dico_tags, tag_to_id, id_to_tag = tag_mapping(train_sentences)

# Index data
train_data = prepare_dataset(
    train_sentences, word_to_id, char_to_id, tag_to_id, lower
)
dev_data = prepare_dataset(
    dev_sentences, word_to_id, char_to_id, tag_to_id, lower
)
test_data = prepare_dataset(
    test_sentences, word_to_id, char_to_id, tag_to_id, lower
)

# Save the mappings to disk
print 'Saving the mappings to disk...'
model.save_mappings(id_to_word, id_to_char, id_to_tag)


#f_train, f_eval = model.build(**parameters)
index = 1
singletons = set([word_to_id[k] for k, v
                  in dico_words_train.items() if v == 1])
input = create_input(train_data[index], parameters, True, singletons)

s_len = len(input[0])
input.append([1]*s_len)

f_train, f_eval = model.build(**parameters)

# Reload previous model values
if opts.reload:
    print 'Reloading previous model...'
    model.reload()

if True:
    n_epochs = 100  # number of epochs over the training set
    freq_eval = 1000  # evaluate on dev every freq_eval steps
    best_dev = -np.inf
    best_test = -np.inf
    count = 0
    for epoch in xrange(n_epochs):
        epoch_costs = []
        print "Starting epoch %i..." % epoch
        for i, index in enumerate(np.random.permutation(len(train_data))):
            count += 1
            input = create_input(train_data[index], parameters, True, singletons, wid=wid[index])
            new_cost = f_train(*input)
            epoch_costs.append(new_cost)
            if i % 50 == 0 and i > 0 == 0:
                print "%i, cost average: %f" % (i, np.mean(epoch_costs[-50:]))
            if count % freq_eval == 0:
                dev_score = evaluate(parameters, f_eval, dev_sentences,
                                     dev_data, id_to_tag, dico_tags)
                test_score = evaluate(parameters, f_eval, test_sentences,
                                      test_data, id_to_tag, dico_tags)
                print "Score on dev: %.5f" % dev_score
                print "Score on test: %.5f" % test_score
                if dev_score > best_dev:
                    best_dev = dev_score
                    print "New best score on dev."
                    print "Saving model to disk..."
                    model.save()
                if test_score > best_test:
                    best_test = test_score
                    print "New best score on test."
        print "Epoch %i done. Average cost: %f" % (epoch, np.mean(epoch_costs))
