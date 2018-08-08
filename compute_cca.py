# -*- coding: utf-8 -*-
import sys
import pickle
from collections import Counter
from tqdm import tqdm
from sklearn.cross_decomposition import CCA
import numpy as np

# Load English word embeddings
# Return a map with words as keys and vectors as values
def load_word_embeddings(embed_filename, dictionary):
	'''
	embed_file = open(embed_filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
	num, dim = map(int, embed_file.readline().split())
	embed_data = {}
	for line in embed_file:
		tokens = line.rstrip().split(' ')
		if tokens[0] in dictionary.values():
			embed_data[tokens[0]] = list(map(float, tokens[1:]))

	pickle.dump(embed_data, open("word_emb.pkl", "wb"))
	'''
	embed_data = pickle.load(open(embed_filename, "rb"))
	return embed_data

# Make a Chinese-to-English dictionary on nouns from a given dictionary file
def make_dictionary(dict_filename):
	dict_file = open(dict_filename, 'r')
	dictionary = {}
	for line in dict_file.readlines():
		words = line.split()
		dictionary[words[1].decode("utf-8")] = words[0]
	return dictionary

# Translate a list of Chinese nouns into English
def translate_eng(nouns, dictionary):
	nouns_cmn_eng = {}
	for noun in nouns:
		if noun in dictionary:
			nouns_cmn_eng[noun] = dictionary[noun]

	return nouns_cmn_eng

# Compute CCA paramters based on training data
# Calculate correlation coefficient on dev/test data
def cca_analysis(X, Y, X_dev, Y_dev):
	cca = CCA(n_components=1, max_iter=2000)
	cca.fit(X, Y)
	X_dev_c, Y_dev_c = cca.transform(X_dev, Y_dev)

	corrcoef = np.corrcoef(X_dev_c.T, Y_dev_c.T)[0,1]

	return corrcoef

# Generate CCA input data
def make_cca_input(classifier_on_noun_counters, classifier_counter,
					adj_on_noun_counters, adj_counter,
					dictionary, word_emb):
	nouns = set(classifier_on_noun_counters.keys()) & set(adj_on_noun_counters.keys())
	nouns_cmn_eng = translate_eng(nouns, dictionary)

	# Create 3 lists: noun embeddings, distribution of classifier for each noun,
	# distribution of adjectives for each noun
	nouns_emb = []
	classifier_on_noun_dists = []
	adj_on_noun_dists = []

	for noun_cmn in tqdm(nouns_cmn_eng):
		noun_eng = nouns_cmn_eng[noun_cmn]
		if noun_eng in word_emb:
			noun_emb = word_emb[noun_eng]
			nouns_emb.append(noun_emb)

			cond_classifier_counter = classifier_on_noun_counters[noun_cmn]
			classifier_count = sum(cond_classifier_counter.values())
			classifier_dist = []
			for classifier in classifier_counter:
				prob = (cond_classifier_counter[classifier]) * 1.0 / classifier_count
				classifier_dist.append(prob)
			classifier_on_noun_dists.append(classifier_dist)

			cond_adj_counter = adj_on_noun_counters[noun_cmn]
			adj_count = sum(cond_adj_counter.values())
			adj_dist = []
			for adj in adj_counter:
				prob = (cond_adj_counter[adj]) * 1.0 / adj_count
				adj_dist.append(prob)
			adj_on_noun_dists.append(adj_dist)
	
	return nouns_emb, classifier_on_noun_dists, adj_on_noun_dists


# Entry routine for CCA
def perform_cca_on_data(classifier_on_noun_counters_train, classifier_counter, 
						classifier_on_noun_counters_dev,
						adj_on_noun_counters_train, adj_counter, 
						adj_on_noun_counters_dev,
						dictionary, word_emb):

	nouns_emb_train, classifier_on_noun_dists_train, adj_on_noun_dists_train = \
		make_cca_input(classifier_on_noun_counters_train, classifier_counter,
						adj_on_noun_counters_train, adj_counter,
						dictionary, word_emb)

	nouns_emb_dev, classifier_on_noun_dists_dev, adj_on_noun_dists_dev = \
		make_cca_input(classifier_on_noun_counters_dev, classifier_counter,
						adj_on_noun_counters_dev, adj_counter,
						dictionary, word_emb)

	print("Done making CCA input. Start performing CCA...")

	corr_noun_classifier = cca_analysis(nouns_emb_train, classifier_on_noun_dists_train, 
							nouns_emb_dev, classifier_on_noun_dists_dev)
	corr_noun_adj = cca_analysis(nouns_emb_train, adj_on_noun_dists_train,
					nouns_emb_dev, adj_on_noun_dists_dev)
	corr_adj_classifier = cca_analysis(classifier_on_noun_dists_train, adj_on_noun_dists_train,
							classifier_on_noun_dists_dev, adj_on_noun_dists_dev)

	return corr_noun_classifier, corr_noun_adj, corr_adj_classifier

# build counters for input data for easy computing
def build_counters(noun_adj_counter, noun_classifier_counter, noun_adj_classifier_counter):
	# (adj, count)
	adj_counter = Counter()
	# (classifier, count)
	classifier_counter = Counter()
	# (noun, (classifier, count))
	classifier_on_noun_counters = {}
	# (noun, (adjective, count))
	adj_on_noun_counters = {}
	# (adj, count)
	adj_with_classifier_counter = Counter()

	for noun, adj in noun_adj_counter:
		adj_counter[adj] += 1
		if noun not in adj_on_noun_counters:
			adj_on_noun_counters[noun] = Counter()
		adj_on_noun_counters[noun][adj] += 1

	for noun, classifier in noun_classifier_counter:
		classifier_counter[classifier] += 1
		if noun not in classifier_on_noun_counters:
			classifier_on_noun_counters[noun] = Counter()
		classifier_on_noun_counters[noun][classifier] += 1

	for noun, adj, classifier in noun_adj_classifier_counter:
		adj_with_classifier_counter[adj] += 1

	print("Done building counters.")

	return adj_counter, classifier_counter, \
			adj_on_noun_counters, classifier_on_noun_counters, adj_with_classifier_counter


if __name__ == '__main__':
	noun_adj_counter_train = pickle.load(open("data/train_noun_adj.pkl", "rb"))
	noun_classifier_counter_train = pickle.load(open("data/train_noun_classifier.pkl", "rb"))
	noun_adj_classifier_counter_train = pickle.load(open("data/train_noun_adj_classifier.pkl", "rb"))
	noun_adj_counter_dev = pickle.load(open("data/dev_noun_adj.pkl", "rb"))
	noun_classifier_counter_dev = pickle.load(open("data/dev_noun_classifier.pkl", "rb"))
	noun_adj_classifier_counter_dev = pickle.load(open("data/dev_noun_adj_classifier.pkl", "rb"))
	dictionary = make_dictionary("data/eng-cmn_noun.dms")
	word_emb = load_word_embeddings("data/word_emb.pkl", dictionary)

	adj_counter_train, classifier_counter_train, \
	adj_on_noun_counters_train, classifier_on_noun_counters_train, adj_with_classifier_counter_train = \
		build_counters(noun_adj_counter_train, noun_classifier_counter_train, noun_adj_classifier_counter_train)

	adj_counter_dev, classifier_counter_dev, \
	adj_on_noun_counters_dev, classifier_on_noun_counters_dev, adj_with_classifier_counter_dev = \
		build_counters(noun_adj_counter_dev, noun_classifier_counter_dev, noun_adj_classifier_counter_dev)

	corr_noun_classifier, corr_noun_adj, corr_adj_classifier = \
						perform_cca_on_data(classifier_on_noun_counters_train, classifier_counter_train, 
											classifier_on_noun_counters_dev, 
											adj_on_noun_counters_train, adj_with_classifier_counter_train, 
											adj_on_noun_counters_dev,
											dictionary, word_emb)


	print("correlation coefficient between noun embeddings and classifier distribution over nouns is %.4f"
		  % (corr_noun_classifier))
	print("correlation coefficient between noun embeddings and adjective distribution over nouns is %.4f"
		  % (corr_noun_adj))	
	print("correlation coefficient between classifier and adjective distribution over nouns is %.4f"
		  % (corr_adj_classifier))

