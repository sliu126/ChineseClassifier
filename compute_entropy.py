import sys
import pickle
from collections import Counter
import math
from tqdm import tqdm
import numpy as np

def calculate_entropy(type_counter):
	total_count = sum(type_counter.values())
	result = 0.0

	for _type in type_counter:
		prob = type_counter[_type] * 1.0 / total_count
		result += (-prob * math.log(prob, 2))

	return result

def calculate_conditional_entropy(conditional_type_counters, conditioned_type_counter):
	conditioned_type_total_count = sum(conditioned_type_counter.values())
	result = 0.0

	for _type in conditioned_type_counter:
		conditional_type_counter = conditional_type_counters[_type] \
									if _type in conditional_type_counters \
									else Counter()

		prob_type = conditioned_type_counter[_type] * 1.0 / conditioned_type_total_count
		result += prob_type * calculate_entropy(conditional_type_counter)

	return result

# calculate the average entropy of classifiers over nouns
def calculate_avg_H_classifier(classifier_on_noun_counters):
	num_noun_total = len(classifier_on_noun_counters)
	result = 0.0
	for noun, classifier_counter in tqdm(classifier_on_noun_counters.items()):
		result += calculate_entropy(classifier_counter)

	result /= num_noun_total

	return result

# calculate the mutual information between adjectives and classifiers
def calculate_I_adj_classifier(adj_counter, classifier_counter, adj_on_classifier_counters):

	adj_total_count = sum(adj_counter.values())
	classifier_total_count = sum(classifier_counter.values())

	result = 0.0
	classifier_prob_list = []
	inner_sum_list = []
	# calculate the MI as $\sum_{c}p(c)\sum_{a}p(a|c)\log\frac{p(a|c)}{p(a)}$
	for classifier in tqdm(classifier_counter):
		conditional_adj_counter = adj_on_classifier_counters[classifier] \
									if classifier in adj_on_classifier_counters \
									else Counter()

		adj_count = sum(conditional_adj_counter.values())
		if adj_count == 0:
			p_c = classifier_counter[classifier] * 1.0 / classifier_total_count
			classifier_prob_list.append(p_c)
			inner_sum_list.append(0.0)
			continue
		inner_sum = 0.0
		for adj in adj_counter:
			prob_a_on_c = conditional_adj_counter[adj] * 1.0 / adj_count
			prob_a = adj_counter[adj] * 1.0 / adj_total_count
			if prob_a_on_c > 0.0:
				inner_sum += prob_a_on_c * math.log(prob_a_on_c / prob_a, 2)

		# add the sum to the outer sum loop
		p_c = classifier_counter[classifier] * 1.0 / classifier_total_count
		result += p_c * inner_sum
		classifier_prob_list.append(p_c)
		inner_sum_list.append(inner_sum)

	# Sanity Check
	H_adj = calculate_entropy(adj_counter)
	H_adj_on_classifier = calculate_conditional_entropy(adj_on_classifier_counters, classifier_counter)
	print("H(A)=%.4f, H(A|C)=%.4f, H(A)-H(A|C)=%.4f, our calculated I(A,C)=%.4f" % \
		(H_adj, H_adj_on_classifier, H_adj - H_adj_on_classifier, result))

	return result, classifier_prob_list, inner_sum_list

def calculate_I_confidence_interval(prob_list, inner_sum_list, confidence_level=0.95, num_bootstrap=10000):
	prob_list = np.array(prob_list)
	inner_sum_list = np.array(inner_sum_list)

	num_elements = len(prob_list)
	quantile_low = 0.5 * (1.0 - confidence_level) * 100.0
	quantile_high = quantile_low + confidence_level * 100.0

	result_list = []
	for idx_bootstrap in range(num_bootstrap):
		list_idx_selected = np.random.choice(
			a = num_elements, size = num_elements, replace = True, p = prob_list 
		)

		result = np.sum(inner_sum_list[list_idx_selected]) / num_elements
		result_list.append(result)

	result_list.sort()
	result_list = np.array(result_list)

	interval_low = np.percentile(a = result_list, q = quantile_low)
	interval_high = np.percentile(a = result_list, q = quantile_high)

	interval_mean = np.mean(result_list)

	print("%d%% confidence interval is (%.4f, %.4f)" % (100 * confidence_level, interval_low, interval_high))
	return (interval_low, interval_high)

# build counters for input data for easy computing
def build_counters(noun_adj_counter, noun_classifier_counter, noun_adj_classifier_counter):
	# (adj, count)
	adj_counter = Counter()
	# (classifier, count)
	classifier_counter = Counter()
	# (classifier, (adj, count))
	adj_on_classifier_counters = {}
	# (noun, (classifier, count))
	classifier_on_noun_counters = {}
	# (adj, (classifier, count))
	classifier_on_adj_counters = {}

	for noun, classifier in noun_classifier_counter:
		if noun not in classifier_on_noun_counters:
			classifier_on_noun_counters[noun] = Counter()
		classifier_on_noun_counters[noun][classifier] += 1

	for noun, adj, classifier in noun_adj_classifier_counter:
		if classifier not in adj_on_classifier_counters:
			adj_on_classifier_counters[classifier] = Counter()
		adj_on_classifier_counters[classifier][adj] += 1
		adj_counter[adj] += 1
		classifier_counter[classifier] += 1
		if adj not in classifier_on_adj_counters:
			classifier_on_adj_counters[adj] = Counter()
		classifier_on_adj_counters[adj][classifier] += 1

	print("Done building counters")
	return adj_counter, classifier_counter, adj_on_classifier_counters, \
			classifier_on_noun_counters, classifier_on_adj_counters


if __name__ == '__main__':
	noun_adj_counter = pickle.load(open("data/all_noun_adj.pkl", "rb"))
	noun_classifier_counter = pickle.load(open("data/all_noun_classifier.pkl", "rb"))
	noun_adj_classifier_counter = pickle.load(open("data/all_noun_adj_classifier.pkl", "rb"))

	adj_counter, classifier_counter, adj_on_classifier_counters, \
	classifier_on_noun_counters, classifier_on_adj_counters = \
		build_counters(noun_adj_counter, noun_classifier_counter, noun_adj_classifier_counter)


	if sys.argv[1] == '-I': # mutual_information between adjectives and classifiers
		I_adj_classifier, classifier_prob_list, inner_sum_list = \
			calculate_I_adj_classifier(adj_counter, classifier_counter, adj_on_classifier_counters)
		calculate_I_confidence_interval(classifier_prob_list, inner_sum_list)
	elif sys.argv[1] == '-H': # average (over nouns) entropy of classifiers
		avg_H_classifier = calculate_avg_H_classifier(classifier_on_noun_counters)
		print("Average entropy of classifiers over nouns is %.4f" % (avg_H_classifier))





