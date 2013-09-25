/**
 * Author: Michael Christensen
 * Date: March 21, 2013
 *
 * For BYU CS 478 Instance-based Learning Project
 */

#ifndef INSTANCEBASED_H
#define INSTANCEBASED_H

#include "learner.h"
#include "rand.h"
#include "error.h"
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <map>
#include <vector>
#include <queue>

#define DEBUG_IB 1
#define NUM_K_NEIGHBORS 11
#define DISTANCE_WEIGHTED 0
#define CNN 1
#define UV_VALUE 0.75

using namespace std;

class InstanceBasedLearner : public SupervisedLearner
{
	
private:
	Rand& m_rand; // pseudo-random number generator (not actually used by the baseline learner)
	int num_classes;
	int num_training_examples;
	bool nominal;
	Matrix training_examples;
	Matrix training_labels;
	vector<std::pair<double, double> > low_high_list;

public:
	InstanceBasedLearner(Rand& r)
	: SupervisedLearner(), m_rand(r)
	{
	}

	virtual ~InstanceBasedLearner()
	{
	}

	virtual void train(Matrix& features, Matrix& labels)
	{
		/*******************
		 * Initializations *
		 *******************/
		num_classes = labels.valueCount(0);
		nominal = (num_classes) ? true : false;
		num_training_examples = features.rows();

		if (DEBUG_IB) {
			printf("Number of classes: %d\n", num_classes);
			printf("Nominal or discrete: %s\n", (nominal ? "nominal" : "discrete"));
			printf("Number of neighbors: %d\n", NUM_K_NEIGHBORS);
			printf("Weighted? %s\n", (DISTANCE_WEIGHTED ? "Y" : "N"));
		}
		
		// Training Algorithm:
		//
		// For each training example (x, f(x)), add the example to the list training_examples
		if (CNN) { 
			// Condensed Nearest Neighbor Rule
			int num_training_ex_superset = features.rows();
			num_training_examples = 0;
			int num_features = features.cols();
			int num_labels = labels.cols();
			training_examples.setSize(0, num_features);
			training_labels.setSize(0, num_labels);
			int num_rand_instances_selected = 0;
			bool classes_selected[num_classes];
			std::fill(classes_selected, classes_selected + num_classes, 0);
			// While we haven't added all the representative instances for each class
			//  or we're discrete labels, so just add one
			while (num_rand_instances_selected < num_classes || (!nominal)) {
				int ran_instance = (int)m_rand.next(num_training_ex_superset);
				vector<double> possible_row = features[ran_instance];
				vector<double> possible_row_label = labels[ran_instance];

				if (!nominal) {
					training_examples.copyRow(possible_row);
					training_labels.copyRow(possible_row_label);
					num_training_examples++;
					break;
				}
				else { // else nominal
					int label_val = (int)possible_row_label[0];	
					
					// If we haven't added an instance for this class, add it
					if (classes_selected[label_val] == 0) {
						classes_selected[label_val] = 1; // Note we've added it
						num_rand_instances_selected++;
						training_examples.copyRow(possible_row);
						training_labels.copyRow(possible_row_label);
						num_training_examples++;
					}
				}
			}

			for (int instance = 0; instance < num_training_ex_superset; instance++) {
				vector<double> training_instance = features[instance];
				vector<double> training_instance_label = labels[instance];
				vector<double> predicted_label;
				predicted_label.resize(1);
				predict(training_instance, predicted_label);
				if (training_instance_label[0] != predicted_label[0]) {
					training_examples.copyRow(training_instance);
					training_labels.copyRow(training_instance_label);
					num_training_examples++;
				}
			}
			if (DEBUG_IB) printf("Num inst in training subset: %d\n", num_training_examples);
			// T = training matrix
			// S = subset of training matrix (to fill via this algorithm)
			// Randomly select one instance belonging to each output class T, put it in S
			// For each instance in T, classify them (use "predict(...)")
			//  using S as training set, T as "test" set
			//  If an instance in T is misclassified, add it to S
			//  Stop when no more instances in T are misclassified
			// training_examples.copyPart(subset_te, 0, 0 ...)
			// training_labels.copyPart(...)
		}
		else {
			training_examples.copyPart(features, 0, 0, features.rows(), features.cols());
			training_labels.copyPart(labels, 0, 0, labels.rows(), labels.cols());
		}
	}

	virtual void predict(const std::vector<double>& features, std::vector<double>& labels)
	{
		// Local variables
		int num_features = features.size();
		// Each entry is a distance,label pair
		std::priority_queue<typename std::pair<double, double>,
							typename std::vector<std::pair<double, double> >, 
							cmp_dl_pairs> 
																distances;
		if (DEBUG_IB) {
			printf("**************************************************************\n");
			printf("Num inst in training set: %d\n", num_training_examples);
		}
				
		// Find the distances from this instance to every other instance of training_examples
		// and push them into the priority queue so we can get the k closest
		for (int train_ex = 0; train_ex < num_training_examples; train_ex++) { // for each exam.
			double distance_summation = 0.0;
			double distance = 0.0;
			vector<double> training_example = training_examples[train_ex];
			for (int feature = 0; feature < num_features; feature++) { // for each feature
				double inter_distance = 0.0;
				double current_feature = features[feature];
				double train_feature = training_example[feature];
				if (current_feature == UNKNOWN_VALUE || train_feature == UNKNOWN_VALUE)
					//inter_distance = UV_VALUE;
					inter_distance = (train_ex > 0) ? // For experiment 
						(training_examples[train_ex - 1][feature] == UNKNOWN_VALUE ?
							UV_VALUE : training_examples[train_ex - 1][feature]) : UV_VALUE;
				else {
					bool nominal_feature = training_examples.valueCount(feature) > 0;

					// if set of values for this particular feature is nominal
					if (nominal_feature) // was just if (nominal(class))
						inter_distance = ((int)current_feature == (int)train_feature) ? 0 : 1; //TODO this probably should have 0 and 1 switched
					else 
						inter_distance = current_feature - train_feature;
				}
				distance_summation += pow(inter_distance, 2);
			}
			distance = sqrt(distance_summation);
			std::pair <double, double> distance_label_pair;
			distance_label_pair = std::make_pair(distance, training_labels[train_ex][0]);
			distances.push(distance_label_pair);
		}
		
		if (DEBUG_IB) {
			printf("Instance to classify's attributes: ");
			for (int i = 0; i < num_features; i++) { printf("%lf ", features[i]); }
			printf("\nClosest %d training examples' distances from this feature: \n",
					NUM_K_NEIGHBORS);
		}
		
		// Depending on whether this is 1) nominal or 2) discrete, find the 1) most common label
		//  (mode) or 2) mean discrete-valued label, respectively, among the k closest training 
		//  examples
		vector<int> tallies(num_classes);					// for nominal (unweighted)
		std::fill_n(tallies.begin(), num_classes, 0);
		vector<double> summations_for_classes(num_classes); // for nominal (weighted, plus above)
		std::fill_n(summations_for_classes.begin(), num_classes, 0);
		double summation = 0.0; 			  	// for discrete (both weighted and unweighted)
		double denominator_summation = 0.0; 	// for discrete weighted, in addition to above
		for (int k = 0; k < NUM_K_NEIGHBORS; k++) {
			if (k >= num_training_examples)
				break;

			std::pair<double, double> dl_pair = distances.top();
			distances.pop();
			
			// For distance weighting formula, if used later (in both nominal and discrete)
			double denom_squared = pow(dl_pair.first /*distance*/, 2);
			if (denom_squared == 0) { 
				denom_squared = 0.01; /*TODO Do something*/
				if (DEBUG_IB) printf("This instance's distance to example is 0.\n");
			}

			denominator_summation += denom_squared;
			double weighted_dist = 0.0;

			if (nominal) {
				if (DISTANCE_WEIGHTED) {
					weighted_dist = 1.0 / denom_squared;
					summations_for_classes[(int)(dl_pair.second)] += weighted_dist;
				}
				else
					tallies[(int)(dl_pair.second)]++;
			} // else discrete-valued
			else {
				if (DISTANCE_WEIGHTED) {
					weighted_dist = dl_pair.second /*label*/ / denom_squared;
					summation += weighted_dist;
				}
				else
					summation += dl_pair.second;
			}
	
			if (DEBUG_IB) {	
				printf(" Closest example %d's%sdistance: %lf\n", 
						k, 
						(DISTANCE_WEIGHTED ? " weighted " : " "), 
						(DISTANCE_WEIGHTED ? weighted_dist : (dl_pair.first))); 
				if (nominal) 
					printf("  That distance's nominal class label: %d\n", (int)(dl_pair.second));
				else 
					printf("  That distance's discrete class label: %lf\n",
							dl_pair.second);
			}
		}

		if (nominal) {
			int modal_label = 0;
			if (DEBUG_IB) printf("Resulting %s array:\n", 
					(DISTANCE_WEIGHTED ? "summations" : "tallies"));
			for (int clas = 0; clas < num_classes; clas++) {
				if (DISTANCE_WEIGHTED) {
					if (summations_for_classes[clas] > summations_for_classes[modal_label])
						modal_label = clas;
					if (DEBUG_IB) printf("Label %d: %lf\n", clas, summations_for_classes[clas]);
				}
				else {
					if (tallies[clas] > tallies[modal_label])
						modal_label = clas;
					if (DEBUG_IB) printf("Label %d: %d\n", clas, tallies[clas]);
				}
			}
			labels[0] =  modal_label;
			if (DEBUG_IB) printf("Modal label for nominal class: %d\n", (int)labels[0]);
		}
		else {
			double label = 0.0;
			if (DISTANCE_WEIGHTED) {
				label = (summation / denominator_summation);
				if (DEBUG_IB) printf("Discrete summation: %lf, denominator summation: %lf\n",
						summation, denominator_summation);
			}
			else {
				double denom = (num_training_examples >= NUM_K_NEIGHBORS)
						? NUM_K_NEIGHBORS : num_training_examples;
				label = (summation / denom);
				if (DEBUG_IB) printf("Discrete summation: %lf, denom: %lf\n",
						summation, denom);
			}
			labels[0] = label;
			if (DEBUG_IB) printf("Discrete value for label: %lf\n", labels[0]);
		}
	}	

	struct cmp_dl_pairs {
		bool operator()(const std::pair<double, double>& a, 
							const std::pair<double, double>& b) const {
			return a.first > b.first;
		}
	};	
};



#endif // INSTANCEBASED_H
