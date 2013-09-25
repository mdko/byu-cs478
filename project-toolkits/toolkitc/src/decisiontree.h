// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef DECISION_TREE_H
#define DECISION_TREE_H

#include "learner.h"
#include "rand.h"
#include "error.h"
#include <vector>
#include <cmath>
#include <string>

#define DEBUG 1

using namespace std;

class DecisionTreeLearner : public SupervisedLearner {

	class Node {
	private:
		Node* parent;
		Matrix& features;
		Matrix& labels;
		int attr_split_on; 	//column/attribute
		int class_split_on; //class of that attribute
		string attr_name;
		string class_name;
		vector<Node*> children;
		double entropy;
		int num_attributes;
		int num_instances;
		int num_labels;
		int num_instances_in_subset;
		int output;
		bool is_leaf;
		

	public:
		Node(Node* parent, int attr, int class_a, Matrix& features, Matrix& labels) :
		   												   	parent(parent),
														    features(features),
														    labels(labels),
															attr_split_on(attr),
															class_split_on(class_a)	{
			if (attr == -1) {
				attr_name = "Root";
				class_name = "";
			}
			else {
				attr_name = features.attrName(attr);
				class_name = features.attrValue(attr_split_on,class_split_on);
			}
			entropy = 0.0;

			num_attributes = features.cols();
			num_instances = features.rows();
			num_labels = labels.valueCount(0);

			num_instances_in_subset = 0;
			output = -1;
			is_leaf = false;

		}
		
		void create_children(int attr, int num_children) {
			for (int class_num = 0; class_num < num_children; class_num++) {
				Node* child = new Node(this, attr, class_num, features, labels);
				children.push_back(child);
			}
		}

		void calculate_entropy() {

			printf("\n***************************\n*** Node %s ***\n***************************\n",
					this->get_name().c_str());
			//To find distribution of classes in label vector, instantiate
			int distr_labels[num_labels];
			for (int i = 0; i < num_labels; i++) {
				distr_labels[i] = 0;
			}
		
			//TODO need to take into account parent's class for their attributes, filter rows
			//where those aren't applicable either!!!
			//review this	
			for (int row = 0; row < num_instances; row++) {
				if (attr_split_on == -1 || 
						(features[row][attr_split_on] == class_split_on
						 && attr_split_on >= 0 )) {
					distr_labels[(int)labels[row][0]]++;
					num_instances_in_subset++;
				}
			}
			
			if (num_instances_in_subset == 0) {
				printf("No instances in set...returning.");
				is_leaf = true;
				printf("Node's output: %d\n", output);
				printf("%s\n", get_name().c_str());
				return;
			}

			if (DEBUG) printf("%s: ", attr_name.c_str());
			int highest_output_class = 0;
			for (int output_class = 0; output_class < num_labels; output_class++) {
				if (DEBUG)
					printf("%d/%d ", distr_labels[output_class], num_instances_in_subset);
				
				if (distr_labels[output_class] > distr_labels[highest_output_class])
					highest_output_class = output_class;
				
				double arg = distr_labels[output_class]/(double)num_instances_in_subset;
				double base = 2.0;
				if (arg == 0.0)
					entropy += 0.0;
				else
					entropy += -(arg * (log(arg) / log(base))); // log(base2)arg
			}
			if (DEBUG) printf(" Entropy for %s: %.14lg\n", 
					class_name.c_str(),
				   	entropy);

			//if (entropy == 0.0)
			//save in case we stop splitting because of lack of more attributes
				output = highest_output_class;
		}	

		void split_on_best_attribute(bool remaining_attr[]) {

			if (DEBUG) {
				printf("Remaining attributes:\n");
				for (int i = 0; i < num_attributes; i++) {
					if (remaining_attr[i] == 1)
						printf(" \"%s\" ", features.attrName(i).c_str());
				}
				printf("\n");
			}

			for (int i = 0; i < num_attributes; i++) {
				if (remaining_attr[i] == 1)
					break;
				if (i == (num_attributes - 1)) {
					printf("No attributes to split on...returning.\n");
					is_leaf = true;
					printf("Node's output: %d\n", output);
					printf("%s\n", get_name().c_str());
					return;
				}
			}

			if (this->entropy == 0.0) {
				printf("Entropy for this attribute is 0...returning.\n");
				is_leaf = true;
				printf("Node's output: %d\n", output);
				printf("%s\n", get_name().c_str());
				return;
			}
		
			int num_in_rem_attrs = 0;
			vector<Attribute*> attr_vector;
			for (int col = 0; col < num_attributes; col++) {
				if (remaining_attr[col] == 1) {
					bool new_remaining_attr[num_attributes];
					for (int i = 0; i < num_attributes; i++)
						new_remaining_attr[i] = remaining_attr[i];
					new_remaining_attr[col] = 0;

					if (DEBUG) printf("\nFinding infoGain for attribute %s, %d.\n",
							features.attrName(col).c_str(), col);
					Attribute* a = new Attribute(features.attrName(col), col);
					//figure out infoGain for possible attributes to split on from this node
					a->compute_infoGain(col, remaining_attr, features, labels, this); 
					a->print_infoGain();
					attr_vector.push_back(a);
					num_in_rem_attrs++;
				}
			}

			//Get best attribute to split on
			int best_a = 0;
			for (int i = 0; i < num_in_rem_attrs; i++) {
				if (attr_vector[i]->get_infoGain() > attr_vector[best_a]->get_infoGain())
					best_a = i;
			}
			printf("\nBest attribute: ");
			attr_vector[best_a]->print_infoGain();
			int best_att_num = attr_vector[best_a]->get_attrNumber();
			int num_classes_for_attr = features.valueCount(best_att_num);
			
			bool new_remaining_attr[num_attributes];
			for (int i = 0; i < num_attributes; i++)
				new_remaining_attr[i] = remaining_attr[i];
			new_remaining_attr[best_att_num] = 0;
	
			//so we know when to stop creating children
			int num_attr_rem = 0;
			for (int i = 0; i < num_attributes; i++) {
				if (new_remaining_attr[i] == 1)
					num_attr_rem++;
			}

			//TODO review next part
			//4/14/13: I took:
			//if there are no more attrs to split on or
			//the entropy for every class in this attribute to split on is 0 (ie dead end)
			/*if (num_attr_rem == 0 || attr_vector[best_a]->get_infoGain() == this->entropy) {
				printf("No attributes to split on...returning.\n");
				is_leaf = true;
				printf("Node's output: %d\n", output);
				printf("%s\n", get_name().c_str());
				return;
			}*/
			//And split it into:
			//
			//There are no children to split on, so this node
			//is a leaf node and so we label it with the most common
			//class of the parent?
			if (num_attr_rem == 0) {
				// label with most common class of parent
				printf("No attributes to split on...returning.\n");
				is_leaf = true;
				printf("Node's output: %d\n", output);
				printf("%s\n", get_name().c_str());
				return;
			}
			//else if (attr_vector[best_a]->get_infoGain() == this->entropy) {
				// create a leaf node for each class of the best_attribute
				
			//	return;
			//}

			attr_split_on = best_att_num;
			create_children(best_att_num, num_classes_for_attr);
			for (int i = 0; i < num_classes_for_attr; i++)
				children[i]->do_it(new_remaining_attr);
		}

		double get_entropy() {
			return entropy;
		}

		string get_name() {
			return attr_name + " " + class_name;
		}

		int get_attribute() {
			return attr_split_on;
		}

		int get_class() {
			return class_split_on;
		}

		int get_num_instances_in_subset() {
			return num_instances_in_subset;
		}

		void do_it(bool remaining_attr[]) {
			this->calculate_entropy();
			this->split_on_best_attribute(remaining_attr);
		}

		Node* get_child(int class_num) {
			return children[class_num];
		}

		int get_output() {
			return output;
		}

		bool is_leaf_node() {
			return is_leaf;
		}

	};

	class Attribute {
	private:
		string att_name;
		string parent_name;
		int attr_number;
		double infoGain;

	public:
		Attribute(string name, int num) : att_name(name),
	   									  attr_number(num),
									  	  infoGain(0) {
		}

		~Attribute() {
		}

		void compute_infoGain(
				int attrCol, 
				bool remaining_attr[], 
				Matrix& features,
				Matrix& labels,
				Node* poss_parent) {

			if (DEBUG)
				printf("Computing infoGain for attribute \"%s\",\n whose parent is \"%s\"\n",
						features.attrName(attrCol).c_str(), poss_parent->get_name().c_str());

			//int num_attributes = features.cols();
			int num_instances = features.rows();
			int num_labels = labels.valueCount(0);
			double entropies[features.valueCount(attrCol)];
			int parent_class = poss_parent->get_class();
			int parent_attr = poss_parent->get_attribute();
			int num_instances_of_parent = poss_parent->get_num_instances_in_subset();
			
			//initialize 2-d array to hold counts for each class for this attr.
			int distr_labels[features.valueCount(attrCol)][num_labels];
			for (int i = 0; i < (int)features.valueCount(attrCol); i++)
				for (int j = 0; j < num_labels; j++) 
					distr_labels[i][j] = 0;
			
			//figure out how many yes/no's belong to each class for this attribute
			//(ie for outlook, sunny has 3 no's and 2 yes's, overcast has 0 no's and ...)
			for (int row = 0; row < num_instances; row++) {
				if (features[row][parent_attr] == parent_class ||
						parent_class == -1) { //only get rows where outlook is sunny
					 distr_labels[(int)features[row][attrCol]][(int)labels[row][0]]++;
				}
			}

			//figure out how many times in the set of instances this class for this
			//attribute occurs (ie in sit. above, sunny has 5 (3 no + 2 yes)
			int total_of_this_class[features.valueCount(attrCol)];
			//for each class in attribute, where the value in the parent's column for this row
			// is equal to the parent's class
			if (DEBUG) printf("Parent attr: %d, class: %d\n", parent_attr, parent_class);
			for (int i = 0; i < (int)features.valueCount(attrCol); i++) {
				total_of_this_class[i] = 0;
				for (int j = 0; j < num_labels; j++) {
						total_of_this_class[i] += distr_labels[i][j];
				}
				if (DEBUG)
					printf("Class %s occurs %d times.\n",
							features.attrValue(attrCol,i).c_str(),
							total_of_this_class[i]);
			}
			
			if (DEBUG) {
				printf("Attribute \"%s\":\n", features.attrName(attrCol).c_str());

				for (int j = 0; j < (int)features.valueCount(attrCol); j++) {
					printf("\tClass: %s: ", features.attrValue(attrCol,j).c_str());
					printf("( ");
					for (int k = 0; k < num_labels; k++) {
						printf("%d/%d ",
							distr_labels[j][k],
							total_of_this_class[j]);
					}
					printf(")\n");
				}
			}
		
			/**
			 * Calculate entropy for each class in this attribute	
			 */
			//for each class for this attribute (ie for outlook: sunny, overcast, rain)
			for (int j = 0; j < (int)features.valueCount(attrCol); j++) {
				//for each label in the output (ie yes/no)
				for (int k = 0; k < num_labels; k++) {
					double arg;
					if ((double)total_of_this_class[j] == 0.0)
						arg = 0.0;
					else	
						arg = distr_labels[j][k]/(double)total_of_this_class[j];
					double base = 2.0;
					if (arg == 0.0)
						entropies[j] += 0.0;
					else 
						entropies[j] += -(arg * (log(arg) / log(base))); // log(base2)arg
				}
				if (DEBUG) {
					printf("Entropy for class %s: %lf\n",
							features.attrValue(attrCol,j).c_str(),
							entropies[j]);
				}
			}

			double summation = 0.0;
			for (int i = 0; i < (int)features.valueCount(attrCol); i++) {
				summation += entropies[i] * total_of_this_class[i];
			}
			double attr_entropy_avg = (summation / num_instances_of_parent);
			if (DEBUG)
				printf("Parent(%s)'s entropy: %lf, attribute %s's avg entropy: %lf\n",
						poss_parent->get_name().c_str(),
						poss_parent->get_entropy(),
						features.attrName(attrCol).c_str(),
						attr_entropy_avg);
			infoGain = poss_parent->get_entropy() - attr_entropy_avg;

		}

		void print_infoGain() {
			printf("Attribute \"%s\"'s infoGain: %lf\n",
				att_name.c_str(), infoGain);
		}

		double get_infoGain() {
			return infoGain;
		}

		int get_attrNumber() {
			return attr_number;
		}
	};

private:
	Rand& m_rand; 
	int num_attributes;
	int num_instances;
	int num_labels;
	Node* root;

public:
	DecisionTreeLearner(Rand& r)
	: SupervisedLearner(), m_rand(r) {
	}

	virtual ~DecisionTreeLearner() {
	}

	virtual void train(Matrix& features, Matrix& labels) {
		/****************
		 * Declarations *
		 ***************/
		bool remaining_attr[num_attributes];
		int num_remaining_attr;

		/******************
		 * Instantiations *
		 ******************/
		num_attributes = features.cols();
		num_instances = features.rows();
		num_labels = labels.valueCount(0);
		num_remaining_attr = num_attributes;
		for (int i = 0; i < num_attributes; i++)
			remaining_attr[i] = 1; // O = used, 1 = unused

		if (DEBUG) {
			printf("NumAttributes: %d\n",num_attributes);
			printf("NumInstances: %d\n",num_instances);
			printf("NumLabels: %d\n",num_labels);
			for (int i = 0; i < num_attributes; i++) {
				printf("Number of classes for attribute \"%s\": %d\n",
						features.attrName(i).c_str(),
					   	(int)features.valueCount(i));

				printf("Names of classes:\n");

				for (int j = 0; j < (int)features.valueCount(i); j++)
					printf("  %s\n", features.attrValue(i,j).c_str());
			}
		}

		// Check assumptions
		//if(features.rows() != labels.rows())
		//	ThrowError("Num rows features unequal to num rows labels");

		// Shuffle the rows. 
		features.shuffleRows(m_rand, &labels);

		// Root node entropy calculation
		root = new Node(NULL, -1, -1, features, labels);
		root->do_it(remaining_attr);

	}

	

	// Evaluate the features and predict the labels
	virtual void predict(const std::vector<double>& features, 
			std::vector<double>& labels) {

		if (DEBUG) {
			printf("Predicting...\n");
			for (int i = 0; i < num_attributes; i++) {
				printf("Attribute \"%d\" value: %lf\n",
					i, features.at(i));
			}
		}
	
		Node* curr = root;
		while (!(curr->is_leaf_node())) {
			int curr_attribute = curr->get_attribute();
			if (DEBUG) printf("Curr attr: %d\n", curr_attribute);
			double actual_class = features.at(curr_attribute);
			if (DEBUG) {
				printf("Node \"%s\", actual class: %d\n",
						curr->get_name().c_str(), (int)actual_class);
			}
			curr = curr->get_child((int)actual_class);
		}
		int predicting_output = curr->get_output();
		if (DEBUG) printf("Predicted output: %d\n", predicting_output);
		labels[0] = predicting_output;

	}
};


#endif // DECISION_TREE_H
