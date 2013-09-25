// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef DECISION_TREE_B_H
#define DECISION_TREE_B_H

#include "learner.h"
#include "rand.h"
#include "error.h"
#include <vector>
#include <string>
#include <assert.h>

#define DEBUG_DTB 0
#define DEBUG_DTB_E 0
#define DEBUG_RRP 0
#define PRINT_TREE_INFO 1
#define PRINT_TREE 0
#define REDUCED_ERROR_PRUNE 1
#define EXPERIMENT 0

using namespace std;

vector<int> majority_value_for_each_attribute;
Matrix validation_features;
Matrix validation_labels;
double best_accuracy_so_far;
int deepest_depth;

SupervisedLearner* learner;

class DecisionTreeBLearner : public SupervisedLearner
{
	
	class Node;

	// Each attribute has m nodes, where m is number of classes for that attribute
	class Attribute
	{
	private:
		Matrix subset_features;
		Matrix subset_labels;
		Node* parent_node;
		vector<bool> attributes_availability;
		int attribute_num;

		string attribute_name;
		int num_classes;
		vector<Node*> nodes_for_each_class;
		double info_gain;

	public:
		Attribute(Matrix& features,
				  Matrix& labels,
				  Node* parent,
				  vector<bool> attr_avail,
				  int attr_num)
		: subset_features(features),
		  subset_labels(labels),
		  parent_node(parent),
		  attributes_availability(attr_avail),
		  attribute_num(attr_num)
		{
			subset_features.copyPart(features, 0, 0, features.rows(), features.cols());
			subset_labels.copyPart(labels, 0, 0, labels.rows(), labels.cols());
			num_classes = subset_features.valueCount(attr_num);
			nodes_for_each_class.resize(0);
			assert((int)nodes_for_each_class.size() == 0);
			if (DEBUG_DTB_E) printf("Number of nodes for this class: %d\n", num_classes);
			attribute_name = subset_features.attrName(attribute_num);
		}

		~Attribute()
		{
		}

		void calculateInfoGain() {
			// Set this attribute as unavailable for future children nodes doing the same thing
			attributes_availability[attribute_num] = 0;
			
			double info_gain_on_attr = 0.0;
			// Assemble the subset of features/labels for each attribute
			if (DEBUG_DTB) printf("Attribute %d-%s:\n", attribute_num, attribute_name.c_str());
			for (int class_num = 0; class_num < num_classes; class_num++) {

				Matrix attr_subset_features(subset_features);
				Matrix attr_subset_labels(subset_labels);
				
				for (int row_num = 0; row_num < (int)subset_features.rows(); row_num++) {
					vector<double>& poss_feature_row_to_copy = subset_features[row_num];
					vector<double>& poss_label_row_to_copy = subset_labels[row_num];

					double class_value = poss_feature_row_to_copy[attribute_num];

					// Dealing with unknown values for this attribute also
					assert(majority_value_for_each_attribute[attribute_num] >= 0);
					int majority_value;
					if (EXPERIMENT_NON_MAJ)
						majority_value = (int)(subset_features.mostCommonValue(attribute_num) + 1) % num_classes;
					else
						majority_value = majority_value_for_each_attribute[attribute_num];
					bool unknown_class = (class_value == UNKNOWN_VALUE);
					bool add_unknown = (unknown_class && (majority_value == class_num));
					
					if (add_unknown || (!unknown_class && (int)class_value == class_num)) {
						if (DEBUG_DTB_E) printf("Copying a row for subset matrix...\n");
						attr_subset_features.copyRow(poss_feature_row_to_copy);
						attr_subset_labels.copyRow(poss_label_row_to_copy);
					}
				}

				// Error checking
				// TODO meta data doesn't seem to be right...
				if (DEBUG_DTB_E) printf("Size of attr_features_subset: %d\nSize of attr_labels_subset: %d\n",
										(int)attr_subset_features.rows(),
										(int)attr_subset_labels.rows());
				for (int col_num = 0; col_num < (int)attr_subset_features.cols(); col_num++) {
					assert(subset_features.valueCount(col_num) == attr_subset_features.valueCount(col_num));
					if (DEBUG_DTB_E) printf ("Number of classes for attribute %d: %d\n", col_num, (int)attr_subset_features.valueCount(col_num));
				}
				
				Node* class_node = new Node(attr_subset_features,
											attr_subset_labels,
											parent_node,
											attributes_availability,
											attribute_num,
											class_num);
											
											
				class_node->calculateEntropyOfThisNode();

				double numerator = attr_subset_features.rows(); // Number of instances with this class as its value for this attribute
				double denominator = subset_features.rows(); 	// Total number of instances in parent (subset matrix passed, which we split right beforehand for each class)
				double class_entropy = class_node->getEntropy();
				
				info_gain_on_attr += ((numerator / denominator) * class_entropy);

				nodes_for_each_class.push_back(class_node);

				if (DEBUG_DTB) { printf("\t"); class_node->printEntropy(); }
			}
			double parent_entropy = parent_node->getEntropy();
			info_gain = parent_entropy - info_gain_on_attr;
			if (DEBUG_DTB) { printf("\t"); printInfoGain(); }
		}

		double getInfoGain() {
			return info_gain;
		}

		int getAttributeNum() {
			return attribute_num;
		}

		string getAttributeName() {
			return attribute_name;
		}

		vector<Node*>& getNodesForEachClass() {
			if (DEBUG_DTB_E) printf("Number of class nodes for this attribute: %d\n",
									(int)nodes_for_each_class.size());
			return nodes_for_each_class;
		}

		void printInfoGain() {
			printf("InfoGain=%lf\n", info_gain); 
		}
	};
	
	class Node
	{
	private:
		Matrix subset_features;
		Matrix subset_labels;
		Node* parent_node;
		vector<bool> attributes_availability;
		int attribute_num;
		int class_num;
		
		double entropy;
		int num_output_classes;
		int num_instances_in_subset;
		vector<int> num_per_output_class;
		string attr_name;
		string class_name;

		vector<Attribute*> child_attributes;
		int num_attributes;
		vector<Node*> children_nodes;
		bool is_leaf;
		int majority_label;
		int attribute_children_split_on; // For traversing down tree during prediction

	public:
		Node(Matrix& features, Matrix& labels, Node* parent, vector<bool> attr_avail, int attr_num, int clas_num)
		: subset_features(features),
		  subset_labels(labels),
		  parent_node(parent),
		  attributes_availability(attr_avail),
		  attribute_num(attr_num),
		  class_num(clas_num)
		{
			assert(features.rows() == labels.rows());
			if (DEBUG_DTB_E) printf("Argument features matrix size: %d\nArgument labels matrix size: %d\n",
										(int)features.rows(),
										(int)labels.rows());

			subset_features.copyPart(features, 0, 0, features.rows(), features.cols());
			subset_labels.copyPart(labels, 0, 0, labels.rows(), labels.cols());
			
			entropy = 0;
			num_output_classes = subset_labels.valueCount(0);					

			assert(subset_features.rows() == subset_labels.rows());
			if (DEBUG_DTB_E) printf("Private features matrix size: %d\nPrivate labels matrix size: %d\n",
										(int)subset_features.rows(),
										(int)subset_labels.rows());
			assert(num_output_classes > 0);

			if (attribute_num == -1) assert(parent_node == NULL);
			else assert(parent_node != NULL);
			
			num_instances_in_subset = subset_features.rows();
			num_attributes = subset_features.cols();
			num_per_output_class.resize(num_output_classes);
			if (attribute_num == -1) {
				attr_name = "Root";
				class_name = "Root";
			}
			else {
				attr_name = subset_features.attrName(attribute_num);
				class_name = subset_features.attrValue(attribute_num, class_num);
			}

			is_leaf = false;
			majority_label = -1;
		}

		~Node()
		{
		}

		void split()
		{
			if (DEBUG_DTB) { printf("Node: "); printEntropy(); }
			if (entropy == 0.0 || !calculateInfoGainForEachAttribute()) // Either our entropy is 0.0, or there aren't any 
				makeLeaf();												// more attributes to split on, make a leaf
			else
				splitOnBestAttribute();	
		}

		//Entropy = Info
		void calculateEntropyOfThisNode() {
			// Get number of entries for each output class
			fill_n(num_per_output_class.begin(), num_output_classes, 0);
			
			if (DEBUG_DTB_E) { printf("Num instances in subset: %d\n", num_instances_in_subset);
							   printf("Num output classes: %d\n", num_output_classes); }
			for (int row = 0; row < num_instances_in_subset; row++) {
				if (DEBUG_DTB_E) printf("Row %d's label: %d\n", row, (int)subset_labels[row][0]);
				num_per_output_class[(int)(subset_labels[row][0])]++;
			}

			// Sum the p(log2(p)) for each class
			for (int class_num = 0; class_num < num_output_classes; class_num++) {
				double numerator = num_per_output_class[class_num];
				double denominator = (double)num_instances_in_subset;
				double p;
				if (denominator)
					p = (numerator / denominator);
				else
					p = 0;
				double base = 2.0;
				double sum;
				if (p)
					sum = p * (log(p) / (log(base)));
				else
					sum = 0;
				entropy += -sum;
			}

			if (DEBUG_DTB_E)
				printf("Entropy for Node %s-%s: %lf\n", attr_name.c_str(),
														class_name.c_str(),
														entropy);
		}

		// return 1 for success, 0 otherwise
		int calculateInfoGainForEachAttribute()
		{
			int num_unavailable = 0;
			for (int attr_num = 0; attr_num < num_attributes; attr_num++)
			{
				// 1 = available, 0 = used already
				if (!((int)attributes_availability[attr_num])) {
					num_unavailable++;
					continue;
				}

				Attribute* a = new Attribute(subset_features,
											 subset_labels,
											 this,
											 attributes_availability,
											 attr_num);
				a->calculateInfoGain();
				child_attributes.push_back(a);
			}

			if (num_unavailable == num_attributes)
				return 0; // can't proceed any longer, so this node is a leaf (will be handled in calling function)

			return 1; // success
		}

		int splitOnBestAttribute()
		{
			Attribute* best_attribute = child_attributes[0];
			for (int attr_num = 0; attr_num < (int)child_attributes.size(); attr_num++) {
				Attribute* a = child_attributes[attr_num];
				double curr_info_gain = a->getInfoGain();
				double best_info_gain = best_attribute->getInfoGain();
				if (curr_info_gain > best_info_gain)
					best_attribute = a;
			}

			attribute_children_split_on = best_attribute->getAttributeNum();

			if (DEBUG_DTB) { printf("Maximum InfoGain=%lf\n", best_attribute->getInfoGain());
							 printf("Split on Attribute %d-%s\n", best_attribute->getAttributeNum(),
																  best_attribute->getAttributeName().c_str()); }

			vector<Node*>& nodes_for_each_class = best_attribute->getNodesForEachClass();
			if (DEBUG_DTB_E) printf("Number of class nodes for this best attribute: %d\n",
									 (int)nodes_for_each_class.size());
			assert(nodes_for_each_class.size() == subset_features.valueCount(best_attribute->getAttributeNum()));
			
			for (int node_num = 0; node_num < (int)nodes_for_each_class.size(); node_num++) {
				Node* curr_node = nodes_for_each_class[node_num];
				children_nodes.push_back(curr_node);
				curr_node->split();
			}
			
			return 1; // success
		}

		double getEntropy() {
			return entropy;
		}

		void calculateOutputLabel() {
			majority_label = 0;
			for (int output_label = 0; output_label < num_output_classes; output_label++) {
				if (num_per_output_class[output_label] > num_per_output_class[majority_label])
					majority_label = output_label;
			}
		}

		void makeLeaf() {
			is_leaf = true;
			calculateOutputLabel();
			if (DEBUG_DTB) { printf("\t"); printAsLeaf(); }
			return;
		}

		void unMakeLeaf() {
			is_leaf = false;
			return;
		}

		bool isLeaf() {
			return is_leaf;
		}

		int getAttributeChildrenSplitOn() {
			return attribute_children_split_on;
		}

		Node* getChildNode(int class_for_split_attribute) {
			for (int child_node = 0; child_node < (int)children_nodes.size(); child_node++) {	
				Node* child = children_nodes[child_node];
				if (child->getClassNum() == class_for_split_attribute)
					return child;
			}
			assert(0); // Should never reach here
		}

		vector<Node*>& getChildren() {
			return children_nodes;
		}

		int getClassNum() {
			return class_num;
		}

		int getMajorityLabel() {
			return majority_label;
		}

		void printEntropy() {
			printf("Value %d-%s: (", class_num, class_name.c_str());
			for (int output_class = 0; output_class < num_output_classes; output_class++) {
				printf(" %d/%d ", num_per_output_class[output_class], num_instances_in_subset);
			}
			printf(") Entropy=%lf\n", entropy);
		}

		void printAsLeaf() {
			printf("***Leaf node \'%d-%s\' output: %d (%s)\n",
					class_num,
					class_name.c_str(),
					majority_label,
					subset_labels.attrValue(0, majority_label).c_str());
		}

		void printAsInnerNode() {
			printf("Inner node \'%d-%s\'...Split on Attr %d-%s\n",
					 class_num,
					 class_name.c_str(),
					 attribute_children_split_on,
					 subset_features.attrName(attribute_children_split_on).c_str());
		}

		void printNodeName() {
			printf("Node %d-%s. Leaf? %s\n", class_num,
											 class_name.c_str(),
											 (is_leaf) ? "yes" : "no");
		}

		void printTree(int generation) {
			// Print this node
			string ast(generation,'\t');
			if (generation) printf("%s",ast.c_str());
			generation++;
			if (is_leaf)
				printAsLeaf();
			else {
				printAsInnerNode();
				// Call print on all its children
				if (generation - 1) printf("%s",ast.c_str());
				printf("(Children of %s:)\n", class_name.c_str());
				for (int child_node = 0; child_node < (int)children_nodes.size(); child_node++) {
					Node* child = children_nodes[child_node];
					child->printTree(generation);
				}
			}
		}

	};
	
private:
	Rand& m_rand;
	Node* root;

public:
	DecisionTreeBLearner(Rand& r)
	: SupervisedLearner(), m_rand(r)
	{
		learner = this;
		root = NULL;
	}

	virtual ~DecisionTreeBLearner()
	{
	}

	virtual void train(Matrix& features, Matrix& labels)
	{
		//Shuffle before we split into training and validation sets
		features.shuffleRows(m_rand, &labels);
		
		// Create validation, training set
		int total_num_instances = (int)features.rows();
		int num_attributes = (int)features.cols();
		int number_rows_in_training_set = total_num_instances * 0.8;
		int number_rows_in_validation_set = total_num_instances - number_rows_in_training_set;
		
		Matrix training_features(features);
		Matrix training_labels(labels);
		//validation_features(features);
		//validation_labels(labels);
		best_accuracy_so_far = -1.0;
		
		// Initialize training set
		training_features.copyPart(features, 0, 0, number_rows_in_training_set, num_attributes);
		training_labels.copyPart(labels, 0, 0, number_rows_in_training_set, labels.cols());
		
		// Initialize validation set
		validation_features.copyPart(features, number_rows_in_training_set, 0, number_rows_in_validation_set, num_attributes);
		validation_labels.copyPart(labels, number_rows_in_training_set, 0, number_rows_in_validation_set, labels.cols());

		// For dealing with unknown values (could have just used features.mostCommonValue(col)...)
		majority_value_for_each_attribute.resize(training_features.cols());
		fill_n(majority_value_for_each_attribute.begin(), training_features.cols(), -1);
		for (int col_num = 0; col_num < (int)training_features.cols(); col_num++) {

			int num_classes_for_feature = training_features.valueCount(col_num);
			int classes_for_feature[num_classes_for_feature];
			fill_n(classes_for_feature, num_classes_for_feature, 0);

			// Calculate sums for each class in this attribute
			for (int row_num = 0; row_num < (int)training_features.rows(); row_num++) {
				double clas = training_features[row_num][col_num];
				if (clas != UNKNOWN_VALUE)
					classes_for_feature[(int)clas]++;
			}

			// Find mode class for this attribute
			int mode_value = 0;
			for (int class_num = 0; class_num < num_classes_for_feature; class_num++) {
				if (classes_for_feature[class_num] > classes_for_feature[mode_value])
					mode_value = class_num;
			}
			majority_value_for_each_attribute[col_num] = mode_value;
		}

		// Start training tree
		vector<bool> attributes_avail(training_features.cols());
		fill_n(attributes_avail.begin(), training_features.cols(), 1);
		
		root = new Node(training_features, training_labels, NULL, attributes_avail, -1, -1);
		
		root->calculateEntropyOfThisNode();
		root->split();
		// Tree is now fully trained (empty or consistent partitions, or no more attributes)
		double original_tree_accuracy_on_vs = 0.0;
		if (PRINT_TREE_INFO) {
			printf("**************\nOriginal Tree:\n");
			if (PRINT_TREE) root->printTree(0);
			original_tree_accuracy_on_vs = measureAccuracy(validation_features, validation_labels, false, NULL);
			printf("Original tree accuracy on validation set: %lf\n", original_tree_accuracy_on_vs);
			int total_num_nodes = 0;
			countNodes(root, total_num_nodes);
			printf("Number of nodes in original tree: %d\n", total_num_nodes);
			deepest_depth = 0;
			getTreeDepth(root, 0);
			printf("Depth of original tree: %d\n", deepest_depth);
		}

		if (REDUCED_ERROR_PRUNE) {
			reduceErrorPrune(original_tree_accuracy_on_vs);
			if (PRINT_TREE_INFO) {
				printf("************\nPruned Tree:\n");
				if (PRINT_TREE) root->printTree(0);
				double pruned_tree_accuracy_on_vs = measureAccuracy(validation_features, validation_labels, false, NULL);
				assert(pruned_tree_accuracy_on_vs >= original_tree_accuracy_on_vs);
				printf("Pruned tree accuracy on valiation set: %lf\n", pruned_tree_accuracy_on_vs);
				int total_pruned_num_nodes = 0;
				countNodes(root, total_pruned_num_nodes);
				printf("Number of nodes in pruned tree: %d\n", total_pruned_num_nodes);
				deepest_depth = 0;
				getTreeDepth(root, 0);
				printf("Depth of pruned tree: %d\n", deepest_depth);
			}
		}
	}

	virtual void predict(const std::vector<double>& features, std::vector<double>& labels)
	{
		Node* curr_node = root;
		while (!(curr_node->isLeaf())) {
			if (DEBUG_DTB) curr_node->printNodeName();
			int split_attribute = curr_node->getAttributeChildrenSplitOn();
			double actual_class_for_split_attribute = features[split_attribute];
			if (actual_class_for_split_attribute == UNKNOWN_VALUE)
				actual_class_for_split_attribute = majority_value_for_each_attribute[split_attribute];
			curr_node = curr_node->getChildNode(actual_class_for_split_attribute);
		}
		
		int predicting_output = curr_node->getMajorityLabel();
		labels[0] = predicting_output;
	}

	void reduceErrorPrune(double orig_tree_accuracy_on_vs) {
		
		double best_accuracy_so_far = orig_tree_accuracy_on_vs;
		while (1) {
			Node* best_node_with_subtree_to_prune = NULL;

			vector<Node*> non_leaf_nodes;
			getNonLeafNodes(root, non_leaf_nodes);

			if (DEBUG_RRP) { for (int node_n = 0; node_n < (int)non_leaf_nodes.size(); node_n++)
										non_leaf_nodes[node_n]->printNodeName(); }

			// For each non-leaf node
			for (int non_leaf_node = 0; non_leaf_node < (int)non_leaf_nodes.size(); non_leaf_node++) {
				Node* non_leaf_node_with_subtree_to_prune = non_leaf_nodes[non_leaf_node];
				non_leaf_node_with_subtree_to_prune->makeLeaf(); 	// "Prune" it

				// Find accuracy on tree if this node's subtree was pruned (ie if it was a leaf)
				double pruned_accuracy = measureAccuracy(validation_features, validation_labels, false, NULL);
				if (pruned_accuracy >= best_accuracy_so_far) {
					// Remember node whose pruning improves the accuracy on validation set
					best_accuracy_so_far = pruned_accuracy;
					best_node_with_subtree_to_prune = non_leaf_node_with_subtree_to_prune;
				}

				non_leaf_node_with_subtree_to_prune->unMakeLeaf();	// "Un-prune" it, so next pruning can use entire tree
			}
			// If we iterated through every non-leaf node and found that by removing
			// any of them resulting in a pruned_accuracy that's worse than the best
			// accuracy so far, further pruning is harmful, so stop here
			if (best_node_with_subtree_to_prune == NULL)
				break;
			// Else prune node in tree that gives us the best resulting accuracy
			best_node_with_subtree_to_prune->makeLeaf();
		}
	}

	int getNonLeafNodes(Node* curr_node, vector<Node*>& non_leaf_nodes) {

		// Stopping criteria for recursion
		if (curr_node->isLeaf())
			return 0;

		vector<Node*>& child_nodes = curr_node->getChildren();
		for (int child_node = 0; child_node < (int)child_nodes.size(); child_node++) {
			Node* child = child_nodes[child_node];
			getNonLeafNodes(child, non_leaf_nodes);
		}
		non_leaf_nodes.push_back(curr_node);

		return 0;
	}

	void countNodes(Node* curr_node, int& total_num_nodes) {

		// Count both inner and leaf nodes
		total_num_nodes++;
		
		if (curr_node->isLeaf()) {
			return;
		}
		
		vector<Node*>& child_nodes = curr_node->getChildren();
		for (int child_node = 0; child_node < (int)child_nodes.size(); child_node++) {
			Node* child = child_nodes[child_node];
			countNodes(child, total_num_nodes);
		}
	}

	void getTreeDepth(Node* curr_node, int depth) {
		depth++;
		if (curr_node->isLeaf()) {
			if (depth > deepest_depth)
				deepest_depth = depth;
			return;
		}

		vector<Node*>& child_nodes = curr_node->getChildren();
		for (int child_node = 0; child_node < (int)child_nodes.size(); child_node++) {
			Node* child = child_nodes[child_node];
			getTreeDepth(child, depth);
		}
		
	}
};

#endif // DECISION_TREE_B_H
