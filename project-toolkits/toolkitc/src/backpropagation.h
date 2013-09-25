/**
 * Author: Michael Christensen
 * Date: February 15, 2013
 *
 * For BYU CS 478 Backpropagation Project
 */

#ifndef BACKPROPAGATION_H
#define BACKPROPAGATION_H

#include "learner.h"
#include "rand.h"
#include "error.h"
#include <cmath>	
#include <iostream>
#include <stdio.h>
#include <map>
#include <vector>

#define DEBUG_BP 0
#define DEBUG_BP_E 1
#define INFO_OUT 1 // For printing final results

// Adjust these to find best combination
#define INCLUDE_MOMENTUM 1
#define MOMENTUM_TERM 0.875				// was 0.875 (best)
#define LEARNING_RATE 0.05 				// comment out for exper. (see L157)
#define NUM_HIDDEN_LAYERS 2				// not sure if 1 or 2 best 
#define NUM_HIDDEN_NODE_MULTIPLIER 8	// was 16 (best)
#define IGNORE_LAST_ATTRIBUTES 0		// number of columns from the end to ignore (eg pic #)

#define EULER_NUM 2.71828182845904523536028747135266249775724709369995
#define NON_IMPROVEMENT_THRESHOLD 50

// For experimentation
#define INCLUDE_SMALL_DELTA_MOMENTUM 0
#define SMALL_DELTA_TERM 0.05
#define CHANGE_LEARNING_RATE 0
#define LR_MULT 1.13

using namespace std;

class BackpropagationLearner : public SupervisedLearner
{
	class Node 
	{
	private:
		Rand& m_r;
		double my_output;
		double small_delta_weight;
		double previous_small_delta_weight;
		vector<double> big_delta_weights; //These point upwards
		vector<double> previous_big_delta_weights;
		vector<double> output_weights; //These point upwards
		vector<double> best_weights;
		int num_output_weights;
		
	public: 
		Node(int now, Rand& r) : m_r(r)
		{
			//Initializations
			my_output = 0;
			small_delta_weight = 0;
			previous_small_delta_weight = 0;
			
			//Set all weights going out of this node to random num, mean 0
			num_output_weights = now;
			for (int i = 0; i < num_output_weights; i++) {
				double ran_num = ((double)m_r.next(10) / 10) - .5;
				output_weights.push_back(ran_num);
				best_weights.push_back(ran_num);
				big_delta_weights.push_back(0);
				previous_big_delta_weights.push_back(0);
			}
		}
		
		~Node()
		{
		}
		
		/*******************
		 * Getters/setters *
		 *******************/
		void set_output(double output) {
			my_output = output;
		}
		
		double get_output() {
			//TODO problem here, called in predict L562, when using more than 1 hidden layer
			return my_output;
		}
		
		int get_num_output_weights() {
			return num_output_weights;
		}
		
		double get_weight(int to_which_node) {
			return output_weights[to_which_node];
		}
		
		void set_small_delta_weight(double sdw) {
			small_delta_weight = sdw;
		}
		
		double get_small_delta_weight() {
			return small_delta_weight;
		}
		
		double get_previous_small_delta() {
			return previous_small_delta_weight;
		}
		
		void remember_previous_small_delta_weight() {
			previous_small_delta_weight = small_delta_weight;
		}
		
		void set_big_delta_weight(int pos, double bdw) {
			big_delta_weights[pos] = bdw;
		}
		
		double get_previous_big_delta_weight(int pos) {
			return previous_big_delta_weights[pos];
		}
		
		void remember_previous_weights() {
			for (int i = 0; i < num_output_weights; i++)
				previous_big_delta_weights[i] = big_delta_weights[i];
		}
		
		void update_weights() {
			for (int i = 0; i < num_output_weights; i++)
				output_weights[i] += big_delta_weights[i];
		}
		
		void update_best_weights() {
			for (int i = 0; i < num_output_weights; i++)
				best_weights[i] = output_weights[i];
		}
		
		void use_best_weights() {
			for (int i = 0; i < num_output_weights; i++)
				output_weights[i] = best_weights[i];
		}
		
		void clear_delta_weights() {
			for (int i = 0; i < num_output_weights; i++)
				big_delta_weights[i] = 0;
		}
		
		void print_weights() {
			cout << "\tWeights: ";
			for (int i = 0; i < num_output_weights - 1; i++)
				cout << output_weights[i] << ", ";
			cout << output_weights[num_output_weights - 1] << endl;
		}
	};
	
	
	
private:
	Rand& m_rand; // pseudo-random number generator (not actually used by the baseline learner)
	vector<Node*> input_nodes;
	vector <vector<Node*> > hidden_nodes_matrix;
	vector<Node*> output_nodes;
	int num_classes;
	double learning_rate = LEARNING_RATE; // for experimentation

public:
	BackpropagationLearner(Rand& r)
	: SupervisedLearner(), m_rand(r)
	{
	}

	virtual ~BackpropagationLearner()
	{
	}

	virtual void train(Matrix& total_features, Matrix& total_labels)
	{		
		/****************
		 * Declarations *
		 ****************/
		int total_num_instances = total_features.rows();
		int num_features = total_features.cols() - IGNORE_LAST_ATTRIBUTES;
		int num_hidden_nodes = num_features * NUM_HIDDEN_NODE_MULTIPLIER;
		num_classes = !total_labels.valueCount(0) ? 2 : total_labels.valueCount(0); 
		double best_accuracy_so_far = 0.0;
		int number_without_improvement = 0;
		int number_epochs_completed  = 0;
		Matrix validation_features;
		Matrix validation_labels;
		Matrix features; //training
		Matrix labels; //training

		if (INFO_OUT) {
			cout << "Momentum Term: " << MOMENTUM_TERM << endl;
			cout << "Learning Rate: " << learning_rate << endl;
			cout << "Num Hidden Layers: " << NUM_HIDDEN_LAYERS << endl;
			cout << "Num Hidden Node Mult: " << NUM_HIDDEN_NODE_MULTIPLIER << endl;
			cout << "Num Instances: " << total_num_instances << endl;
			cout << "Num Features: " << num_features << endl;
			cout << "Num Hidden Nodes: " << num_hidden_nodes << endl;
			cout << "Num Output Classes: " << num_classes << endl;
		}

		if (DEBUG_BP)
			for (int i = 0; i < total_num_instances; i++)
				cout << total_features.attrValue(num_features, /*attr*/
										   total_features[i][num_features] /*value*/) << endl;

		if (NUM_HIDDEN_LAYERS < 1) {
			cout << "Error: number of hidden layers is less than 1 . . . returning" << endl;
			return;
		}
		
		/*******************
		 * Initializations *
		 *******************/
		//Initialize input nodes (which inits their weights in constructor)
		for (int i = 0; i < num_features + 1; i++) { //+ 1 for bias
			input_nodes.push_back(new Node(num_hidden_nodes, m_rand));
		}
		
		//Initialize hidden nodes (which inits their weights in constructor)
		hidden_nodes_matrix.resize(NUM_HIDDEN_LAYERS);
		for (int i = 0; i < NUM_HIDDEN_LAYERS; i ++) {
			for (int j = 0; j < num_hidden_nodes + 1; j++) { //+ 1 for bias
				if (i == (NUM_HIDDEN_LAYERS - 1)) //each node in topmost layer of hidden nodes only has output weights to # of output nodes
					hidden_nodes_matrix[i].push_back(new Node(num_classes, m_rand));
				else
					hidden_nodes_matrix[i].push_back(new Node(num_hidden_nodes, m_rand));
			}
		}
		
		//Initialize output nodes (they have no weights pointing upwards, so 0 param)
		for (int i = 0; i < num_classes; i++) {
			output_nodes.push_back(new Node(0, m_rand));
		}
		
		//Initialize training, validation sets
		int number_rows_in_training_set = total_num_instances * 0.8;
		int number_rows_in_validation_set = total_num_instances - number_rows_in_training_set;
		if (DEBUG_BP)  {
			cout << "Total rows: " << total_num_instances << endl;
			cout << "Number rows in ts: " << number_rows_in_training_set << endl;
			cout << "Number rows in vs: " << number_rows_in_validation_set << endl;
		}
		
		//Shuffle before we split into training and validation sets
		total_features.shuffleRows(m_rand, &total_labels);
		
		//Initialize test set
		features.copyPart(total_features, 0, 0, number_rows_in_training_set, num_features);
		labels.copyPart(total_labels, 0, 0, number_rows_in_training_set, total_labels.cols());
		
		//Initialize validation set
		validation_features.copyPart(total_features, number_rows_in_training_set, 0, number_rows_in_validation_set, num_features);
		validation_labels.copyPart(total_labels, number_rows_in_training_set, 0, number_rows_in_validation_set, total_labels.cols());

		if (DEBUG_BP_E) {
			cout << "Instances in test set: " << endl;
			for (int i = 0;  i < (int)features.rows(); i++) {
				for (int j = 0; j < (int)features.cols(); j++) {
					cout << features[i][j] << " ";
				}
				cout << endl;
			}

			cout << "Instances in validation set: " << endl;
			for (int i = 0;  i < (int)validation_features.rows(); i++) {
				for (int j = 0; j < (int)validation_features.cols(); j++) {
					cout << validation_features[i][j] << " ";
				}
				cout << endl;
			}
		}
		
		//So we don't have to change all the references in following code to num_instances
		int num_instances = number_rows_in_training_set;
		
		if (DEBUG_BP) {
			cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;
			for (int i = 0; i < (int)features.rows(); i++) {
				for (int j = 0; j < (int)features.cols(); j++) {
					cout << features[i][j] << ", ";
				}
				cout << endl;
			}
			cout << "++++++++++++++" << endl;
			for (int i = 0; i < (int)labels.rows(); i++) {
				for (int j = 0; j < (int)labels.cols(); j++) {
					cout << labels[i][j] << ", ";
				}
				cout << endl;
			}
			cout << "&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&" << endl;
			for (int i = 0; i < (int)validation_features.rows(); i++) {
				for (int j = 0; j < (int)validation_features.cols(); j++) {
					cout << validation_features[i][j] << ", ";
				}
				cout << endl;
			}
			cout << "++++++++++++++" << endl;
			for (int i = 0; i < (int)validation_labels.rows(); i++) {
				for (int j = 0; j < (int)validation_labels.cols(); j++) {
					cout << validation_labels[i][j] << ", ";
				}
				cout << endl;
			}
		}
		/*****************
		 * Run algorithm *
		 *****************/
		if (INFO_OUT) cout << "Epoch Number, Training Set Acc, Validation Set Acc, MSE" << endl;
		do {
			if (DEBUG_BP) cout << "********************" << endl;
			//Shuffle before each epoch
			features.shuffleRows(m_rand, &labels);
			double mean_squared_error = 0.0;
			
			//Start an epoch
			for (int i = 0; i < num_instances; i++) {
				
				//Print out some information for debugging purposes
				if (DEBUG_BP) {
					cout << "++++++++++++++++++++++\nInstance: " << i << endl;
				    cout << "Weights at beginning of instance: " << endl;
				
					for (int i = 0; i < num_features + 1; i++) {
						cout << "Input node " << i << ":" << endl;
						input_nodes[i]->print_weights();
					}
					
					for (int i = 0; i < NUM_HIDDEN_LAYERS; i ++) {
						for (int j = 0; j < num_hidden_nodes + 1; j++) {
							cout << "Hidden layer " << i << ", node " << j << ":" << endl;
							hidden_nodes_matrix[i][j]->print_weights();
						}
					}
				}
				
				/*****************************
				 * Go up toward output layer *
				 *****************************/
				 
				//Set input nodes's outputs
				for (int j = 0; j < num_features; j++) {
					input_nodes[j]->set_output(features[i][j]);
					if (DEBUG_BP) cout << "Input feature in node: " << input_nodes[j]->get_output() << endl;
				}
				input_nodes[num_features]->set_output(1); //Bias node
				 
				//Set ouputs for each node in hidden layer based on nodes below it
				//For each hidden layer
				for (int j = 0; j < NUM_HIDDEN_LAYERS; j++) {
					
					//For each node in that layer, determine its output (so don't do this for bias nodes on end)
					for (int k = 0; k < num_hidden_nodes; k++) {
						
						double net = 0;
						if (DEBUG_BP) cout << "NET: " << net << endl;
						
						//If we're on the first hidden layer, get outputs and weights from input layer
						if (j == 0) {
							//Get output and weight from each node below that points to this one
							for (int l = 0; l < num_features + 1; l++) {
								if (DEBUG_BP) cout << "Inputs' outputs: " << input_nodes[l]->get_output() << ", inputs' weights for that output: " << input_nodes[l]->get_weight(k) << endl;
								net += input_nodes[l]->get_output() * input_nodes[l]->get_weight(k);
							}
						}
						else {
							//Get output and weight from each node in hidden layer below
							for (int l = 0; l < num_hidden_nodes + 1; l++) {
								net += hidden_nodes_matrix[j - 1][l]->get_output() * hidden_nodes_matrix[j - 1][l]->get_weight(k);
							}
						}
						double output = 1 / (1 + pow(EULER_NUM,-net));
						hidden_nodes_matrix[j][k]->set_output(output);
						
						if (DEBUG_BP) cout << "Net for hidden layer " << j << ", hidden node " << k << ": " << net << endl;
						if (DEBUG_BP) cout << "Its output: " << hidden_nodes_matrix[j][k]->get_output() << endl;
					}
					hidden_nodes_matrix[j][num_hidden_nodes]->set_output(1); //Bias node
				}
				
				//Set output nodes' outputs
				for (int j = 0; j < num_classes; j++) {
					double net = 0;
					//Get output and weights for each node in top-most layer of hidden nodes layer
					for (int l = 0; l < num_hidden_nodes + 1; l++) {
						net += hidden_nodes_matrix[NUM_HIDDEN_LAYERS - 1][l]->get_output() * hidden_nodes_matrix[NUM_HIDDEN_LAYERS - 1][l]->get_weight(j);
					}
					double output = 1 / (1 + pow(EULER_NUM,-net));
					output_nodes[j]->set_output(output);
				}
				
				/***********************************
				 * Go down back toward input layer *
				 ***********************************/
				 //Set small deltas for each output node
				 for (int j = 0; j < num_classes; j++) {
					 double target_output = (labels[i][0] == j) ? 1 : 0;
					 double my_output = output_nodes[j]->get_output();
					 
					 if (i == (num_instances - 1))
						mean_squared_error += pow((target_output - my_output), 2);
						
					 double small_delta = (target_output - my_output) * my_output * (1 - my_output);
					 
					 if (DEBUG_BP) {
						cout << "Target for output node " << j << ": " << target_output << endl;
						cout << "Current output for output node " << j << ": " << my_output << endl;
						cout << "Small delta for output node " << j << ": " << small_delta << endl;
					 }
					 
					 if (INCLUDE_SMALL_DELTA_MOMENTUM) {
						 small_delta += output_nodes[j]->get_previous_small_delta() * SMALL_DELTA_TERM;
					 }
					 
					 output_nodes[j]->set_small_delta_weight(small_delta);
				 }
				 if (i == (num_instances - 1))
					mean_squared_error = mean_squared_error / num_classes;
				 
				 //Set big delta and small delta for each node in each layer of hidden nodes
				 for (int j = NUM_HIDDEN_LAYERS - 1; j >= 0; j--) {
					 
					//For each node in this layer of hidden nodes, including bias, change big delta weight
					for (int k = 0; k < num_hidden_nodes + 1; k++) {
							
						Node* hidden_node = hidden_nodes_matrix[j][k];
						//Set big delta for each weight it points up to
						//By adding momentum, we look to previous iteration's weight update to know how much to update this one
						for (int l = 0; l < hidden_node->get_num_output_weights(); l++) {
							double small_delta = 0;
							
							if (j == (NUM_HIDDEN_LAYERS - 1)) //if we're in last hidden layer, small delta we need is in output layer list
								small_delta = output_nodes[l]->get_small_delta_weight();
							else
								small_delta = hidden_nodes_matrix[j + 1][l]->get_small_delta_weight(); //else get sdw from node above me in another hidden layer
							
							double big_delta = learning_rate * small_delta * hidden_node->get_output();
							if (INCLUDE_MOMENTUM)
								big_delta += MOMENTUM_TERM * hidden_node->get_previous_big_delta_weight(l);
								
							if (DEBUG_BP) {
								cout << "Small delta for node " << l << " in layer above: " << small_delta << endl;
								cout << "Big delta for hidden node " << k << " in layer " << j << " to node " << l << " in layer above: " << big_delta << endl;
							}
							
							hidden_node->set_big_delta_weight(l, big_delta);
						}
					}
						
					//For each node in this layer, excluding bias, change small delta weight
					for (int k = 0; k < num_hidden_nodes; k++) {
							
						Node* hidden_node = hidden_nodes_matrix[j][k];
						//And set this node's small delta for use by lower layers
						//(Could probably combine these for loops with above for loops, but want to maintain readability)
						if (DEBUG_BP) cout << "Now setting hidden node " << k << "'s small delta weight." << endl;
						double net = 0;
						for (int l = 0; l < hidden_node->get_num_output_weights(); l++) {
							double small_delta = 0;
							
							if (j == (NUM_HIDDEN_LAYERS - 1))
								small_delta = output_nodes[l]->get_small_delta_weight();
							else
								small_delta = hidden_nodes_matrix[j + 1][l]->get_small_delta_weight(); 
							
							double weight_to_upper_node = hidden_node->get_weight(l);
							net += small_delta * weight_to_upper_node;
							if (DEBUG_BP) {
								cout << "Small delta: " << small_delta << endl;
								cout << "Weight to upper node " << l << ": " << weight_to_upper_node << endl;
								cout << "Net so far: " << net << endl;
								cout << "Having added " << l << " of " << hidden_node->get_num_output_weights() << endl;
							}
						}
						double my_output = hidden_node->get_output();
						double small_delta = my_output * (1 - my_output) * net;
						if (INCLUDE_SMALL_DELTA_MOMENTUM) {
							small_delta += hidden_node->get_previous_small_delta() * SMALL_DELTA_TERM;
						}
						hidden_node->set_small_delta_weight(small_delta);
						
						if (DEBUG_BP) {
							cout << "Node " << k << "'s output: " << my_output << endl;
							cout << "Its small delta: " << small_delta << endl;
						}
					}
				 }
				 
				 //Set big deltas for each node in input layer (including bias), that each points up to
				 for (int j = 0; j < num_features + 1; j++) {
					 Node* input_node = input_nodes[j];
					 
					 if (DEBUG_BP) cout << "For input node " << j << ":" << endl;
					 
					 for (int l = 0; l < input_node->get_num_output_weights(); l++) {
						double small_delta = hidden_nodes_matrix[0][l]->get_small_delta_weight(); //from lowest layer of hidden nodes, hence the hard-coded 0
						double big_delta = learning_rate * small_delta * input_node->get_output();
						if (INCLUDE_MOMENTUM)
							big_delta += MOMENTUM_TERM * input_node->get_previous_big_delta_weight(l);
						input_node->set_big_delta_weight(l, big_delta);
						if (DEBUG_BP) {
							cout << "Small delta for higher node " << l << ": " << small_delta << endl;
							cout << "Big delta from current input node " << j << " to node " << l << " above: " << big_delta << endl;
						}
					}
			     }
				
				/*************************************
				 * Change weights (stochastic update)
				 *************************************/
				if (INCLUDE_SMALL_DELTA_MOMENTUM) {
					for (int j = 0; j < num_classes; j++)
						output_nodes[j]->remember_previous_small_delta_weight();
				}
				//Change weights, remember delta weights for each input layer node and each node in each hidden layer
				for (int j = 0; j < num_features + 1; j++) { //+ 1 because do it for bias too
					input_nodes[j]->update_weights();
					input_nodes[j]->remember_previous_weights();
					input_nodes[j]->clear_delta_weights();
					input_nodes[j]->remember_previous_small_delta_weight();
				}
				for (int j = 0; j < NUM_HIDDEN_LAYERS; j++) { //+ 1 because do it for bias too
					for (int k = 0; k < num_hidden_nodes + 1; k++) {
						hidden_nodes_matrix[j][k]->update_weights();
						hidden_nodes_matrix[j][k]->remember_previous_weights();
						hidden_nodes_matrix[j][k]->clear_delta_weights();
						hidden_nodes_matrix[j][k]->remember_previous_small_delta_weight();
					}
				}
				//Do another instance
			}
			number_epochs_completed++;
			
			//Done with epoch, so check accuracy
			double training_accuracy = measureAccuracy(features, labels, false, NULL); //This is accuracy on training set (set passed in we're training on)
			double validation_accuracy = measureAccuracy(validation_features, validation_labels, false, NULL); //This is accuracy on validation set
			if (INFO_OUT) cout << number_epochs_completed <<  "," << training_accuracy << "," << validation_accuracy << "," << mean_squared_error << endl;
			
			//If we're doing better, update each weight in each node
			if (validation_accuracy > best_accuracy_so_far) {
				best_accuracy_so_far = validation_accuracy;
				number_without_improvement = 0;
				
				for (int j = 0; j < num_features + 1; j++)
					input_nodes[j]->update_best_weights();
				for (int j = 0; j < NUM_HIDDEN_LAYERS; j++) {
					for (int k = 0; k < num_hidden_nodes + 1; k++) {
						hidden_nodes_matrix[j][k]->update_best_weights();
					}
				}
				
				if (CHANGE_LEARNING_RATE) {
					learning_rate = learning_rate * LR_MULT; //experiment--we like the way we're going, so increas0 learning rate
				}
			}
			else
				number_without_improvement++;
			if (DEBUG_BP) cout << "Number without improvement: " << number_without_improvement << endl;
		}
		while (number_without_improvement < NON_IMPROVEMENT_THRESHOLD);
		
		//Now that we're finally done, make sure every node is using its best set of weights
		for (int j = 0; j < num_features + 1; j++)
			input_nodes[j]->use_best_weights();
		for (int j = 0; j < NUM_HIDDEN_LAYERS; j++) {
			for (int k = 0; k < num_hidden_nodes + 1; k++) {
				hidden_nodes_matrix[j][k]->use_best_weights();
			}
		}
	}

	virtual void predict(const std::vector<double>& features, std::vector<double>& labels)
	{
		//Local variables
		int num_features = features.size() - IGNORE_LAST_ATTRIBUTES;
		int num_hidden_nodes = num_features * NUM_HIDDEN_NODE_MULTIPLIER;
		vector<double> possible_outputs;
		
		//Go up toward input layer
		//Set inputs' outputs
		for (int i = 0; i < num_features; i++) {
			input_nodes[i]->set_output(features[i]);
		}
		input_nodes[num_features]->set_output(1); //Bias node
		 
		//Set ouputs for each node in hidden layer based on nodes below it
		//For each hidden layer
		for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
			
			//For each node in that layer, determine its output (so don't do this for bias nodes on end)
			for (int j = 0; j < num_hidden_nodes; j++) {
				
				double net = 0;
				
				//If we're on the first hidden layer, get outputs and weights from input layer
				if (i == 0) {
					//Get output and weight from each node below that points to this one
					for (int k = 0; k < num_features + 1; k++) {
						net += input_nodes[k]->get_output() * input_nodes[k]->get_weight(j);
					}
				}
				else {
					for (int k = 0; k < num_hidden_nodes + 1; k++) {
						//TODO changed j - 1 to i - 1
						net += hidden_nodes_matrix[i - 1][k]->get_output() * hidden_nodes_matrix[i - 1][k]->get_weight(j);
					}
				}
				
				double output = 1 / (1 + pow(EULER_NUM,-net));
				hidden_nodes_matrix[i][j]->set_output(output);
			}
			hidden_nodes_matrix[i][num_hidden_nodes]->set_output(1); //Bias node
		}
		
		//Set output nodes' outputs
		for (int i = 0; i < num_classes; i++) {
			double net = 0;
			//Get output and weights for each node in top-most layer of hidden nodes layer
			for (int j = 0; j < num_hidden_nodes + 1; j++) {
				net += hidden_nodes_matrix[NUM_HIDDEN_LAYERS - 1][j]->get_output() * hidden_nodes_matrix[NUM_HIDDEN_LAYERS - 1][j]->get_weight(i);
			}
			double output = 1 / (1 + pow(EULER_NUM,-net));
			possible_outputs.push_back(output);
		}
		
		int index_of_largest = 0;
		for (int i = 0; i < num_classes; i++) {
			if (possible_outputs[i] > possible_outputs[index_of_largest])
				index_of_largest = i;
		}
		labels[0] = index_of_largest;
	}
};



#endif // BACKPROPAGATION_H
