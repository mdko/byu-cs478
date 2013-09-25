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
#include <map>
#include <vector>

#define DEBUG 1
#define TEST 0
#define NUM_HIDDEN_LAYERS 1
#define NUM_HIDDEN_NODE_MULTIPLIER 2
#define LEARNING_RATE .1
#define INCLUDE_MOMENTUM 1
#define EULER_NUM 2.71828182845904523536028747135266249775724709369995
#define NON_IMPROVEMENT_THRESHOLD 5

using namespace std;

class BackpropagationLearner : public SupervisedLearner
{
private:
	Rand& m_rand; // pseudo-random number generator (not actually 
				  // used by the baseline learner)
	vector<double> m_labelVec;
	map<int, vector<vector<vector<double> > > > weights_classes_map;
	map<int, vector<vector<vector<double> > > > best_weights_so_far_map;
	vector<vector<double> > output_weight_matrix;
	vector<vector<double> > input_weight_matrix;
	int num_classes;
	double best_acc_so_far;

public:
	BackpropagationLearner(Rand& r)
	: SupervisedLearner(), m_rand(r)
	{
	}

	virtual ~BackpropagationLearner()
	{
	}

	// Train the model to predict the labels
	// features = inputs (first n - # of output columns in matrix)
	// labels = output (last column in matrix)
	virtual void train(Matrix& features, Matrix& labels)
	{
		// Local variable declarations
		int num_features = features.cols();
		int num_instances = features.rows();
		int num_nodes_per_hidden_layer = num_features * NUM_HIDDEN_NODE_MULTIPLIER;
		int num_output_labels = labels.cols(); //how many output (currently not set up to deal with more than one output)			
		num_classes = !labels.valueCount(0) ? 2 : labels.valueCount(0); //number of values associated with label[0] column (we're only expecting one label) TODO check this, 2 or 1
		vector<double> output_layer;
		int number_without_improvement = 0;
		
		// Error handling
		if(features.rows() != labels.rows())
			ThrowError("Expected the features and labels to have the same \
					number of rows");
					
		if (!NUM_HIDDEN_LAYERS)
			ThrowError("Number of hidden layers must be greater than zero.");
		
		if (num_output_labels > 1)
			ThrowError("Unable to handle more than one output");
		
		// Throw away any previous training
		//m_labelVec.clear();
		
		vector<vector<vector<double> > > weights_matrix;
		weights_matrix.resize(NUM_HIDDEN_LAYERS + 1);
		
		//before training, set all weights
		//for (int y = 0; y < num_classes; y++) {
		//TODO should there be the same starting weight for each respective location, regardless of class?
		//if (TEST)
		//	initialize_special_example_weights(weights_matrix);
		//else {
		//for weights between hidden layers (from a hidden layer to a hidden layer)
		//only needed if we have more than 1 hidden layer
			for (int i = 0; i < NUM_HIDDEN_LAYERS - 1/* + 1*/; i++) {
				weights_matrix[i].resize(num_nodes_per_hidden_layer + 1/*+ 1 for bias*/);
				for (int j = 0; j < num_nodes_per_hidden_layer + 1; j++) {
					for (int k = 0; k < num_nodes_per_hidden_layer; k++) { //each node only goes into num_features upper nodes (no nodes go into bias node), BUT there
						double ran_num = ((double)m_rand.next(100) / 100) - .50;	 //there is a place for the bias node to record the weight for each of the upper node IT enters
																					 //why wasn't (double)(m_rand(100) - 50)/100 working properly? -- 
																					 //because m_rand(100) returns an integer, and an integer divided by an integer was zero no matter what
						weights_matrix[i][j].push_back(ran_num); //initalize all weights
					}
				}
			}
		//}
		//for (int y = 0; y < num_classes; y++) {
		weights_classes_map.insert(std::pair<int, vector<vector<vector<double> > > >(0, weights_matrix));
		best_weights_so_far_map.insert(std::pair<int, vector<vector<vector<double> > > >(0, weights_matrix));
		//}
		//}
		
		// to remember the weights going toward each other node from each node in 2nd to top layer
		vector<vector<double> > output_weight_change_matrix;
		vector<vector<double> > output_weight_best_so_far;
		output_weight_matrix.resize(num_nodes_per_hidden_layer + 1);
		output_weight_change_matrix.resize(num_nodes_per_hidden_layer + 1);
		output_weight_best_so_far.resize(num_nodes_per_hidden_layer + 1);
		vector<vector<double> > input_weight_change_matrix;
		vector<vector<double> > input_weight_best_so_far;
		input_weight_change_matrix.resize(num_features + 1);
		input_weight_best_so_far.resize(num_features + 1);
		/*if (TEST) {
			initialize_output_weights(output_weight_matrix);
			for (int w = 0; w < num_features + 1; w++) {
				for (int y = 0; y < num_classes; y++) {
					output_weight_change_matrix[w].push_back(0);
					output_weight_best_so_far[w].push_back(output_weight_matrix[w][y]);
				}
			}
		}
		else {*/
			for (int w = 0; w < num_nodes_per_hidden_layer + 1; w++) {
				for (int y = 0; y < num_classes; y++) { //each node at layer below output will have y number of entries, which are weights pointing up to respective output nodes
					double ran_num = ((double)m_rand.next(100) / 100) - .50;
					output_weight_matrix[w].push_back(ran_num); //TODO change to random weight, like in line 82
					output_weight_change_matrix[w].push_back(0);
					output_weight_best_so_far[w].push_back(ran_num);
				}
			}
		//}
		
		for (int w = 0; w < num_features + 1; w++) {
			for (int y = 0; y < num_nodes_per_hidden_layer; y++) { //each node at layer below output will have y number of entries, which are weights pointing up to respective output nodes
				double ran_num = ((double)m_rand.next(100) / 100) - .50;
				input_weight_matrix[w].push_back(ran_num); //TODO change to random weight, like in line 82
				input_weight_change_matrix[w].push_back(0);
				input_weight_best_so_far[w].push_back(ran_num);
			}
		}
		
		do {
			//shuffle before each epoch
			features.shuffleRows(m_rand, &labels);
			
			for (int x = 0; x < num_instances; x++) {
				//Print weights (for debugging purposes)
				if (DEBUG) cout << "\n**********\nINSTANCE " << x << endl;
				if (DEBUG) {
					cout << "WEIGHTS:" << endl;
					//for (int y = 0; y < num_classes; y++) {
						//cout << "\tCLASS " << y << ":" << endl;
						cout << "\t\tLAYER 0:" << endl;
						cout << "\t\t(NODES labeled from left to right)" << endl;
						for (int j = 0; j < num_features + 1; j++) {
							for (int k = 0; k < num_nodes_per_hidden_layer; k++) {
								cout << "\t\t\tNODE " << j << " to NODE " << k << " in ouput layer above: " << input_weight_matrix[j][k] << endl;
							}
						}
						for (int i = 0; i < NUM_HIDDEN_LAYERS - 1; i++) {
							cout << "\t\tLAYER " << i << ":" << endl;
							cout << "\t\t(NODES labeled from left to right)" << endl;
							for (int j = 0; j < num_nodes_per_hidden_layer + 1; j++) {
								for (int k = 0; k < num_nodes_per_hidden_layer; k++) {
									cout << "\t\t\tNODE " << j << " to NODE " << k << " in layer above: " << (weights_classes_map.at(0))[i][j][k] << endl;
								}
							}
						}
						cout << "\t\tLAYER " << NUM_HIDDEN_LAYERS << ":" << endl;
						cout << "\t\t(NODES labeled from left to right)" << endl;
						for (int j = 0; j < num_features + 1; j++) {
							for (int k = 0; k < num_classes; k++) {
								cout << "\t\t\tNODE " << j << " to NODE " << k << " in ouput layer above: " << output_weight_matrix[j][k] << endl;
							}
						}
					//}
				}
				
				// Initialize everything
				map<int, vector<vector<double> > > nodes_output_classes_map;
				map<int, vector<vector<double> > > small_delta_classes_map;
				map<int, vector<vector<vector<double> > > > weight_change_classes_map;
				vector<double> input_layer;
				
				//for (int y = 0; y < num_classes; y++) {
					
					////////////////////////////////////
					vector<vector<double> > node_output_matrix;
					node_output_matrix.resize(NUM_HIDDEN_LAYERS); //TODO check this--changed b/c
																	  //output_layer vector added
					for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
						//output_matrix[i].resize(num_features + 1);
						for (int j = 0; j < num_nodes_per_hidden_layer + 1; j++) {
							if (j == num_features)
								node_output_matrix[i].push_back(1); //bias weight at end of every level
							else {
								node_output_matrix[i].push_back(0);
							}
						}
					}
					
					for (int i = 0; i < num_features; i++)
						input_layer.push_back(features[x][i]);
					input_layer.push_back(1);
					
					nodes_output_classes_map.insert(std::pair<int, vector<vector<double> > >(0, node_output_matrix));

					////////////////////////////////////
					//output_layer.resize(/*num_classes*/3);
					for (int i = 0; i < num_classes; i++)
						output_layer.push_back(0);

					////////////////////////////////////
					vector<vector<double> > small_delta_matrix;
					small_delta_matrix.resize(NUM_HIDDEN_LAYERS);
					for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
						//small_delta_matrix.resize(num_features + 1);
						for (int j = 0; j < num_nodes_per_hidden_layer; j++) { //TODO review, took out + 1 in conditional 
															   //because nothing goes into bias, no need for small_delta calculation
							small_delta_matrix[i].push_back(0);
						}
					}
					
					small_delta_classes_map.insert(std::pair<int, vector<vector<double> > >(0, small_delta_matrix));
					
					////////////////////////////////////
					vector<vector<vector<double> > > weight_change_matrix;
					weight_change_matrix.resize(NUM_HIDDEN_LAYERS + 1);
					for (int y = 0; y < num_classes; y++) {
			
						for (int i = 0; i < NUM_HIDDEN_LAYERS - 1/* + 1*/; i++) { //our top layer of hidden nodes has its own special matrix
							weight_change_matrix[i].resize(num_nodes_per_hidden_layer + 1/*+ 1 for bias*/);
							for (int j = 0; j < num_nodes_per_hidden_layer + 1; j++) {
								for (int k = 0; k < num_nodes_per_hidden_layer; k++) {
									weight_change_matrix[i][j].push_back(0);
								}
							}
						}
					
						weight_change_classes_map.insert(std::pair<int, vector<vector<vector<double> > > >(0, weight_change_matrix));
					}
				//}

				// GOING UP THE LAYERS, INPUT TO OUTPUT
				// 0. Calculate output of nodes in 1st layer until output layer (included)
				//    0.1 Calculate  nets of nodes in 1st layer of hidden nodes 
				//        (nodes right above input nodes)
				//        ---> net = Sigma(wi*xi)  [so for all nodes coming in to 
				//        							this one]
				//    0.2 Using those nets, calculate output of each node
				//    	  ---> output = 1/(1+e^-net)
				for (int i = 0; i < NUM_HIDDEN_LAYERS + 1 /*+ 1 for output layer*/; i++) {
					if (DEBUG) cout << "GOING UP" << endl;
					//for (int y = 0; y < num_classes; y++) {
						//if (DEBUG) cout << "\tCLASS " << y << endl;
						if (i == NUM_HIDDEN_LAYERS) {
							for (int k = 0; k < num_classes; k++) {
								double net = 0;
								for (int j = 0; j < num_hidden_nodes_per_layer + 1; j++) { //TODO review
									net += (nodes_output_classes_map.at(0))[i][j] /*corresponds to below nodes' ouput*/ * output_weight_matrix[j][k] /*the weight the lower node points to output node with*/;
								}
								if (DEBUG) cout << "\t\tNET of output layer node: " << net << endl;
								double output = 1 / (1 + pow(EULER_NUM,-net));
								output_layer[k] = output;
								if (DEBUG) cout << "\t\tOUTPUT of output layer node: " << output << endl;
							}
						}
						else if (i == 0) {
							for (int k = 0; k < num_hidden_nodes_per_layer; k++) {
								double net = 0;
								for (int l = 0; l < num_features + 1; l++) {
									double output_from_lower_node = input_layer[l];
									double weight_from_lower_node = input_weight_matrix[k][l];
									
									net += (output_from_lower_node * weight_from_lower_node);
								}
								double output = 1 / (1 + pow(EULER_NUM, -net));
								(nodes_output_classes_map.at(0))[i+1][k] = output;
							}
						}
						//hidden layer to hidden layer
						else {
							for (int j = 0; j < num_nodes_per_hidden_layer/*+1 for bias weight*/; j++) {
							
								double net = 0;
								for (int k = 0; k < num_nodes_per_hiddenl_layer + 1; k++) { //calculate sum, from nodes belows
							
									//1st corresponds to inputs of features matrix
									//i.e. o_m[0][0] corresponds to features coming in,
									//and w_m[0][0] corresponds to weights for those features
									//going in next higher layer 
									double output_from_a_lower_node = (nodes_output_classes_map.at(0))[i][k];
									//cout << "*********" << output_from_a_lower_node << endl;
									double weight_for_lower_node_coming_in = (weights_classes_map.at(0))[i][k][j]; //TODO review ijk
									net += (output_from_a_lower_node * weight_for_lower_node_coming_in);
								}
								if (DEBUG) cout << "\t\tNET of node " << j << " of layer " << (i + 1) << ": " << net << endl;
								double output = 1 / (1 + pow(EULER_NUM,-net));
								(nodes_output_classes_map.at(0))[i + 1][j] = output;
								if (DEBUG) cout << "\t\tOUTPUT of node " << j << " of layer " << (i + 1) << ": " << output << endl;
							}
						}
					//}
				}

				// GOING DOWN THE LAYERS, OUTPUT TO INPUT
				vector<double> output_layer_delta_sm_vector;
				for (int i = NUM_HIDDEN_LAYERS; i >= 0; i--) {
					if (DEBUG) cout << "GOING DOWN" << endl;
					//for (int y = 0; y < num_classes; y++) {
						//if (DEBUG) cout << "\tCLASS " << y << endl;
						// 1. Calculate big Delta each output node
						//    1.1 Calculate little delta for each output node
						//    --> dj = (tj - outputj)f'(net)
						//           = (tj - outputj)(outputj(1 - outputj))
						//    1.2 Using those little deltas, calculate big Delta for nodes below coming in to it
						//    --> Dij = C * outputi * dj
						if (i == NUM_HIDDEN_LAYERS) { //we're at the output layer, find error(s) so
													  //we can start propagating back
							for (int k = 0; k < num_classes; k++) {
								double output = output_layer[k];
								double target_output = (labels[x][0] == k) ? 1 : 0;
								double output_layer_delta_sm = (target_output - output)  //'x' is instance #, TODO fix so we go across each output node
																* output
																* (1 - output);
								output_layer_delta_sm_vector.push_back(output_layer_delta_sm);
								if (DEBUG) cout << "\t\tOUTPUT of output layer node: " << output << endl;
								if (DEBUG) cout << "\t\tTARGET of output layer node: " << target_output << endl;
								if (DEBUG) cout << "\t\tSMALL DELTA of output layer node: " << output_layer_delta_sm << endl;
								for (int j = 0; j < num_features + 1; j++) {

									double delta_bg = LEARNING_RATE 
													  * output_layer_delta_sm 
													  * (nodes_output_classes_map.at(0))[i][j];
									if (DEBUG) cout << "\t\tBIG DELTA for weight between output layer and node " << j << " in layer " << i << ": " << delta_bg << endl;
									//(weight_change_classes_map.at(0))[i][j][k] = delta_bg; //TODO fix hc
									output_weight_change_matrix[j][k] = delta_bg; //TODO review
								}
							}
						}
						
						// 2. Calculate big Delta for each hidden node in each layer of hidden
						//    nodes until you reach input layer (don't do input layer)
						//    2.1 Calculate little delta for each hidden node [looking forward]
						//    --> dj = Sigmak(dk [node it goes into in higher level] * 
						//    				  wjk [weight b/w those two] * outputj(1 - outputj)
						//    2.2 Calculate big Delta for each hidden node
						//    -->Dij= C * outputi * dj
						else {
							
							for (int j = 0; j < num_features; j++) { //TODO review, took out + 1 in middle of conditional
								double output = (nodes_output_classes_map.at(0))[i + 1][j];
								double summation = 0;

								//START trickyish part
								if (i == (NUM_HIDDEN_LAYERS - 1)) {//we're on the layer just below output layer, so no summation (just one small delta from output)
									for (int k = 0; k < num_classes; k++) {
										summation = output_layer_delta_sm_vector[k]
											* output_weight_matrix[j][k];		
									}
								}
								else { //we're in a hidden layer whose layer above is also a hidden layer
									for (int z = 0; z < num_features; z++) //TODO review removed + 1
										summation += (small_delta_classes_map.at(0))[i + 1][j] //TODO review
													* (weights_classes_map.at(0))[i + 1][j][z]; //all the arrows out of this node to upper nodes
								}
								
								double delta_sm = output * (1 - output) * summation;
								if (DEBUG) cout << "\t\tSMALL DELTA for node " << j << " of layer " << (i + 1) << ": " << delta_sm << endl;
								(small_delta_classes_map.at(0))[i][j] = delta_sm; //TODO review, changed i-1 to i, for use in layer below
								
								for (int k = 0; k < num_features + 1; k++) { //get big deltas for nodes below that go into this current node
				
									double delta_bg = 
											LEARNING_RATE
											* delta_sm
											* (nodes_output_classes_map.at(0))[i][k];
									if (DEBUG) cout << "\t\tBIG DELTA for weight between node " << j << " of layer " << (i + 1) << "and node " << k << " in layer " << i << ": " << delta_bg << endl; //TODO check
									(weight_change_classes_map.at(0))[i][k][j] = delta_bg; //TODO NEED TO CHANGE AFTER END OF EPOCH, NOT IN MIDDLE OF EPOCHS
								}
							}
						}
					//}
				}
				//end of an instance, update weights (on-line/stochastic)
				//for (int y = 0; y < num_classes; y++) {
					for (int i = 0; i < NUM_HIDDEN_LAYERS/* + 1*/; i++) {
						for (int j = 0; j < num_features + 1; j++) {
							for (int k = 0; k < num_features; k++) { 
								(weights_classes_map.at(0))[i][j][k] += (weight_change_classes_map.at(0))[i][j][k];
							}
						}
					}
					
					//TODO MAKE sure this is working properly, fix outputting too
					for (int w = 0; w < num_features + 1; w++) {
						for (int y = 0; y < num_classes; y++) {
							output_weight_matrix[w][y] += output_weight_change_matrix[w][y];
						}
					}
				//}
				
				//test on validation set? save best weights so far
				double mseTraining = measureAccuracy(features, labels, NULL);
				cout << "ACCURACY: " << mseTraining << endl;
				if (mseTraining > best_acc_so_far) { //our weights are better than the best so far
					best_acc_so_far = mseTraining;
					number_without_improvement = 0;
					//for (int y = 0; y < num_classes; y++) {
						for (int i = 0; i < NUM_HIDDEN_LAYERS/* + 1*/; i++) {
							for (int j = 0; j < num_features + 1; j++) {
								for (int k = 0; k < num_features; k++) { 
									(best_weights_so_far_map.at(0))[i][j][k] = (weights_classes_map.at(0))[i][j][k];
								}
							}
						}
						
						for (int w = 0; w < num_features + 1; w++) {
							for (int y = 0; y < num_classes; y++) {
								output_weight_best_so_far[w][y] += output_weight_matrix[w][y];
							}
						}
						
					//}
				}
				else
					number_without_improvement++;
			}
		}
		while (number_without_improvement < NON_IMPROVEMENT_THRESHOLD);
		
		//We've stopped, so make sure we're now using the best weights found during the entire process
		//for (int y = 0; y < num_classes; y++) {
			for (int i = 0; i < NUM_HIDDEN_LAYERS/* + 1*/; i++) {
				for (int j = 0; j < num_features + 1; j++) {
					for (int k = 0; k < num_features; k++) { 
						(weights_classes_map.at(0))[i][j][k] = (best_weights_so_far_map.at(0))[i][j][k];
					}
				}
			}
			
			for (int w = 0; w < num_features + 1; w++) {
				for (int y = 0; y < num_classes; y++) { //each node at layer below output will have y number of entries, which are weights pointing up to respective output nodes
					output_weight_matrix[w][y] += output_weight_best_so_far[w][y];
				}
			}
		//}
	}

	// Evaluate the features and predict the labels
	// 
	// features are the attributes (cols) of the one passed-in instance we want to produce output for
	// 
	// labels is what where we report our answer(s)/prediction(s) of those features
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels)
	{
		if (DEBUG) {
			cout << "\nWEIGHTS BEFORE VALIDATION:" << endl;
			int num_features = features.size();
			for (int i = 0; i < NUM_HIDDEN_LAYERS; i++) {
				cout << "\t\tLAYER " << i << ":" << endl;
				cout << "\t\t(NODES labeled from left to right)" << endl;
				for (int j = 0; j < num_features + 1; j++) {
					for (int k = 0; k < num_features; k++) {
						cout << "\t\t\tNODE " << j << " to NODE " << k << " in layer above: " << (weights_classes_map.at(0))[i][j][k] << endl;
					}
				}
			}
			cout << "\t\tLAYER " << NUM_HIDDEN_LAYERS << ":" << endl;
			cout << "\t\t(NODES labeled from left to right)" << endl;
			for (int j = 0; j < num_features + 1; j++) {
				for (int k = 0; k < num_classes; k++) {
					cout << "\t\t\tNODE " << j << " to NODE " << k << " in ouput layer above: " << output_weight_matrix[j][k] << endl;
				}
			}
		}
		
		vector<double> possible_outputs;

		//for (int i = 0; i < num_classes; i++) {
			vector<vector<double> > intermediate_outputs;
			intermediate_outputs.resize(NUM_HIDDEN_LAYERS + 1);
			if (DEBUG) cout << "FEATURES OF INSTANCE: " << endl;
			for (int x = 0; x < features.size(); x++) {
				intermediate_outputs[0].push_back(features[x]);
				cout << features[x] << endl;
			}
			intermediate_outputs[0].push_back(1); //for bias
			
			for (int j = 0; j < NUM_HIDDEN_LAYERS; j++) {
				for (int k = 0; k < features.size(); k++) {
					double net = 0;
					for (int l = 0; l < features.size() + 1; l++) {
						net += intermediate_outputs[j][l] * (weights_classes_map.at(0))[j][l][k]; //TODO review
					}
					double output = 1 / (1 + pow(EULER_NUM,-net));
					intermediate_outputs[j+1].push_back(output);
					cout << "INTERM. OUTPUT for layer "  << j + 1 << " node " << k << ": " << output << endl;
				}
				intermediate_outputs[j+1].push_back(1); //for bias
			}
			//calculate final output
			for (int i = 0; i < num_classes; i++) {
				double net = 0;
				cout << "x" << endl;
				for (int j = 0; j < features.size() + 1; j++) {
					cout << "interout: " << intermediate_outputs[NUM_HIDDEN_LAYERS][j] << " outweight: " << output_weight_matrix[j][i] << endl;
					net += intermediate_outputs[NUM_HIDDEN_LAYERS][j] * output_weight_matrix[j][i];
				}
				double final_output = 1 / (1 + pow(EULER_NUM,-net));
				possible_outputs.push_back(final_output);
			}
			//if (DEBUG) cout << "CLASS " << i << " output: " << final_output << endl;
		//}
		
		int index_of_largest = 0;
		for (int i = 0; i < num_classes; i++) {
			if (DEBUG) cout << "POSS OUT: " << possible_outputs[i] << endl;
			if (possible_outputs[i] > possible_outputs[index_of_largest])
				index_of_largest = i;
		}
		labels[0] = index_of_largest;
		if (DEBUG) cout << "INDEX of LARGEST: " << index_of_largest << endl;
	}
	
	virtual void initialize_special_example_weights(vector<vector<vector<double> > > &m) {
		for (int i = 0; i < NUM_HIDDEN_LAYERS/* + 1*/; i++) {
			m[i].resize(2 + 1/*+ 1 for bias*/);
		}
		m[0][0].push_back(0.2);
		m[0][0].push_back(0.3);
		m[0][1].push_back(-0.1);
		m[0][1].push_back(-0.3);
		m[0][2].push_back(0.1);
		m[0][2].push_back(-0.2);
		
		m[1][0].push_back(-0.2);
		m[1][0].push_back(-0.1);
		m[1][1].push_back(-0.3);
		m[1][1].push_back(0.3);
		m[1][2].push_back(0.1);
		m[1][2].push_back(0.2);
	}
	
	virtual void initialize_output_weights(vector<vector<double> > &m) {
		m[0].push_back(-0.1);
		m[0].push_back(-0.2);
		m[1].push_back(0.3);
		m[1].push_back(-0.3);
		m[2].push_back(0.2);
		m[2].push_back(0.1);
	}
};


#endif // BACKPROPAGATION_H
