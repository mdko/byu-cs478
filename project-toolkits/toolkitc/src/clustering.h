/**
 * Author: Michael Christensen
 * Date: April 1, 2013
 *
 * For BYU CS 478 Clustering Learning Project
 */

#ifndef CLUSTERING_H
#define CLUSTERING_H

#include "learner.h"
#include "rand.h"
#include "error.h"
#include <cmath>
#include <stdio.h>
#include <iostream>
#include <map>
#include <vector>
#include <queue>
#include <assert.h>

#define DEBUG_CL 0
#define DEBUG_CL_E 0
#define INFO_OUT_CL 1
// HAC
#define HAC 0			// 1 = HAC, 0 = k-means
#define SINGLE_LINK 0	// 1 = Single link, 0 = Complete link
#define HAC_LO_K 2
#define HAC_HI_K 7
// K-means
#define K_MEAN_K 2
#define USE_1ST_N 0		// > 0 = use 1st n instances as initial clusters for debugging, 0 = random instances
#define NUM_WO_IMPROV 3

#define INC_LABL 1

#define EXPERIMENT 1

using namespace std;

class ClusteringLearner : public SupervisedLearner
{
	class Node
	{

	private:
		int instance_number;
		int number_instances;
		vector<double> features_values;
		//const vector<double>& features_values;
		//const vector<double>& labels_values;
		int number_features;
		vector<double> distance_to_other_instances;

	public:
		Node(int inst_num, int num_instances, const vector<double>& features, const vector<double>& labels, int num_features) :
			instance_number(inst_num),
			number_instances(num_instances),
			//features_values(features),
			//labels_values(labels),
			number_features(num_features)
		{
			for (int instance_n = 0; instance_n < num_instances; instance_n++)
				distance_to_other_instances.push_back(0.0);

			for (int feat_n = 0; feat_n < num_features; feat_n++) {
				if (INC_LABL && feat_n == (num_features - 1))
					features_values.push_back(labels[0]);
				else
					features_values.push_back(features[feat_n]);
			}
		}

		~Node()
		{
		}

		void printNodeFeatures() {
			cout << "Instance number " << instance_number << endl;
			cout << "\tFeatures: ";
			for (int fea_n = 0; fea_n < number_features; fea_n++)
				cout << features_values[fea_n] << " ";
			cout << endl;
		}

		void printNodeDistances() {
			cout << "Distance from instance " << instance_number << " to " << endl;
			for (int inst_n = 0; inst_n < number_instances; inst_n++)
				cout << inst_n << ": " << distance_to_other_instances[inst_n] << endl;
		}

		void setDistance(int other_instance, double value) {
			distance_to_other_instances[other_instance] = value;
		}

		double getDistance(int other_instance) {
			return distance_to_other_instances[other_instance];
		}

		int getInstanceNumber() {
			return instance_number;
		}

		const vector<double>& getFeatureValues() {
			return features_values;
		}
	};

	class Cluster
	{
	private:
		int num_members;
		vector<Node*> members;
		Matrix& features_matrix;
		Matrix& labels_matrix;
		// Would be used if optimizing, that is if we only computed distance between
		// to clusters if the contents of one or both of them had changed since the
		// last computation (would be determined via the "changed" variable)
		vector<double> distance_to_other_clusters;
		int total_num_instances;
		bool changed;
		// For k-means
		vector<double> centroid_features;
		double SSE;

	public:
		Cluster(Node* initial_member, Matrix& features, Matrix& labels, int num_instances) :
				features_matrix(features),
				labels_matrix(labels),
				total_num_instances(num_instances) {
			members.push_back(initial_member);
			num_members = 1;
			for (int inst_n = 0; inst_n < num_instances; inst_n++)
				distance_to_other_clusters.push_back(0.0); // 0 signifies cluster exists, -1 it doesn't, > 0 actual distance
			changed = 1;

			const vector<double>& initial_node_features = initial_member->getFeatureValues();
			for (int feature_n = 0; feature_n < (int)(initial_node_features.size()); feature_n++)
				centroid_features.push_back(initial_node_features[feature_n]);

			SSE = 0.0;
		}

		~Cluster() {
		}

		Node* getInitialMemberNode() {
			return members[0];
		}

		const vector<double>& getCentroidFeatures() {
			return centroid_features;
		}

		void setDistance(int other_cluster, double value) {
			distance_to_other_clusters[other_cluster] = value;
			changed = true;
		}

		// The other cluster doesn't exist any more
		void nullDistance(int other_cluster) {
			distance_to_other_clusters[other_cluster] = -1;
		}

		double getDistanceToOtherCluster(int other_cluster) {
			return distance_to_other_clusters[other_cluster];
		}

		void setUnchanged() {
			changed = false;
		}

		bool isChanged() {
			return changed;
		}

		const vector<Node*>& getNodeMembers() {
			return members;
		}

		void addNode(Node* node) {
			members.push_back(node);
			num_members++;
		}

		void addNodes(const vector<Node*>& other_nodes) {
			for (int i = 0; i < (int)other_nodes.size(); i++) {
				members.push_back(other_nodes[i]);
				num_members++;
			}
		}

		void calculateCentroid() {
			// For each feature/attribute
			for (int feature_n = 0; feature_n < (int)centroid_features.size(); feature_n++) {
				int num_dont_knows = 0;
				double summation = 0.0; // For determining mean discrete class
				int num_features_in_denom = 0;
				
				int size_nom_sum;
				if (INC_LABL && feature_n == ((int)centroid_features.size() - 1))
					size_nom_sum = labels_matrix.valueCount(0);
				else
					size_nom_sum = features_matrix.valueCount(feature_n);
				int nominal_summation[size_nom_sum]; // For determining majority nominal class
				fill_n(nominal_summation, size_nom_sum, 0);
				
				bool nominal;
				if (INC_LABL && feature_n == ((int)centroid_features.size() - 1))
					nominal = labels_matrix.valueCount(0);
				else
					nominal = features_matrix.valueCount(feature_n);
				
				if (DEBUG_CL_E) cout << "Feature " << feature_n << endl;
				// Find average for each feature by iterating over feature in each node
				for (int node_n = 0; node_n < num_members; node_n++) {
					double feature = members[node_n]->getFeatureValues()[feature_n];
					if (feature == UNKNOWN_VALUE)
						num_dont_knows++;
					else {
						if (nominal){// If nominal
							nominal_summation[(int)feature]++;
							if (DEBUG_CL_E) {
								cout << "Feature class (" << (int)feature << ") : "
									<< ((INC_LABL && feature_n == ((int)centroid_features.size() - 1))
										? labels_matrix.attrValue(0, feature)
										: features_matrix.attrValue(feature_n, feature)) << endl;
							}
						}
						else {		// If discrete
							summation += feature;
							num_features_in_denom++;
						}
					}
				}

				// Change value of feature in centroid features array
				if (num_dont_knows == num_members)
					centroid_features[feature_n] = UNKNOWN_VALUE;
				else if (nominal) {
					int mode_class = 0;
					int num_classes = (INC_LABL && feature_n == ((int)centroid_features.size() - 1))
										? (int)labels_matrix.valueCount(0)
										: (int)features_matrix.valueCount(feature_n);
					for (int class_n = 0; class_n < num_classes; class_n++) {
						if (DEBUG_CL_E) cout << "Summation of nominal feature " << class_n << ": " << nominal_summation[class_n] << endl;
						if (nominal_summation[class_n] > nominal_summation[mode_class])
							mode_class = class_n;
					}
					if (DEBUG_CL_E) cout << "Mode class: " << mode_class << endl;
					centroid_features[feature_n] = mode_class;
				}
				else {
					double mean = summation / num_features_in_denom;
					centroid_features[feature_n] = mean;
				}
			}
		}

		double calculateSSE() {
			double total_error = 0.0;
			
			for (int feature_n = 0; feature_n < (int)centroid_features.size(); feature_n++) {
				double centroid_feature = centroid_features[feature_n];
				bool nominal = (INC_LABL && feature_n == ((int)centroid_features.size() - 1))
								? labels_matrix.valueCount(0)
								: features_matrix.valueCount(feature_n);
				
				for (int member_n = 0; member_n < num_members; member_n++) {
					double inter_error = 0.0;
					double member_feature = members[member_n]->getFeatureValues()[feature_n];

					if (centroid_feature == UNKNOWN_VALUE || member_feature == UNKNOWN_VALUE)
						inter_error = 1.0;
					else if (nominal)
						inter_error = ((int)centroid_feature == (int)member_feature) ? 0 : 1;
					else
						inter_error = centroid_feature - member_feature;
					total_error += pow(inter_error, 2);
				}
			}
			SSE = total_error;
			return total_error; 
		}

		void clearMembers() {
			members.clear();
			num_members = 0;
		}

		int getNumMembers() {
			return num_members;
		}

		bool isCentroidFeatureNominal(int feat_n) {
			if (INC_LABL && (feat_n == ((int)centroid_features.size() - 1)))
				return labels_matrix.valueCount(feat_n);
			return features_matrix.valueCount(feat_n);

		}

		double getScatter() {
			return calculateSSE() / members.size();
		}

		void printNodesDetailed() {
			for (int member_n = 0; member_n < num_members; member_n++) {
				members[member_n]->printNodeFeatures();
				members[member_n]->printNodeDistances();
			}
		}

		void printNodes() {
			for (int member_n = 0; member_n < num_members; member_n++)
				cout << members[member_n]->getInstanceNumber() << " ";
		}

		void printCentroidFeatures() {
			for (int feature_n  = 0; feature_n < (int)centroid_features.size(); feature_n++) {
				double feature = centroid_features[feature_n];
				if (feature == UNKNOWN_VALUE) // Unknown
					cout << "?";
				else if (INC_LABL && (feature_n == (int)centroid_features.size() - 1)) {
					if (labels_matrix.valueCount(0)) 
						cout << labels_matrix.attrValue(0, feature);
					else
						cout << feature;
				}
				// We're either not including the label, or we are but we're still in the bounds of the input features vector
				else if (features_matrix.valueCount(feature_n))
					cout << features_matrix.attrValue(feature_n, feature);
				else
					cout << feature;
					
				if (feature_n < (int)centroid_features.size() - 1)
					cout << ", ";
			}
		}
	};

	
	class DistanceMatrix
	{

	private:
		int num_features;
		vector<vector<double> > matrix;

	public:
		DistanceMatrix(int _num_features) : num_features(_num_features) {
			matrix.resize(_num_features);
			for (int i = 0; i < _num_features; i++) {
				for (int j = 0; j < _num_features; j++) {
					matrix[i].push_back(0.0);
				}
			}
		}

		~DistanceMatrix() {
		}

		void setDistance(int row, int col, double value) {
			matrix[row][col] = value;
			if (DEBUG_CL) cout << "Setting " << row << ", "
							   << col << ": to " << value << endl;
			//printMatrix();
		}

		void printMatrix() {
			for (int row = 0; row < num_features; row++) {
				for (int col = 0; col < num_features; col++) {
					cout << matrix[row][col] << " ";
				}
				cout << endl;
			}
		}
	};
	
private:
	Rand& m_rand;	

public:
	ClusteringLearner(Rand& r)
	: SupervisedLearner(), m_rand(r)
	{
	}

	virtual ~ClusteringLearner()
	{
	}

	virtual void train(Matrix& features, Matrix& labels)
	{
		// -Be able to compute with and without the label as an input feature
		// -For continuous attributes, use Euclidean distance
		// -For nominal/unknown attributes, use (0,1): matching attributes have
		//   distance of 0, else 1
		// -When calculating centroids, ignore missing values
		// -When determining total sum squared error, assign distance of 1
		// -In case of tie in number of each nominal valuef for an attribute,
		//   choose nominal value which appears first in the meta data list
		// -In case of ties between node/cluster distance to another (very rare)
		//   go with earlier cluster in list
		// -If all attributes in cluster have "don't know" for one attr, use dk for that attr.

		int num_instances = (int)features.rows();
		int num_features = (int)features.cols();
		if (INC_LABL) num_features++;

		// HAC (single link, complete link; pass in a range of k to output)
		if (HAC) {
			if (INFO_OUT_CL) cout << "Running HAC..." << endl;
			// 1. Fill nxn adjacency matrix to give dist between each pair of instances
			DistanceMatrix* dm = new DistanceMatrix(num_instances);

			// 2. Initialize each instance to be its own cluster (initialize all
			// here so we can do symmetric assignment of distances for two nodes
			// at the same time)
			vector<Cluster*> clusters;
			for (int instance_n = 0; instance_n < num_instances; instance_n++)
				clusters.push_back(new Cluster(
					new Node(instance_n, num_instances, features[instance_n], labels[instance_n], num_features), features, labels, num_instances));

			// For each instance in the training set
			for (int instance_n = 0; instance_n < num_instances; instance_n++) {
				const vector<double>& current_instance = features[instance_n];

				// Find its distance from every other instance in training set
				for (int other_instance_n = 0; other_instance_n < num_instances; other_instance_n++) {
					double distance_summation = 0.0;
					double distance = 0.0;
					
					// Since symmetrical, no need to calculate distances below diagonal,
					// since it's already been copied over from corresponding entry from upper part of matrix
					// or 0's along the diagonal, which are set in constructor
					if (instance_n >= other_instance_n) {
						if (DEBUG_CL) cout << instance_n << " is greater than or equal to " << other_instance_n << endl;
						continue;
					}
					else {
						const vector<double>& other_instance = features[other_instance_n];

						// Find distance from current_instance to other_instance using
						// square of summation of distance between each feature
						for (int feature_n = 0; feature_n < num_features; feature_n++) {								
							double inter_distance = 0.0;
							double current_feature;
							double other_feature;

							if (INC_LABL && feature_n == (num_features - 1)) {
								current_feature = labels[instance_n][0];
								other_feature = labels[other_instance_n][0];
							}
							else {
								current_feature = current_instance[feature_n];
								other_feature = other_instance[feature_n];
							}
							
							if (DEBUG_CL) cout << "Feature " << feature_n << " is ";
							if (current_feature == UNKNOWN_VALUE || other_feature == UNKNOWN_VALUE) {
								inter_distance = 1.0;
								if (DEBUG_CL) cout << "unknown ?" << endl;
							}
							else {
								bool nominal_feature;
								if (INC_LABL && feature_n == (num_features - 1))
									nominal_feature = (labels.valueCount(feature_n) > 0);
								else
									nominal_feature = (features.valueCount(feature_n) > 0);

								if (nominal_feature) {
									inter_distance = ((int)current_feature == (int)other_feature)
													? 0.0 : 1.0;
									if (DEBUG_CL) {
										cout << "nominal " <<
											((INC_LABL && feature_n == (num_features - 1))
												? labels.attrValue(0, other_feature)
												: features.attrValue(feature_n, other_feature)) << endl;

									}
								}
								else {
									inter_distance = current_feature - other_feature;
									if (DEBUG_CL) cout << "discrete " << other_feature << endl;
								}
							}
							if (DEBUG_CL) cout << "inter-dist " << inter_distance << endl;
							distance_summation += pow(inter_distance, 2);
						}
						distance = sqrt(distance_summation);
						clusters[instance_n]->getInitialMemberNode()->setDistance(other_instance_n, distance);
						clusters[other_instance_n]->getInitialMemberNode()->setDistance(instance_n, distance); // Since symmetric
						dm->setDistance(instance_n, other_instance_n, distance);
						dm->setDistance(other_instance_n, instance_n, distance);
					}
				}
			}
			
			if (DEBUG_CL) {
				dm->printMatrix();
				for (int instance_n = 0; instance_n < num_instances; instance_n++)
					clusters[instance_n]->printNodesDetailed();
			}

			// 3. Repeat:
			//		Merge the two "closest" remaining clusters into one cluster
			//	  Until there is just one cluster containing all instances
			//  Where "closest" is defined as:
			//		Single link -- smallest distance between any 2 points in A and B
			//		Complete link -- largest distance between any 2 points in A and B
			int iteration_number = 0;
			while (clusters.size() > HAC_LO_K) {
				// One closest merge per iteration, recompute distances between clusters
				// each time one is removed
				if (INFO_OUT_CL && clusters.size() <= HAC_HI_K)
					cout << "--------------\nIteration " << iteration_number << "\n--------------" << endl;
				iteration_number++;

				if (EXPERIMENT) {
					// Given n pairs of clusters that are closest, random choose one pair to combine, not just
					// the closest like normal
				}
				
				double shortest_distance = -1.0;
				int cluster_0_to_combine = 0;
				int cluster_1_to_combine = 1;
				for (int clus_n = 0; clus_n < (int)clusters.size(); clus_n++) {
					for (int clus_m = 0; clus_m < (int)clusters.size(); clus_m++) {
						if (clus_n >= clus_m)
							continue; //Diagonal, and distances symmetric

						double clus_distance = computeXLinkDistance(clusters[clus_n], clusters[clus_m]);

						if (shortest_distance == -1.0 || clus_distance < shortest_distance) {
							cluster_0_to_combine = clus_n;
							cluster_1_to_combine = clus_m;
							shortest_distance = clus_distance;
						}
					}
				}
				Cluster* cl_0 = clusters[cluster_0_to_combine];
				Cluster* cl_1 = clusters[cluster_1_to_combine];
				combineClusters(cl_0, cl_1);
				if (INFO_OUT_CL && clusters.size() <= HAC_HI_K) {
					cout << "Merging clusters " << cluster_0_to_combine << " and " << cluster_1_to_combine;
					cout << "\tDistance: " << shortest_distance << endl;
				}
				
				// Remove one of the clusters that was just put with another one
				removeCluster(clusters, cl_1);

				if (INFO_OUT_CL && clusters.size() <= HAC_HI_K) {
					cout << "There are " << (int)clusters.size() << " clusters" << endl;
					for (int clus_n = 0; clus_n < (int)clusters.size(); clus_n++) {
						cout << "Cluster " << clus_n << ": ";
						clusters[clus_n]->printNodes();
						cout << endl;
					}
				}

				//	Calculate centroids (just for outputting information purposes)
				if (INFO_OUT_CL && clusters.size() <= HAC_HI_K) cout << "Recomputing the centroids for each cluster..." << endl;
				double total_SSE = 0.0;
				for (int cluster_n = 0; cluster_n < (int)clusters.size(); cluster_n++) {
					clusters[cluster_n]->calculateCentroid();
					double cluster_SSE = clusters[cluster_n]->calculateSSE();
					total_SSE += cluster_SSE;
					if (INFO_OUT_CL && clusters.size() <= HAC_HI_K) {
						cout << "Centroid " << cluster_n << " = ";
						clusters[cluster_n]->printCentroidFeatures();
						cout << endl;
						cout << "\tMembers of Cluster (" << clusters[cluster_n]->getNumMembers() << "): ";
						clusters[cluster_n]->printNodes();
						cout << endl;
						cout << "\tCluster " << cluster_n << " SSE = " << cluster_SSE << endl;
					}
				}
				if (INFO_OUT_CL && clusters.size() <= HAC_HI_K) {
					cout << "Total SSE (sum squared-distance of each row with its centroid) = " << total_SSE << endl;
					cout << "Davies-Bouldin Index: " << computeDaviesBouldinIndex(clusters) << endl;
				}
			}
		}
		// k-means (choose k beforehand)
		else {
			cout << "Running k-means..." << endl;
			
			vector<Cluster*> clusters;

			// 1. Choose first n instances from the set of features
			if (USE_1ST_N) {
				for (int instance_n = 0; instance_n < USE_1ST_N; instance_n++) {
					clusters.push_back(new Cluster
							(new Node(instance_n, num_instances, features[instance_n], labels[instance_n], num_features),
										features, labels, num_instances));
				}
			}
			// 1. Randomly choose k instances from the data set to be the initial k centroids
			//     (the features in the first node added make up the initial features in the centroid vector)
			else {
				if (EXPERIMENT) {
					// Add k random instances, but make sure the number of output classes represented is evenly distributed
					// (ie if there are 4 possible outputs, there must be at least 4 initial clusters (even if k given is less than
					// that), and each centroid's feature representing output value must be different). Think of an injective function,
					// each output class is covered. If k = 5 and there are 3 classes, clusters 0 and 3 will be class 0, clusters 1 and 4
					// will be class 1, and cluster 5 will be class 2
					int classes_num = labels.valueCount(0);
					bool instances_chosen[num_instances];
					fill_n(instances_chosen, num_instances, 0);
					int initial_clusters_n = 0;
					
					while (initial_clusters_n < K_MEAN_K) {
						
						bool class_chosen[classes_num];
						fill_n(class_chosen, classes_num, 0);
						for (int class_n = 0; class_n < classes_num && initial_clusters_n < K_MEAN_K; class_n++) {
							int ran_instance_n = 0;
							int class_label = 0;
							while (1) {
								ran_instance_n = (int)m_rand.next(num_instances);
								class_label = labels[ran_instance_n][0];
								if (instances_chosen[ran_instance_n])
									continue;
								if (class_label == class_n)
									break;
							}

							instances_chosen[ran_instance_n] = 1;
							class_chosen[class_n] = 1;
							clusters.push_back(new Cluster
									(new Node(ran_instance_n, num_instances, features[ran_instance_n], labels[ran_instance_n], num_features),
													features, labels, num_instances));
							initial_clusters_n++;
						}
					}
				}
				else {
					bool instances_chosen[num_instances];
					fill_n(instances_chosen, num_instances, 0); // 0 = not chosen yet, 1 = chosen
					int initial_clusters_n = 0;
					while (initial_clusters_n <  K_MEAN_K) {
						int ran_instance_n = (int)m_rand.next(num_instances);
						if (!instances_chosen[ran_instance_n]) {
							clusters.push_back(new Cluster
								(new Node(ran_instance_n, num_instances, features[ran_instance_n], labels[ran_instance_n], num_features),
												features, labels, num_instances));
							instances_chosen[ran_instance_n] = 1;
							initial_clusters_n++;
						}
					}
				}
			}

			if (INFO_OUT_CL) {
				for (int cluster_n = 0; cluster_n < (int)clusters.size(); cluster_n++) {
					cout << "Centroid " << cluster_n << " = ";
					clusters[cluster_n]->printCentroidFeatures();
					cout << endl;
				}
			}

			// 2. Repeat:
			//	   Group each instance with its closest centroid
			//	 	Until no/negligible changes occur
			int num_iterations_wo_improv = 0;
			double lastSSE = -1.0;
			while (num_iterations_wo_improv < NUM_WO_IMPROV) {
				
				if (INFO_OUT_CL) {
					cout << "------------------------------------------------------------" << endl;
					cout << "Assigning each row to the cluster of the nearest centroid..." << endl;
					cout << "------------------------------------------------------------" << endl;
				}

				// Clear out the nodes in each cluster (but not the centroid value),
				// so we can re-add the pertinent ones
				for (int clus_n = 0; clus_n < (int)clusters.size(); clus_n++) {
					clusters[clus_n]->clearMembers();
				}
				
				// For each instance
				for (int instance_n = 0; instance_n < num_instances; instance_n++) {
					vector<double>& current_instance = features[instance_n];
					vector<double>& current_instance_label = labels[instance_n]; // if INC_LABL

					// Find distance from current instance to each cluster, keeping track of which one is closest
					double shortest_distance = -1.0;
					int closest_cluster_num = 0;
					for (int cluster_n = 0; cluster_n < (int)clusters.size(); cluster_n++) {

						const vector<double>& cluster_centroid = clusters[cluster_n]->getCentroidFeatures();
						double distance_summation = 0.0;
						// Find distance from current_instance to centroid using
						// square of summation of distance between each feature in current
						// instance and centroid's features
						for (int feature_n = 0; feature_n < num_features; feature_n++) {								
							double inter_distance = 0.0;
							double current_feature;
							double cluster_feature;

							if (INC_LABL && feature_n == (num_features - 1)) {
								current_feature = current_instance_label[0];
								cluster_feature = cluster_centroid[feature_n];
							}
							else {
								current_feature = current_instance[feature_n];
								cluster_feature = cluster_centroid[feature_n];
							}
							
							if (DEBUG_CL) cout << "Feature " << feature_n << " is ";
							if (current_feature == UNKNOWN_VALUE || cluster_feature == UNKNOWN_VALUE) {
								inter_distance = 1.0;
								if (DEBUG_CL) cout << "unknown ?" << endl;
							}
							else {
								bool nominal_feature;
								if (INC_LABL && feature_n == (num_features - 1))
									nominal_feature = (labels.valueCount(feature_n) > 0);
								else
									nominal_feature = (features.valueCount(feature_n) > 0);

								if (nominal_feature) {
									inter_distance = ((int)current_feature == (int)cluster_feature)
													? 0.0 : 1.0;
									if (DEBUG_CL) {
										cout << "nominal " <<
											((INC_LABL && feature_n == (num_features - 1))
												? labels.attrValue(0, cluster_feature)
												: features.attrValue(feature_n, cluster_feature)) << endl;
									}
								}
								else {
									inter_distance = current_feature - cluster_feature;
									if (DEBUG_CL) cout << "discrete " << cluster_feature << endl;
								}
							}
							if (DEBUG_CL) cout << "inter-dist " << inter_distance << endl;
							distance_summation += pow(inter_distance, 2);
						}
						double distance = sqrt(distance_summation);

						if (shortest_distance == -1.0 || distance < shortest_distance) {
							shortest_distance = distance;
							closest_cluster_num = cluster_n;
						}
					}
					clusters[closest_cluster_num]->addNode(
						new Node(instance_n, num_instances, features[instance_n], labels[instance_n], num_features));
				}
				if (INFO_OUT_CL) cout << "There are " << (int)clusters.size() << " clusters" << endl;
				//	Recalculate the centroid based on its new cluster
				if (INFO_OUT_CL) cout << "Recomputing the centroids for each cluster..." << endl;
				double total_SSE = 0.0;
				for (int cluster_n = 0; cluster_n < (int)clusters.size(); cluster_n++) {
					clusters[cluster_n]->calculateCentroid();
					double cluster_SSE = clusters[cluster_n]->calculateSSE();
					total_SSE += cluster_SSE;
					if (INFO_OUT_CL) {
						cout << "Centroid " << cluster_n << " = ";
						clusters[cluster_n]->printCentroidFeatures();
						cout << endl;
						cout << "\tMembers of Cluster (" << clusters[cluster_n]->getNumMembers() << "): ";
						clusters[cluster_n]->printNodes();
						cout << endl;
						cout << "\tCluster " << cluster_n << " SSE = " << cluster_SSE << endl;
					}
				}
				if (INFO_OUT_CL) cout << "Total SSE (sum squared-distance of each row with its centroid) = " << total_SSE << endl;
				if (INFO_OUT_CL) cout << "Davies-Bouldin Index: " << computeDaviesBouldinIndex(clusters) << endl;

				if (total_SSE < lastSSE || lastSSE == -1.0) // We've improved since last time
					num_iterations_wo_improv = 0;
				else
					num_iterations_wo_improv++;
				lastSSE = total_SSE;
			}
		if (INFO_OUT_CL)
			cout << "Convergence is detected because the sum squared-distance did not improve, and more than 3 iterations were performed" << endl;
		}
	}

	const double computeDaviesBouldinIndex(const vector<Cluster*>& clusters) {
		double r = 0.0;
		double summation_r_i = 0.0;

		// Compute r_i for each cluster
		for (int clus_m = 0; clus_m < (int)clusters.size(); clus_m++) {
			double r_i = 0.0;
			Cluster* curr_cluster = clusters[clus_m];
			
			// Find max r_i by comparing r_i between this cluster and every other cluster
			for (int clus_n = 0; clus_n < (int)clusters.size(); clus_n++) {
				if (clus_m == clus_n)
					continue;	// No need to compare same cluster against itself
				Cluster* other_cluster = clusters[clus_n];

				double new_r_i = (curr_cluster->getScatter() + other_cluster->getScatter())
									/ (computeDistanceBwClusters(curr_cluster, other_cluster));

				if (new_r_i > r_i)
					r_i = new_r_i;
			}

			summation_r_i += r_i;
		}	

		r = summation_r_i / clusters.size();
		return r; 
	}

	const double computeDistanceBwClusters(Cluster* c0, Cluster* c1) {
		const vector<double>& c0_features = c0->getCentroidFeatures();
		const vector<double>& c1_features = c1->getCentroidFeatures();
		double distance = 0.0;

		assert((int)c0_features.size() == (int)c1_features.size());
		
		for (int feat_n = 0; feat_n < (int)c0_features.size(); feat_n++) {
			double feature_c0 = c0_features[feat_n];
			double feature_c1 = c1_features[feat_n];
			double inter_dist = 0.0;
			bool nominal = c0->isCentroidFeatureNominal(feat_n);
			
			if (feature_c0 == UNKNOWN_VALUE || feature_c1 == UNKNOWN_VALUE)
				inter_dist = 1;
			else if (nominal)
				inter_dist = ((int)feature_c0 == (int)feature_c1) ? 0.0 : 1.0;
			else
				inter_dist = feature_c0 - feature_c1;

			distance += pow(inter_dist, 2);
		}

		return distance;
	}

	double computeXLinkDistance(Cluster* a, Cluster* b) {		
		const vector<Node*> nodes_in_a = a->getNodeMembers();
		const vector<Node*> nodes_in_b = b->getNodeMembers();

		// Single link -- smallest distance between any 2 points in A and B
		// Complete link -- largest distance between any 2 points in A and B
		double smallest_distance = -1.0;
		double largest_distance = -1.0;

		for (int i = 0; i < (int)nodes_in_a.size(); i++) {
			for (int j = 0; j < (int)nodes_in_b.size(); j++) {
				Node* m = nodes_in_a[i];
				Node* n = nodes_in_b[j];
				double curr_dist = m->getDistance(n->getInstanceNumber());
				if (SINGLE_LINK && (smallest_distance == -1.0 || curr_dist < smallest_distance))
					smallest_distance = curr_dist;
				else if (curr_dist > largest_distance)
					largest_distance = curr_dist;
			}
		}	
		return SINGLE_LINK ? smallest_distance : largest_distance;
	}

	void combineClusters(Cluster* a, Cluster* b) {
		a->addNodes(b->getNodeMembers());
	}

	void removeCluster(vector<Cluster*>& clusters, Cluster* to_remove) {
		int size_of_orig_clusters = clusters.size();
		vector<Cluster*> copy_clusters;
		// Remove cluster
		for (int i = 0; i < (int)clusters.size(); i++) {
			if (clusters[i] != to_remove)
				copy_clusters.push_back(clusters[i]);
		}

		// Update passed in list
		clusters.swap(copy_clusters);
		int size_of_swapped_clusters = clusters.size();
		assert(size_of_swapped_clusters == (size_of_orig_clusters - 1));
	}

	virtual void predict(const std::vector<double>& features, std::vector<double>& labels)
	{
		// Do nothing (trivial return value)
		labels[0] = 1;
	}
};

#endif // CLUSTERING_H
