// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef FILTER_H
#define FILTER_H

#include "matrix.h"
#include "learner.h"
#include <vector>


// A data filter wraps a supervised learner so that the supervised learner
// only has to deal with the data in filtered form.
class Filter : public SupervisedLearner
{
protected:
	SupervisedLearner* m_pInnerModel;

public:
	// Takes ownership of pInnerModel
	Filter(SupervisedLearner* pInnerModel) : m_pInnerModel(pInnerModel) {}
	virtual ~Filter() { delete(m_pInnerModel); }

	// Trains the filter, filters the training set, and then calls train on
	// the inner model.
	// (This is a required method of the SupervisedLearner class.)
	virtual void train(Matrix& features, Matrix& labels);

	// Filters the features, makes a prediction, then unfilters the label(s).
	// (This is a required method of the SupervisedLearner class.)
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels);

	// Trains the filter. (This method is called from train. It prepares
	// the filter to do its job.)
	virtual void trainFilter(Matrix& features, Matrix& labels) = 0;

	// Filters the features prior to training the inner model
	virtual Matrix* filterFeatures(Matrix& features) = 0;

	// Filters the labels prior to training the inner model
	virtual Matrix* filterLabels(Matrix& labels) = 0;

	// Returns the size of the filtered label vector.
	virtual size_t filteredLabelDims() = 0;

	// Filters a single feature vector prior to making a prediction.
	virtual std::vector<double> filterFeatures(const std::vector<double>& before) = 0;

	// Unfilters the predicted label(s).
	virtual void unfilterLabels(std::vector<double>& before, std::vector<double>& after) = 0;
};




// This normalizes a matrix so that all continuous values fall from 0 to 1.
// (It leaves nominal values unchanged.)
// (This filter is especially useful in the Perceptron and Neural Network labs.)
//
// Sometimes an ascii-art picture example is worth many words...
// ____________________                  ________________________________
// | Meta Data        |  Data            | Normalize Filter             |
// |                  |                  |                              |
// |Features          |                  | filterfeatures ____________  |
// |1)G.P.A. (0-4)----|--3.0(Real)------>|---0.75(Real)-->|Supervised|  |
// |2)gender {M,F}----|--F(Nominal)----->|---F(Nominal)-->| Learner  |  |
// |3)shoe size (3-14)|--7.5(Real)------>|--0.409(Real)-->|    v     |  |
// |                  |                  |                |    v     |  |
// |                  |                  |                | predict  |  |
// |Label             |                  | unfilterLabels |    v     |  |
// |4)I.Q. (50-200)   |<--125(Real)------|----0.5(Real)<--|          |  |
// |                  |                  |                |__________|  |
// |__________________|                  |______________________________|
//
class Normalize : public Filter
{
private:
	std::vector<double> m_featureMins;
	std::vector<double> m_featureMaxs;
	double m_labelMin;
	double m_labelMax;

public:
	Normalize(SupervisedLearner* pInnerModel) : Filter(pInnerModel) {}
	virtual ~Normalize() {}

	// Computes the min and max of each column
	virtual void trainFilter(Matrix& features, Matrix& labels);

	// Normalize continuous features in the training set
	virtual Matrix* filterFeatures(Matrix& features);

	// Normalize continuous values to fall from 0 to 1
	virtual Matrix* filterLabels(Matrix& labels);

	// Returns 1.
	virtual size_t filteredLabelDims() { return 1; }

	// Normalize continuous values to fall from 0 to 1
	virtual std::vector<double> filterFeatures(const std::vector<double>& before);

	// De-normalize continuous values back to their original range
	virtual void unfilterLabels(std::vector<double>& before, std::vector<double>& after);
};




// This converts nominal attributes to a categorical distribution.
// (This filter is especially useful in the Perceptron and Neural Network labs.)
//
// Sometimes an ascii-art picture example is worth many words...
// _______________________                  ________________________________
// | Meta Data           |  Data            | NominalToCategorical Filter  |
// |                     |                  |                              |
// |Features             |                  | filterFeatures ____________  |
// |1)age----------------|--22(Real)------->|-----22(Real)-->|Supervised|  |
// |                     |                  | /--1.0(Real)-->| Learner  |  |
// |2)gender {M,F}-------|--M(Nominal)----->|----0.0(Real)-->|          |  |
// |3)shoe size----------|--10(Real)------->|-----10(Real)-->|    v     |  |
// |                     |                  |                | predict  |  |
// |                     |                  | unfilterLabels |    v     |  |
// |Label                |                  | /--0.0(Real)<--|    v     |  |
// |4)class (UU,Harv,BYU)|<--BYU(Nominal)---|----0.1(Real)<--|          |  |
// |                     |                  | \--0.9(Real)<--|__________|  |
// |_____________________|                  |______________________________|
//
// In this example, the data comes from the following relation:
//
// @ATTRIBUTE age continuous
// @ATTRIBUTE gender { M, F }
// @ATTRIBUTE shoe_size continuous
// @ATTRIBUTE school { UofU, Harvard, BYU }
//
// The test data contains this row: < 22, M, 10, BYU >
// When this row is passed to the filtered learner,
// The filterIn method converts it to < 22, 1, 0, 10, 0, 0, 1 >.
// (Now the vector has a 3-dimensional label.)
// The supervised learner predicts the label vector < 0, 0.1, 0.9 >.
// (This is called a categorical distribution.)
// The filterOut method converts this label to the nominal value "BYU"
// (because the mode of the distribution corresponds with that value).
// The inner model has made a correct prediction because it matches the
// label in the test data.
//
class NominalToCategorical : public Filter
{
private:
	size_t m_cap;
	std::vector<size_t> m_featureVals;
	size_t m_totalFeatureVals;
	size_t m_labelVals;

public:
	NominalToCategorical(SupervisedLearner* pInnerModel) : Filter(pInnerModel), m_cap(16) {}
	virtual ~NominalToCategorical() {}

	// Decide how many dims are needed for each column
	virtual void trainFilter(Matrix& features, Matrix& labels);

	// Convert nominal features in the training set to categorical distributions
	virtual Matrix* filterFeatures(Matrix& features);

	// Convert each label to a categorical distribution
	virtual Matrix* filterLabels(Matrix& labels);

	// Returns the number of dims used to represent the label
	virtual size_t filteredLabelDims() { return m_labelVals; }

	// Convert all nominal values to a categorical distribution
	virtual std::vector<double> filterFeatures(const std::vector<double>& before);

	// Convert categorical distributions back to nominal values (by finding the mode)
	virtual void unfilterLabels(std::vector<double>& before, std::vector<double>& after);
};





// This discretizes all of the continuous values into bins. (It leaves nominal values unchanged.)
// This filter is especially useful in the naive Bayes lab.
class Discretize : public Filter
{
private:
	size_t m_bins;
	std::vector<double> m_featureMins;
	std::vector<double> m_featureMaxs;

public:
	Discretize(SupervisedLearner* pInnerModel) : Filter(pInnerModel) {}
	virtual ~Discretize() {}

	// Decides how many bins to use, and computes the min and max of each column.
	virtual void trainFilter(Matrix& features, Matrix& labels);

	// Bin continuous features in the training set
	virtual Matrix* filterFeatures(Matrix& features);

	// Discretize the training set
	virtual Matrix* filterLabels(Matrix& labels);

	// Returns 1
	virtual size_t filteredLabelDims() { return 1; }

	// Bins all the continuous values
	virtual std::vector<double> filterFeatures(const std::vector<double>& before);

	// Copies values straight across. (Throws an exception if there are continuous labels.)
	virtual void unfilterLabels(std::vector<double>& before, std::vector<double>& after);
};

#endif // FILTER_H
