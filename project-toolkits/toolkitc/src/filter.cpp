// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#include "filter.h"
#include "error.h"
#include <memory>
#include <math.h>
#include <algorithm>
#include <stdio.h>

using std::vector;
using std::auto_ptr;


// virtual
void Filter::train(Matrix& features, Matrix& labels)
{
	trainFilter(features, labels);
	Matrix* pTrainFeatures = filterFeatures(features);
	auto_ptr<Matrix> apTrainFeatures(pTrainFeatures);
	Matrix* pTrainLabels = filterLabels(labels);
	auto_ptr<Matrix> apTrainLabels(pTrainLabels);
	m_pInnerModel->train(*pTrainFeatures, *pTrainLabels);
}

// virtual
void Filter::predict(const std::vector<double>& features, std::vector<double>& labels)
{
	vector<double> feat2 = filterFeatures(features);
	vector<double> lab2;
	lab2.resize(filteredLabelDims());
	printf("In filter, about to predict.\n");
	m_pInnerModel->predict(feat2, lab2);
	printf("In filter, after predicting.\n");
	//labels = lab2;
	unfilterLabels(lab2, labels); //TODO this was the problem, so added line above. 
									//It was taking my normalized results, un-normalizing
									//them, and comparing them against the normalized
									//target labels in SupervisedLearner::measureAccuracy
}














// virtual
void Normalize::trainFilter(Matrix& features, Matrix& labels)
{
	m_featureMins.clear();
	m_featureMaxs.clear();
	size_t c = features.cols();
	m_featureMins.reserve(c);
	m_featureMaxs.reserve(c);
	for(size_t i = 0; i < c; i++)
	{
		if(features.valueCount(i) == 0)
		{
			// Compute the min and max
			m_featureMins.push_back(features.columnMin(i));
			m_featureMaxs.push_back(features.columnMax(i));
		}
		else
		{
			// Don't do nominal attributes
			m_featureMins.push_back(UNKNOWN_VALUE);
			m_featureMaxs.push_back(UNKNOWN_VALUE);
		}
	}
	if(labels.cols() != 1)
		ThrowError("Sorry, only one-dimensional labels are currently supported");
	if(labels.valueCount(0) == 0)
	{
		m_labelMin = labels.columnMin(0);
		m_labelMax = labels.columnMax(0);
	}
	else
	{
		m_labelMin = UNKNOWN_VALUE;
		m_labelMax = UNKNOWN_VALUE;
	}
}

// virtual
Matrix* Normalize::filterLabels(Matrix& labels)
{
	Matrix* pOut = new Matrix(labels);
	for(size_t i = 0; i < labels.rows(); i++)
	{
		vector<double> row;
		if(m_labelMin == UNKNOWN_VALUE) // if the attribute is nominal...
			row.push_back(labels[i][0]);
		else
			row.push_back((labels[i][0] - m_labelMin) / std::max(1e-12, m_labelMax - m_labelMin));
		pOut->copyRow(row);
	}
	return pOut;
}

// virtual
vector<double> Normalize::filterFeatures(const vector<double>& before)
{
	if(before.size() != m_featureMins.size())
		ThrowError("Unexpected feature vector size");
	vector<double> after;
	after.reserve(before.size());
	for(size_t c = 0; c < m_featureMins.size(); c++)
	{
		if(m_featureMins[c] == UNKNOWN_VALUE) // if the attribute is nominal...
			after.push_back(before[c]);
		else
		{
			if(before[c] == UNKNOWN_VALUE)
				after.push_back(UNKNOWN_VALUE);
			else
				after.push_back((before[c] - m_featureMins[c]) / std::max(1e-12, m_featureMaxs[c] - m_featureMins[c]));
		}
	}
	return after;
}

// virtual
void Normalize::unfilterLabels(vector<double>& before, vector<double>& after)
{
	if(before.size() != 1 || after.size() != 1)
		ThrowError("Unexpected label vector size");
	if(m_labelMin == UNKNOWN_VALUE) // if the attribute is nominal...
		after[0] = before[0];
	else
		after[0] = before[0] * (m_labelMax - m_labelMin) + m_labelMin;
}

Matrix* Normalize::filterFeatures(Matrix& features)
{
	Matrix* pOut = new Matrix(features);
	for(size_t i = 0; i < features.rows(); i++)
	{
		vector<double> row = filterFeatures(features[i]);
		pOut->copyRow(row);
	}
	return pOut;
}










// virtual
void NominalToCategorical::trainFilter(Matrix& features, Matrix& labels)
{
	// Count the new feature dims
	m_featureVals.clear();
	m_featureVals.reserve(features.cols());
	m_totalFeatureVals = 0;
	for(size_t i = 0; i < features.cols(); i++)
	{
		size_t n = features.valueCount(i);
		if(n == 0 || n > m_cap)
			n = 1;
		m_featureVals.push_back(n);
		m_totalFeatureVals += n;
	}

	// Count the new label dims
	if(labels.cols() != 1)
		ThrowError("Sorry, only one-dimensional labels (on the input side) are currently supported");
	m_labelVals = labels.valueCount(0);
	if(m_labelVals == 0 || m_labelVals > m_cap)
		m_labelVals = 1;
}

// virtual
Matrix* NominalToCategorical::filterLabels(Matrix& labels)
{
	if(labels.cols() != 1)
		ThrowError("unexpected number of label dims");
	Matrix* pOut = new Matrix();
	pOut->setSize(0, m_labelVals);
	if(m_labelVals == 1)
	{
		for(size_t i = 0; i < labels.rows(); i++)
		{
			vector<double> row;
			row.push_back(labels[i][0]);
			pOut->copyRow(row);
		}
	}
	else
	{
		for(size_t i = 0; i < labels.rows(); i++)
		{
			vector<double> row;
			row.resize(m_labelVals, 0.0);
			row[(size_t)labels[i][0]] = 1.0;
			pOut->copyRow(row);
		}
	}
	return pOut;
}

// virtual
vector<double> NominalToCategorical::filterFeatures(const vector<double>& before)
{
	if(before.size() != m_featureVals.size())
		ThrowError("Unexpected feature vector size");
	vector<double> after;
	vector<double>::const_iterator bef = before.begin();
	vector<size_t>::iterator vals = m_featureVals.begin();
	while(bef != before.end())
	{
		if(*vals == 1) // If the value is continuous...
		{
			// Continuous values are just copied straight across
			after.push_back(*bef);
		}
		else
		{
			// Nominal values are converted to a categorical distribution
			if(*bef == UNKNOWN_VALUE)
			{
				// Make a uniform distribution
				double prob = 1.0 / *vals;
				for(size_t j = 0; j < *vals; j++)
					after.push_back(prob);
			}
			else
			{
				// Give all probability to the one value
				size_t v = (size_t)*bef;
				if(v < 0 || v >= *vals)
					ThrowError("value out of range");
				size_t pos = after.size();
				for(size_t j = 0; j < *vals; j++)
					after.push_back(0.0);
				after[pos + v] = 1.0;
			}
		}
		bef++;
		vals++;
	}
	return after;
}

// virtual
void NominalToCategorical::unfilterLabels(vector<double>& before, vector<double>& after)
{
	if(before.size() != m_labelVals || after.size() != 1)
		ThrowError("Unexpected label vector size");
	if(m_labelVals == 1) // If the label is continuous...
	{
		// Continuous labels are just copied straight across
		after[0] = before[0];
	}
	else
	{
		// Find the mode, and use it for the predicted nominal label
		vector<double>::iterator mode = std::max_element(before.begin(), before.end());
		after[0] = (double)(mode - before.begin());
	}
}

Matrix* NominalToCategorical::filterFeatures(Matrix& features)
{
	Matrix* pOut = new Matrix();
	pOut->setSize(0, m_totalFeatureVals);
	for(size_t i = 0; i < features.rows(); i++)
	{
		vector<double> row = filterFeatures(features[i]);
		pOut->copyRow(row);
	}
	return pOut;
}











// virtual
void Discretize::trainFilter(Matrix& features, Matrix& labels)
{
	m_bins = size_t(floor(sqrt((double)features.rows())));
	m_featureMins.clear();
	m_featureMaxs.clear();
	size_t c = features.cols();
	m_featureMins.reserve(c);
	m_featureMaxs.reserve(c);
	for(size_t i = 0; i < c; i++)
	{
		if(features.valueCount(i) == 0)
		{
			// Compute the min and max
			m_featureMins.push_back(features.columnMin(i));
			m_featureMaxs.push_back(features.columnMax(i));
		}
		else
		{
			// Don't do nominal attributes
			m_featureMins.push_back(UNKNOWN_VALUE);
			m_featureMaxs.push_back(UNKNOWN_VALUE);
		}
	}
	if(labels.cols() != 1)
		ThrowError("Sorry, only one-dimensional labels are currently supported");
	if(labels.valueCount(0) == 0)
		ThrowError("Regression and discretization really should not be mixed");
}

// virtual
Matrix* Discretize::filterLabels(Matrix& labels)
{
	if(labels.cols() != 1)
		ThrowError("unexpected number of label dims");
	Matrix* pOut = new Matrix(labels);
	pOut->copyPart(labels, 0, 0, labels.rows(), 1);
	return pOut;
}

// virtual
std::vector<double> Discretize::filterFeatures(const std::vector<double>& before)
{
	if(before.size() != m_featureMins.size())
		ThrowError("Unexpected row size");
	vector<double> after;
	after.reserve(before.size());
	for(size_t c = 0; c < m_featureMins.size(); c++)
	{
		if(m_featureMins[c] == UNKNOWN_VALUE) // if the attribute is nominal...
			after.push_back(before[c]);
		else
		{
			if(before[c] == UNKNOWN_VALUE) // if the feature has an unknown value...
				after.push_back(UNKNOWN_VALUE);
			else
			{
				size_t bucket = size_t(floor((before[c] - m_featureMins[c]) * m_bins / (m_featureMins[c] - m_featureMins[c])));
				after.push_back(std::max((size_t)0, std::min(m_bins - 1, bucket)));
			}
		}
	}
	return after;
}

// virtual
void Discretize::unfilterLabels(std::vector<double>& before, std::vector<double>& after)
{
	after[0] = before[0];
}

Matrix* Discretize::filterFeatures(Matrix& features)
{
	Matrix* pOut = new Matrix(features);
	pOut->makeContinuousAttrsNominal(m_bins);
	for(size_t i = 0; i < features.rows(); i++)
	{
		vector<double> row = filterFeatures(features[i]);
		pOut->copyRow(row);
	}
	return pOut;
}
