// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef BASELINE_H
#define BASELINE_H

#include "learner.h"
#include "rand.h"
#include "error.h"

using namespace std;

// This is a very simple supervised learning algorithm that we provide as an example
// to get you started. It finds the most common class in the training set and
// always predicts that class. (Or, if you are doing regression, it always predicts
// the centroid label.)
class BaselineLearner : public SupervisedLearner
{
private:
	Rand& m_rand; // pseudo-random number generator (not actually used by the baseline learner)
	vector<double> m_labelVec; // The label vector that this learner will always predict

public:
	BaselineLearner(Rand& r)
	: SupervisedLearner(), m_rand(r)
	{
	}

	virtual ~BaselineLearner()
	{
	}

	// Train the model to predict the labels
	virtual void train(Matrix& features, Matrix& labels)
	{
		// Check assumptions
		if(features.rows() != labels.rows())
			ThrowError("Expected the features and labels to have the same number of rows");

		// Shuffle the rows. (This won't really affect this learner, but with other
		// learners it is a good idea to shuffle the rows before doing any training.)
		features.shuffleRows(m_rand, &labels);

		// Throw away any previous training
		m_labelVec.clear();

		// Train it
		for(size_t i = 0; i < labels.cols(); i++)
		{
			size_t values = labels.valueCount(i);
			if(values == 0) // if the label is continuous...
				m_labelVec.push_back(labels.columnMean(i));
			else
				m_labelVec.push_back(labels.mostCommonValue(i));
		}
	}

	// Evaluate the features and predict the labels
	virtual void predict(const std::vector<double>& features, std::vector<double>& labels)
	{
		if(labels.size() != m_labelVec.size())
			ThrowError("Unexpected number of label dims");
		for(size_t i = 0; i < labels.size(); i++)
			labels[i] = m_labelVec[i];
	}
};


#endif // BASELINE_H
