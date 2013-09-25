// ----------------------------------------------------------------
// The contents of this file are distributed under the CC0 license.
// See http://creativecommons.org/publicdomain/zero/1.0/
// ----------------------------------------------------------------

#ifndef MATRIX_H
#define MATRIX_H

#include <vector>
#include <map>
#include <string>

class Rand;

#define UNKNOWN_VALUE -1e308


// This stores a matrix, A.K.A. data set, A.K.A. table. Each element is
// represented as a double value. Nominal values are represented using their
// corresponding zero-indexed enumeration value. For convenience,
// the matrix also stores some meta-data which describes the columns (or attributes)
// in the matrix. You can access the elements in the matrix using square
// brackets. (Row comes first. Column comes second. Both are zero-indexed.)
// For example:
//
// Matrix m;
// m.setSize(3, 2);
// m[0][0] = 1.0;
// m[0][1] = 1.5;
// m[1][0] = 2.3;
// m[1][1] = 3.5;
// m[2][0] = 0.0;
// m[2][1] = 1234.567;
//
class Matrix
{
private:
	// Data
	std::vector< std::vector<double> > m_data; // matrix elements

	// Meta-data
	std::string m_filename; // the name of the file
	std::vector<std::string> m_attr_name; // the name of each attribute (or column)
	std::vector< std::map<std::string, size_t> > m_str_to_enum; // value to enumeration
	std::vector< std::map<size_t, std::string> > m_enum_to_str; // enumeration to value

public:
	// Creates a 0x0 matrix. You should call loadARFF or setSize to 
	Matrix() {}

	// Copies the meta-data from that, but not the data
	Matrix(Matrix& that);

	// Destructor
	~Matrix() {}

	// Loads the matrix from an ARFF file
	void loadARFF(std::string filename);

	// Sets the size of the matrix. (Wipes out any data in the matrix.)
	void setSize(size_t rows, size_t cols);

	// Returns the number of rows in the matrix
	size_t rows() { return m_data.size(); }

	// Returns the number of columns (or attributes) in the matrix
	size_t cols() { return m_attr_name.size(); }

	// Returns the name of the specified attribute
	std::string& attrName(size_t col) { return m_attr_name[col]; }

	// Returns the name of the specified value
	std::string& attrValue(size_t attr, size_t val) { return m_enum_to_str[attr][val]; }

	// Returns a pointer to the specified row
	std::vector<double>& row(size_t index) { return m_data[index]; }

	// Returns a pointer to the specified row
	std::vector<double>& operator [](size_t index) { return m_data[index]; }

	// Shuffles the rows in the matrix. If pBuddy is non-NULL, it also
	// shuffles the rows in pBuddy, keeping the corresponding rows in
	// both matrices in the same locations.
	void shuffleRows(Rand& r, Matrix* pBuddy = NULL);

	// Returns the number of values associated with the specified attribute (or column)
	//return values interpetation: 0=continuous, 2=binary, 3=trinary, etc.
	size_t valueCount(size_t attr) { return m_enum_to_str[attr].size(); }

	// Returns the mean of the elements in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	double columnMean(size_t col);

	// Returns the min elements in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	double columnMin(size_t col);

	// Returns the min elements in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	double columnMax(size_t col);

	// Returns the most common value in the specified column. (Elements with the value UNKNOWN_VALUE are ignored.)
	double mostCommonValue(size_t col);

	// Adds a new row to this matrix that is a copy of row.
	void copyRow(std::vector<double>& row);

	// Copies the specified rectangular portion of that matrix, and adds it to the bottom of this matrix.
	// (If colCount does not match the number of columns in this matrix, then this matrix will be cleared first.)
	void copyPart(Matrix& that, size_t rowBegin, size_t colBegin, size_t rowCount, size_t colCount);

	// Sets every elements in the matrix to the specified value.
	void setAll(double val);

	// Throws an exception if that has a different number of columns than
	// this, or if one of its columns has a different number of values.
	void checkCompatibility(Matrix& that);

	// Turn all the continuous attributes into nominal attributes
	// with the specified number of values. (This method is used by the Discretize filter.)
	void makeContinuousAttrsNominal(size_t values);
};

#endif // MATRIX_H
