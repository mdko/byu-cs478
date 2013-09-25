#include "matrix.h"
#include <iostream>

using namespace std;

int main(int argc, char* argv[])
{
	string s = argv[1];
	Matrix m;
	m.loadARFF(s);

	for (int i = 0; i < m.rows(); i++) {
		for (int j = 0; j < m.cols(); j++) {
			cout << m[i][j] << " ";
		}
		cout << endl;
	}
}
