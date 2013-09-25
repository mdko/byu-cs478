#include <iostream>
#include <cstring>
#include <string>
#include <fstream>
#include <sstream>
#include <sys/types.h>
#include <dirent.h>
#include <errno.h>
#include <cstdlib>

using namespace std;

string convertInt(int number)
{
   stringstream ss;//create a stringstream
   ss << number;//add number to the stream
   return ss.str();//return a string with the contents of the stream
}

void setupARFF(string arffFile, int numRegionsHorizontal, int numRegionsVertical){

	ofstream file (arffFile.c_str());

	//int numSquares = numRegionsHorizontal * numRegionsVertical;
	int numSquares = (numRegionsHorizontal * 2) + ((numRegionsVertical - 2) * 2);

	file << "@relation image";
	for (int i = 1; i <= numSquares; ++i){
		file << endl << "@attribute 'Square " << i << " - hue' real" << endl <<
		"@attribute 'Square " << i << " - saturation' real" << endl <<
		"@attribute 'Square " << i << " - value' real";
	}
	file << endl << "@attribute 'Class' { 'up', 'right'}" << endl;
	//file << endl << "@attribute 'Class' { 'up', 'down', 'left', 'right'}" << endl;
	file << endl << "@data" << endl;

	file.close();
}

int main(int argc, char* argv[]){
	int numRegionsHorizontal = 32;
	int numRegionsVertical = 24;
	//string command = "python ../ImageRegionValues/image_region_values.py ";
	string command = "python ../ImageRegionValues/image_region_values_border.py ";
	string pathToPictures = "../Pictures/";
	string pathToProcessedPictures = "../imgs_2/";
	string arffFile = "test.arff";

	setupARFF(arffFile, numRegionsHorizontal, numRegionsVertical);

	string preprocess = "python ../ImageFolderPreprocessor/image_folder_preprocessor.py " + pathToPictures + 
			" " + pathToProcessedPictures + " --shrink";
	cout << "Running preprocessor: " << preprocess << endl;
	//system(preprocess.c_str());

	DIR *dp;
    struct dirent *dirp;
    if((dp  = opendir(pathToProcessedPictures.c_str())) == NULL) {
        cout << "Error(" << errno << ") opening " << pathToPictures << endl;
        return errno;
    }

    while ((dirp = readdir(dp)) != NULL) {
	        if(strcmp(dirp->d_name, ".") == 0 ||
				strcmp(dirp->d_name, "..") == 0) {
			continue;
		}
		string imageFile = pathToProcessedPictures + dirp->d_name;
		string run = command + convertInt(numRegionsHorizontal) + " " +  convertInt(numRegionsVertical) + " --img '" + imageFile + "' >> " + arffFile;
		system(run.c_str());
		cout << run << endl;
		cout << imageFile << ": Done" << endl;

		unsigned start = imageFile.find_last_of("-");
		unsigned end = imageFile.find_last_of(".");
		string orientation = imageFile.substr(start+1, end-start - 1);
		ofstream file (arffFile.c_str(), ios_base::app);
		file << ", '" << imageFile << "', '"<< orientation << "'" << endl;
		file.close();
    }
    closedir(dp);

	return 0;
}
