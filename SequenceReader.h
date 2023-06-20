#pragma once
#include <iostream>
#include <string>
#include <vector>
#include<fstream>
using namespace std;

// SequenceReader
// read sequence from file
class SequenceReader
{
public:
	SequenceReader();
	~SequenceReader();

	// read sequence from file
	// @param fileName: file name
	// @return: two things: label, sequence
	pair<vector<string>, vector<vector<string>>> readSequence(string fileName);
    void saveFeatureVector(float **featureVector, int line_num, int column_num, string fileName);
	void saveClusterResult(vector<int> clusterResult, string fileName);

private:

};

SequenceReader::SequenceReader()
{
}

SequenceReader::~SequenceReader()
{
}

pair<vector<string>, vector<vector<string>>> SequenceReader::readSequence(string fileName)
{
	vector<string> label;
	vector<vector<string>> sequence;
	ifstream fin(fileName);
	string line;
	// label: before '\t'
	// sequence: after '\t'
	while (getline(fin, line))
	{
		int pos = line.find('\t');
		label.push_back(line.substr(0, pos));
		string sequenceStr = line.substr(pos + 1);
		vector<string> sequenceVector;
		for (int i = 0; i < sequenceStr.size(); i++)
		{
			sequenceVector.push_back(sequenceStr.substr(i, 1));
		}
		sequence.push_back(sequenceVector);
	}
	return make_pair(label, sequence);
}


void SequenceReader::saveFeatureVector(float** featureVector, int line_num, int column_num, string fileName)
{
	ofstream fout(fileName);
	for (int i = 0; i < line_num; i++)
	{
		for (int j = 0; j < column_num; j++)
		{
			fout << featureVector[i][j] << " ";
		}
		fout << endl;
	}
	fout.close();
}


void SequenceReader::saveClusterResult(vector<int> clusterResult, string fileName)
{
	ofstream fout(fileName);
	for (int i = 0; i < clusterResult.size(); i++)
	{
		fout << clusterResult[i] << endl;
	}
	fout.close();
}