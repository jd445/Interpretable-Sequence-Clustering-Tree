

#include <iostream>
#include <string>
#include <vector>
#include "SequenceReader.h"
#include <set>
#include <thread>
// #include "KMeans.h"
using namespace std;
float longest_common_substring_similarity(vector<string> a, vector<string> b)
{
	int length = 0;
	// if a.size() < b.size(), swap a, b
	if (a.size() < b.size())
	{
		vector<string> temp = a;
		a = b;
		b = temp;
	}
	int **dp = new int *[a.size() + 1];
	for (int i = 0; i < a.size() + 1; i++)
	{
		dp[i] = new int[b.size() + 1];
	}
	for (int i = 0; i < a.size() + 1; i++)
	{
		for (int j = 0; j < b.size() + 1; j++)
		{
			dp[i][j] = 0;
		}
	}
	for (int i = 1; i < a.size() + 1; i++)
	{
		for (int j = 1; j < b.size() + 1; j++)
		{
			if (a[i - 1] == b[j - 1])
			{
				dp[i][j] = dp[i - 1][j - 1] + 1;
				length = max(length, dp[i][j] + 1);
			}
			else
				dp[i][j] = 0;
		}
	}
	// delete dp
	for (int i = 0; i < a.size() + 1; i++)
	{
		delete[] dp[i];
	}
	delete[] dp;

	// float length2 = 0;
	// length2 = (float)length / (float)max(a.size(), b.size());
	return (float)length;
}

float subsequence_similarity(vector<string> a, vector<string> b)
{
	// if b is the subsequenece of a, return 1, else return 0
	int i = 0;
	int j = 0;
	while (i < a.size() && j < b.size())
	{
		if (a[i] == b[j])
		{
			i++;
			j++;
		}
		else
			i++;
	}
	if (j == b.size())
	{
		// pring a, b , 1
		// string total = "";
		// for (int i = 0; i < a.size(); i++)
		// {
		// 	total += a[i];
		// 	total += " ";
		// }
		// total += "----- ";
		// for (int i = 0; i < b.size(); i++)
		// {
		// 	total += b[i];
		// 	total += " ";
		// }
		// total += "1";
		// cout << total << endl;

		return 1;
	}
	else
		return 0;
}

class RandomProjection
{
public:
	int n_clusters;
	int iter_num;
	int max_feature_num;
	int max_feature_len;
	string similarity_measure;
	vector<vector<string>> sequence;
	set<vector<string>> itemset;
	float (*similarity_function)(vector<string>, vector<string>);

	set<vector<string>> generate_features();
	RandomProjection(string filename, int iter_num, int max_feature_num, int max_feature_len, string similarity_measure);
	vector<int> fit(int **feature_vector, int feature_size);
	~RandomProjection();
	void generate_itemset(vector<vector<string>> tempsequence);
	float **generate_feature_vector(set<vector<string>> feature);
	void generate_feature_vector_thread(int start, int end, set<vector<string>> feature, float **feature_vector);

	void data_process(vector<vector<string>> tempsequence)
	{
		// generate sequence, based on split " "
		for (int i = 0; i < tempsequence.size(); i++)
		{
			vector<string> onesequence;
			string emptyItem = "";
			for (int j = 0; j < tempsequence[i].size(); j++)
			{
				if (tempsequence[i][j] == " ")
				{
					// add item to itemset
					onesequence.push_back(emptyItem);
					emptyItem = "";
				}
				else
					emptyItem += tempsequence[i][j];
			}
			this->sequence.push_back(onesequence);
		}
	}

private:
};

RandomProjection::RandomProjection(string filename, int iter_num, int max_feature_num, int max_feature_len, string similarity_measure)
{
	this->iter_num = iter_num;
	//
	this->max_feature_num = max_feature_num;
	this->max_feature_len = max_feature_len;
	if (similarity_measure == "lcs")
	{
		cout << "similarity_measure is lcs" << endl;
		similarity_function = longest_common_substring_similarity;
	}
	else if (similarity_measure == "exitence")
	{
		cout << "similarity_measure is exitence" << endl;
		similarity_function = subsequence_similarity;
	}
	else
	{
		cout << "similarity_measure is not supported" << endl;
	}
	// randomly
	// read sequence
	SequenceReader sequenceReader;
	pair<vector<string>, vector<vector<string>>> sequencePair = sequenceReader.readSequence(filename);
	vector<string> label = sequencePair.first;
	set<string> label_set;
	for (int i = 0; i < label.size(); i++)
	{
		label_set.insert(label[i]);
	}
	this->n_clusters = label_set.size();
	vector<vector<string>> tempsequence = sequencePair.second;
	// generate itemset
	this->generate_itemset(tempsequence);
	// generate sequence, based on split " "
	this->data_process(tempsequence);
	for (int i = 0; i < this->iter_num; i++)
	{
		// generate feature
		cout << "iter: " << i + 1 << endl;
		set<vector<string>> feature = this->generate_features();

		// generate feature vector
		float **feature_vector = this->generate_feature_vector(feature);
		// save feature vector to csv file
		sequenceReader.saveFeatureVector(feature_vector, this->sequence.size(), feature.size(), "tempvec/boost" + to_string(i) + ".csv");
	}
}

// vector<int> RandomProjection::fit(int **feature_vector, int feature_size)
// {
// 	// Transfer the feature_vector vec
// 	vector<vector<int>> feature_vector_vec;
// 	for (int i = 0; i < this->sequence.size(); i++)
// 	{
// 		vector<int> temp;
// 		for (int j = 0; j < feature_size; j++)
// 		{
// 			temp.push_back(feature_vector[i][j]);
// 		}
// 		feature_vector_vec.push_back(temp);
// 	}

// 	vector<Point> all_points;
// 	for (int i = 0; i < this->sequence.size(); i++)
// 	{
// 		Point p(i + 1, feature_vector_vec[i]);
// 		all_points.push_back(p);
// 	}
// 	// fit
// 	KMeans kmeans(this->n_clusters, this->iter_num, "ok");
// 	vector<int> cluster = kmeans.run(all_points);

// 	return cluster;
// }
RandomProjection::~RandomProjection()
{
}

void RandomProjection::generate_itemset(vector<vector<string>> tempsequence)
{
	// get item set
	for (int i = 0; i < tempsequence.size(); i++)
	{
		string emptyItem = "";
		for (int j = 0; j < tempsequence[i].size(); j++)
		{
			if (tempsequence[i][j] == " ")
			{
				// add item to itemset
				vector<string> item;
				item.push_back(emptyItem);
				this->itemset.insert(item);
				emptyItem = "";
			}
			else
				emptyItem += tempsequence[i][j];
		}
	}
	// cout << "Length of the itemset " <<this->itemset.size() << endl;
	// print itemset
	// set<vector<string>>::iterator it;
	// for (it = this->itemset.begin(); it != this->itemset.end(); it++)
	// {
	// 	cout << (*it)[0] << endl;
	// }
}
set<vector<string>> RandomProjection::generate_features()
{
	// one feature: vector<string>
	// all features: vector<vector<string>>
	set<vector<string>> feature;
	// generate feature
	for (int i = 0; i < this->max_feature_num; i++)
	{
		// generate one feature
		vector<string> onefeature;
		// generate one feature
		int length_random = rand() % this->max_feature_len;
		for (int j = 0; j < length_random; j++)
		{
			// generate one item
			int random = rand() % this->itemset.size();
			set<vector<string>>::iterator it = this->itemset.begin();
			advance(it, random);
			onefeature.push_back((*it)[0]);
		}
		feature.insert(onefeature);
	}

	// print feature
	// set<vector<string>>::iterator it;
	// for (it = feature.begin(); it != feature.end(); it++)
	// {
	// 	for (int i = 0; i < (*it).size(); i++)
	// 	{
	// 		cout << (*it)[i] << " ";
	// 	}
	// 	cout << endl;
	// }
	cout << "Feature Number = " << feature.size() << endl;
	return feature;
}

float **RandomProjection::generate_feature_vector(set<vector<string>> feature)
{
	// one feature vector: vector<int>
	// all feature vector: vector<vector<int>>
	int num_of_cpu = 12;
	// cout << "Number of CPU: " << num_of_cpu << endl;
	// generate 2d array without using vector
	float **feature_vector = new float *[this->sequence.size()];
	for (int i = 0; i < this->sequence.size(); i++)
	{
		feature_vector[i] = new float[feature.size()];
	}
	vector<thread> threads;
	// split the sequence into num_of_cpu parts
	int num_of_sequence = this->sequence.size();
	int num_of_sequence_per_cpu = num_of_sequence / num_of_cpu;
	// cout << "Number of sequence per CPU: " << num_of_sequence_per_cpu << endl;
	int start = 0;
	// generate feature vector
	for (int i = 0; i < num_of_cpu; i++)
	{
		int end = start + num_of_sequence_per_cpu;
		if (i == num_of_cpu - 1)
			end = num_of_sequence;
		threads.push_back(thread(&RandomProjection::generate_feature_vector_thread, this, start, end, feature, feature_vector));
		start = end;
	}
	for (int i = 0; i < num_of_cpu; i++)
	{
		threads[i].join();
	}

	return feature_vector;
}

void RandomProjection::generate_feature_vector_thread(int start, int end, set<vector<string>> feature, float **feature_vector)
{
	for (int i = start; i < end; i++)
	{
		int j = 0;
		for (set<vector<string>>::iterator it = feature.begin(); it != feature.end(); it++)
		{
			feature_vector[i][j] = similarity_function(this->sequence[i], *it);
			// cout << feature_vector[i][j] << " ";
			j++;
		}
	}
}

int main(int argc, char **argv)
{
	// string filename = "dataset/gene.txt";
	// int iter_num = 64;
	// int max_feature_num = 256;
	// int max_feature_len = 3;

	// set parameters
	srand(unsigned(time(0)));
	string filename = argv[1];
	filename = "dataset/" + filename + ".txt";
	// filename = "dataset/temprest.txt";
	int iter_num = atoi(argv[2]);
	int max_feature_num = atoi(argv[3]);
	int max_feature_len = atoi(argv[4]);
	string similarity = argv[5];

	RandomProjection RandomProjection(filename, iter_num, max_feature_num, max_feature_len, similarity);
}
