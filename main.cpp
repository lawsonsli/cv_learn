#include <iostream>
#include <cstring>
#include <string>

#include "inpaint.h"
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;
int main ()
{
	string filename;
	cout << "please input the filename of mask." << endl;
	cin >> filename;

	Inpaint in;
	while( !in.init(filename) )
	{
		cout << "please input the filename of mask." << endl;
		cin >> filename;
	}

	Mat img;
	img = in.Process();

	imwrite("fixed_by_AIEI.png", img);
	namedWindow("AIEI");
	imshow("AIEI", img);
	waitKey(0);
	return 0;
}