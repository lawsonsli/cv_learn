#include <iostream>
#include <cstring>
#include <string>

#include "inpaint.h"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/photo/photo.hpp"
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;
#define UNKNOWN  2
Inpaint in;
void instruc()
{
	printf("you can use the mouse to get the inpaint area white,\nand press space to get it fixed\n");
	//printf("when it has fixed 100 point, it will pause until you press any key to continue\n");
}

Point prevPt;
static void onMouse(int event, int x, int y, int flags, void* )
{
	if( event == EVENT_LBUTTONUP || !(flags & EVENT_FLAG_LBUTTON) )
		prevPt = Point(-1,-1);
	else if( event == EVENT_LBUTTONDOWN )
		prevPt = Point(x,y);
	else if( event == EVENT_MOUSEMOVE && (flags & EVENT_FLAG_LBUTTON) )
	{
		Point pt(x,y);
		if( prevPt.x < 0 )
			prevPt = pt;
		line( in.mask, prevPt, pt, UNKNOWN, 5, 8, 0 );
		line( in.des, prevPt, pt, Scalar::all(255), 5, 8, 0 );
		prevPt = pt;
		imshow("image", in.des);
	}
}

int main ()
{
	cout << "please input the picture's name." << endl;
	string filename;
	cin >> filename;

	bool flag = in.init(filename);
	
	while(!flag)
	{
		cout << "please input file name again!";
		cin >> filename;
		flag = in.init(filename);
	}	
	instruc();

	namedWindow("image", 1);
	imshow("image", in.des);
	setMouseCallback( "image", onMouse, 0 );

	while(true)
	{
		char c = (char)waitKey();
		if(c == ' ')
		{
			imwrite("ruined.png", in.des);
	         	in.Process();
			break;
		}
	}

	
	namedWindow("AIEI");
	imshow("AIEI", in.des);
	waitKey(0);
	return 0;
}
