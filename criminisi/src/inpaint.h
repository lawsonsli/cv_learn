#include <iostream>
#include <string>
#include <opencv2\opencv.hpp>
using namespace std;
using namespace cv;

struct mypoint
{
	int x;
	int y;
	double pri;
	mypoint()  {  }
	mypoint(int _x, int _y, double _pri = 0):x(_x), y(_y), pri(_pri)  {  }
};
struct Grad
{
	double x;
	double y;
};
struct Norm
{
	double x;
	double y;
};

class Inpaint
{
private:
	Mat conf, prio, label;
	int unFixed;       //要修复的点的个数
public:
	Mat gray;
	Mat des, mask;
	Inpaint() { unFixed = 0; }
	bool init(string & filename);

	Mat& Process();
	void Boundary();
	void Ful_info();
	void Re_boundary(int x, int y);
	void Cal_Prio();
	void Re_prio(int x, int y);
	void Fix(int x, int y);

	mypoint Max_Prio();

	double Cal_conf(int x, int y);
	double Cal_data(int x, int y);
	Grad   Get_grad(int x, int y);
	Norm   Get_norm(int x, int y);
	mypoint Get_patch(int x, int y);
};
	