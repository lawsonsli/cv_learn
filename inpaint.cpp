#include <iostream>
#include <string>
#include <cstring>
#include <stdio.h>
#include <math.h>
#include <iterator>
#include <algorithm>

#include "inpaint.h"
#include <opencv2\opencv.hpp>

using namespace std;
using namespace cv;

#define KNOWN  1
#define UNKNOWN 2
#define BORDER 3
int cc = 0;
const int PATCH_L = 2;   //模板大小为5
Vec3b PAINT;
bool Inpaint::init(string & filename)
{
	unFixed = 0;
	des = imread(filename);
	if(!des.data)
	{
		cout << "image not exist!" << endl;
		return false;
	}
	//cout << des.rows << ' ' << des.cols << endl;

	gray.create(des.rows, des.cols, CV_8UC1);
	cvtColor(des, gray, CV_BGR2GRAY);  //转换成灰度图像
	//gray = imread(filename, 0);

	mask.create(des.rows, des.cols, CV_8UC1);
	label.create(des.rows, des.cols, CV_8UC1);
	conf.create(des.rows, des.cols, CV_64F);
	prio.create(des.rows, des.cols, CV_64F);
	prio.setTo(0);
	mask.setTo(KNOWN);
	return true;
}
	
Mat& Inpaint::Process()
{
	int r, g, b;
	cout << "please input the RGB" << endl;
	cin >> r >> g >> b;
	PAINT = Vec3b(b, g, r);

	Boundary();
	Ful_info();
	Cal_Prio();
	mypoint p; 

	while(unFixed > 0)
	{	
		p = Max_Prio(); 
		
		Fix(p.x, p.y);
		Re_boundary(p.x, p.y);
		Re_prio(p.x, p.y);
	}
	return des;
}
void Inpaint::Boundary()
{
	int i, j;
	Vec3b *pix;
	uchar *mask_ptr, *last, *next;
	double *conf_ptr;
	for(i = 0; i < des.rows; ++ i)
	{
		pix = des.ptr<Vec3b>(i);
		mask_ptr = mask.ptr<uchar>(i);
		conf_ptr = conf.ptr<double>(i);
		for(j = 0; j < des.cols; ++ j)
		{
			if(pix[j] == PAINT)
			{
				unFixed ++;
				mask_ptr[j] = UNKNOWN;
				conf_ptr[j] = 0;
			}
			else
			{
				mask_ptr[j] = KNOWN;
				conf_ptr[j] = 1;
			}
		}
	}
	for(i = 0; i < mask.rows; ++ i)
	{
		mask_ptr = mask.ptr<uchar>(i);
		for(j = 0; j < mask.cols; ++ j)
		{
			if(mask_ptr[j] == UNKNOWN)
			{
				if(i==0||i==mask.rows-1||j==0||j==mask.cols-1)
					mask.at<uchar>(i ,j) = BORDER;
				else 
				{
					last = mask.ptr<uchar>(i-1);
					next = mask.ptr<uchar>(i+1);
					if(last[j] == KNOWN
						||next[j] == KNOWN
						||mask_ptr[j-1] == KNOWN
						||mask_ptr[j+1] == KNOWN)
						mask_ptr[j] = BORDER;
				}
			}
		}
	}
}
void Inpaint::Ful_info()
{
	int i, j;
	uchar *mask_ptr, *p, *label_ptr;
	for(i = 0; i < des.rows; ++ i)
	{
		mask_ptr = mask.ptr<uchar>(i);
		label_ptr = label.ptr<uchar>(i);
		for(j = 0; j < des.cols; ++ j)
		{
			if(mask_ptr[j] == KNOWN)
			{
				label_ptr[j] = true;
				if(i<PATCH_L||i>=mask.rows-PATCH_L||j<PATCH_L||j>=mask.cols-PATCH_L)
					label_ptr[j] = false;
				else 
				{
					for(int m = -PATCH_L; m <= PATCH_L; ++m)
					{
						p = mask.ptr<uchar>(i+m);
						for(int n = -PATCH_L; n <= PATCH_L; ++n)
						{
							if(p[j+n] != KNOWN)
							{
								label_ptr[j] = false;
							}
						}
					}
				}
			}
			else
				label_ptr[j] = false;
		}
	}

}
void Inpaint::Cal_Prio()
{
	int i, j;
	double c, d;
	uchar *mask_ptr;
	double *prio_ptr;
	for(i = 0; i < des.rows; ++ i)
	{
		mask_ptr = mask.ptr<uchar>(i);
		prio_ptr = prio.ptr<double>(i);
		for(j = 0; j < des.cols; ++ j)
		{
			if(mask_ptr[j] == BORDER)
			{
				c = Cal_conf(i, j);
				d = Cal_data(i, j);
				prio_ptr[j] = c*d;
			}
		}
		
	}
}
mypoint Inpaint::Max_Prio()
{
	int i, j;
	mypoint Max(0, 0, -1) ;
	uchar *mask_ptr;
	double *prio_ptr;
	for(i = 0; i < des.rows; ++ i)
	{
		mask_ptr = mask.ptr<uchar>(i);
		prio_ptr = prio.ptr<double>(i);
		for(j = 0; j < des.cols; ++ j)
		{
			if(mask_ptr[j] == BORDER)
			{
				if( prio_ptr[j] > Max.pri )
				{
					Max = mypoint(i, j, prio.at<double>(i, j));
				}
			}
		}
	}
	return Max;
}
void Inpaint::Re_boundary(int x, int y)
{
	int xbeg, xend, ybeg, yend, i, j;
	xbeg = max(0, x-PATCH_L-1);
	xend = min(des.rows-1, x+PATCH_L+1);

	ybeg = max(0, y-PATCH_L-1);
	yend = min(des.cols-1, y+PATCH_L+1);

	uchar *mask_ptr, *last, *next;

	for(i = xbeg; i <= xend; ++ i)
	{
		mask_ptr = mask.ptr<uchar>(i);
		for(j = ybeg; j <= yend; ++ j)
		{
			if(mask.at<uchar>(i ,j) == UNKNOWN)
			{
				if(i==0||i==mask.rows-1||j==0||j==mask.cols-1)
					mask.at<uchar>(i ,j) = BORDER;
				else 
				{
					last = mask.ptr<uchar>(i-1);
					next = mask.ptr<uchar>(i+1);
					if(last[j] == KNOWN
						||next[j] == KNOWN
						||mask_ptr[j-1] == KNOWN
						||mask_ptr[j+1] == KNOWN)
						mask_ptr[j] = BORDER;
				}
			}
		}
	}
}
void Inpaint::Re_prio(int x, int y)
{
	int xbeg, xend, ybeg, yend, i, j;
	xbeg = max(0, x-PATCH_L);
	xend = min(des.rows-1, x+PATCH_L);
	
	ybeg = max(0, y-PATCH_L);
	yend = min(des.cols-1, y+PATCH_L);


	uchar *mask_ptr;
	double *prio_ptr;

	for(i = max(0,xbeg-PATCH_L-1); i <= min(des.rows-1, xend+PATCH_L+1); ++ i)
	{
		mask_ptr = mask.ptr<uchar>(i);
		prio_ptr = prio.ptr<double>(i);
		for(j = max(0,ybeg-PATCH_L-1); j <= min(des.cols-1, yend+PATCH_L+1); ++ j)
		{
			if(mask_ptr[j] == BORDER)
			{
				double c = Cal_conf(i, j);
				double d = Cal_data(i, j);
				prio_ptr[j] = c*d;
			}
		}
	}
}

double Inpaint::Cal_conf(int x, int y)
{
	int i, j;
	double confid = 0;
	for(i = max(x - PATCH_L,0); i <= min(x + PATCH_L, des.rows-1); ++ i)
	{
		for(j = max(y - PATCH_L,0); j <= min(y + PATCH_L, des.cols-1); ++ j)
		{
			confid += conf.at<double>(i, j);
		}
	}
	confid /= (PATCH_L*2 + 1)*(PATCH_L*2 + 1);
	return confid;
}
double Inpaint::Cal_data(int x, int y)
{
	int i, j;
	Grad grad, temp, grad_T;
	double result, model, MAX = 0;

	grad.x = grad.y = 0;
	for( i = max(x - PATCH_L, 0); i <= min(x + PATCH_L, des.rows-1); i ++)
	{
		for(j = max(y - PATCH_L,0); j <= min(y + PATCH_L, des.cols-1); j ++)
		{
			// find the greatest gradient in this patch, this will be the gradient of this pixel(according to "detail paper")
			if(mask.at<uchar>(i, j) == KNOWN) // source pixel
			{
				//since I use four neighbors to calculate the gradient, make sure this four neighbors do not touch target region(big jump in gradient)
				if(i==0||i==des.rows-1||j==0||j==des.cols-1)
					continue;
				if( mask.at<uchar>(i, j+1)!=KNOWN||mask.at<uchar>(i, j-1)!=KNOWN||mask.at<uchar>(i+1, j)!=KNOWN||mask.at<uchar>(i-1, j)!=KNOWN)
					continue;

				temp = Get_grad(i, j); 
				model = temp.x*temp.x+temp.y*temp.y;
				if(model > MAX)
				{
					grad.x = temp.x;
					grad.y = temp.y;
					MAX = model;
				}
			}
		}
	}
	
	grad_T.x = grad.y;
	grad_T.y = -grad.x;

	//grad_T.x = grad.x;
	//grad_T.y = grad.y;

	Norm nn = Get_norm(x, y);
	result = nn.x*grad_T.x + nn.y*grad_T.y; // 内积
	result /= 255;   //即"alpha" 
	result = fabs(result);			
	return result;
}
Grad Inpaint::Get_grad(int x, int y)
{
	Grad result;
	if(x == 0)
		result.x = ( gray.at<uchar>(x+1,y) - gray.at<uchar>(x, y) );
	else if(x == gray.rows-1)
		result.x = ( gray.at<uchar>(x,y) - gray.at<uchar>(x-1, y) );
	else
		result.x = ( gray.at<uchar>(x+1,y) - gray.at<uchar>(x-1, y) ) / 2.0;
	

	if(y == 0)
		result.y = ( gray.at<uchar>(x,y+1) - gray.at<uchar>(x, y) );
	else if(y == gray.cols-1)
		result.y = ( gray.at<uchar>(x,y) - gray.at<uchar>(x, y-1) );
	else
		result.y = ( gray.at<uchar>(x,y+1) - gray.at<uchar>(x, y-1) ) / 2.0;
	return result;
}
Norm Inpaint::Get_norm(int x, int y)                
{
	Norm result;
	int num=0;
	int neighbor_x[9], neighbor_y[9];
	int record[9];
	int count = 0;
	for(int i = max(x-1,0); i <= min(x+1,des.rows-1); i++)
	{
		for(int j = max(y-1,0); j <= min(y+1,des.cols-1); j++)
		{
			count++;
			if(i == x&&j == y)continue;
			if(mask.at<uchar>(i, j)==BORDER)
			{
				num++;
				neighbor_x[num] = i;
				neighbor_y[num] = j;		
				record[num]=count;
			}
		}
	}
	if(num==0||num==1) // 如果neighbor少于2， 随机赋值
	{
		result.x = 0.6;
		result.y = 0.8;
		return result;
	}
	// draw a line between the two neighbors of the boundary pixel, then the norm is the perpendicular to the line
	int n_x = neighbor_x[2]-neighbor_x[1];
	int n_y = neighbor_y[2]-neighbor_y[1];
	int temp=n_x;
	n_x = n_y;
	n_y = temp;
	double square = pow(double(n_x*n_x + n_y*n_y),0.5);

	result.x = n_x/square;
	result.y =n_y/square;
	return result;

}

mypoint Inpaint::Get_patch(int x, int y)
{
	int xbeg, xend, ybeg, yend;
	xbeg = max(0, x-PATCH_L);
	xend = min(des.rows-1, x+PATCH_L);

	ybeg = max(0, y-PATCH_L);
	yend = min(des.cols-1, y+PATCH_L);


	//在300*300区域内寻找匹配块
	int dstx = 2, dsty = 2;
	int i, j, px, py;
	int s, t;
	int b, g, r;   //颜色
	double diff, mindiff = 999999999999;
	int mindis = 1<<20;    //曼哈顿距离
	uchar *mask_ptr = NULL;
	Vec3b *pix = NULL, *patch;
	/*
	for(px = 0; px < des.rows; ++ px)
	{
		for(py = 0; py < des.cols; ++ py)
		{
		*/
	
	for(i = -150; i < 150; ++ i)
	{
		for(j = -150; j < 150; ++ j)
		{
			px = x + i;
			py = y + j;
			diff = 0;
			if(px<0||px>=des.rows||py<0||py>=des.cols)
				continue;
			if(px+xbeg-x<0||px+xend-x>=des.rows||py+ybeg-y<0||py+yend-y>=des.cols)
				continue;

			if(label.at<uchar>(px, py) == false)
			    continue;

			int count = 0;
			for(s = xbeg-x; s <= xend-x; ++ s)
			{
				mask_ptr = mask.ptr<uchar>(x+s);
				pix = des.ptr<Vec3b>(x+s);
				patch = des.ptr<Vec3b>(px+s);
				for(t = ybeg-y; t <= yend-y; ++ t)
				{
					if(mask_ptr[y+t] == KNOWN)
					{
						b = pix[y+t][0]-patch[py+t][0];
						g = pix[y+t][1]-patch[py+t][1];
						r = pix[y+t][2]-patch[py+t][2];
						diff += (b*b + g*g + r*r);
						count ++;
					}
				}
			}
			//diff /= count;
			if(diff < mindiff)
			{
				mindiff = diff;
				dstx = px;
				dsty = py;
				mindis = abs(px-x)+abs(py-y);
			}
			else if(diff == mindiff && abs(px-x) + abs(py-y) < mindis)
			{
				mindis = abs(px-x)+abs(py-y);
				mindiff = diff;
				dstx = px;
				dsty = py;
			}
		}
	}
	return mypoint(dstx, dsty);
	
}
void Inpaint::Fix(int x, int y)
{
	int xbeg, xend, ybeg, yend;
	xbeg = max(0, x-PATCH_L);
	xend = min(des.rows-1, x+PATCH_L);

	ybeg = max(0, y-PATCH_L);
	yend = min(des.cols-1, y+PATCH_L);


	int i, j;
	double c = Cal_conf(x, y);
	Vec3b *pix, *patch;
	uchar *mask_ptr, *gray_ptr;
	double *conf_ptr;

	mypoint p = Get_patch(x, y);
	//注释代码用于平滑边界
	/*
	bool flag[5] = {false};       //如果该点在修复的交界点上
	for(i=xbeg;i<x;i++)   //左上
	{
		for(j=ybeg;j<y;j++)
		{
			if(mask.at<uchar>(i, j) == KNOWN)
				flag[1]=true;
		}
	}
	for(i=x+1;i<xend;i++)   //右上
	{
		for(j=ybeg;j<y;j++)
		{
			if(mask.at<uchar>(i, j) == KNOWN)
				flag[2]=true;
		}
	}
	for(i=xbeg;i<x;i++)   //左下
	{
		for(j=y+1;j<yend;j++)
		{
			if(mask.at<uchar>(i, j) == KNOWN)
				flag[3]=true;
		}
	}
	for(i=x+1;i<xend;i++)   //左上
	{
		for(j=y;j<yend;j++)
		{
			if(mask.at<uchar>(i, j) == KNOWN)
				flag[4]=true;
		}
	}
	if(flag[1] && flag[2] && flag[3] && flag[4])   //四个象限都有点,则该模块内的点无论知道不知道，都以patch内的点填充。达到过渡变化的目的
	{        
		for(i = xbeg; i <= xend; ++ i)     //修补填充
		{
			pix = des.ptr<Vec3b>(i);
			mask_ptr = mask.ptr<uchar>(i);
			gray_ptr = gray.ptr<uchar>(i);
			conf_ptr = conf.ptr<double>(i);
			patch = des.ptr<Vec3b>(p.x+i-x);
			for(j = ybeg; j <= yend; ++ j)
			{
				if(mask.at<uchar>(i, j) != KNOWN)
				{
					unFixed --;
				}
				Vec3b PA = patch[p.y+j-y];

				pix[j] = patch[p.y+j-y];
				mask_ptr[j] = KNOWN;
				conf_ptr[j] = c;                         //更新置信项
				gray_ptr[j] = (double)(PA[0]*9765 + PA[1]*19267 + PA[2]*3735)/32767;  //更新灰度图

			}
		}
		return;
	}
	*/
	for(i = xbeg; i <= xend; ++ i)     //修补填充
	{
		pix = des.ptr<Vec3b>(i);
		mask_ptr = mask.ptr<uchar>(i);
		gray_ptr = gray.ptr<uchar>(i);
		conf_ptr = conf.ptr<double>(i);
		patch = des.ptr<Vec3b>(p.x+i-x);
		for(j = ybeg; j <= yend; ++ j)
		{
			if(mask.at<uchar>(i, j) != KNOWN)
			{
				unFixed --;
				Vec3b PA = patch[p.y+j-y];

				pix[j] = patch[p.y+j-y];
				mask_ptr[j] = KNOWN;
				conf_ptr[j] = c;                         //更新置信项
				gray_ptr[j] = (double)(PA[0]*9765 + PA[1]*19267 + PA[2]*3735)/32767;  //更新灰度图
			}
		}
	}
}

