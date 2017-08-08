#include <iostream>

#include "ImageSeg.h"

using namespace std;
using namespace cimg_library;

int main() {
	
	//ImageSeg img_seg("./input.bmp", Rect{ 140 , 120, 280, 240 }, Rect{ 0, 0, 40, 40 });
	ImageSeg img_seg("./input2.bmp", Rect{ 35 , 30, 70, 60 }, Rect{ 0, 0, 10, 10 });
	img_seg.build_graph();
	set<v_desc> marked = img_seg.max_flow();

	cout << marked.size() << endl;
	int w = img_seg.img.width(), h = img_seg.img.height();
	for (v_desc i : marked) {
		unsigned int x = i / h, y = i % h;
		img_seg.img(x, y, 0, 0) = 255;
		img_seg.img(x, y, 0, 1) = 255;
		img_seg.img(x, y, 0, 2) = 255;
	}

	CImgDisplay win(w, h, "Image Segmentation");
	win.display(img_seg.img);
	img_seg.img.save_bmp("./result.bmp");
	system("pause");
	return 0;
}