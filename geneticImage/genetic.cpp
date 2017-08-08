#include <assert.h>
#include <iostream>
#include <iterator>
#include <list>
#include <cstdint>
#include <ctime>
#include <cstring>
#include <string>

#include <CImg.h>

using namespace std;
using namespace cimg_library;

//#define cimg_imagemagick_path  ./bin/Convert
#define randInt(x)     (rand() % (x))   // generate a random integer from 0 to x-1
#define randRange(x, y) (randInt((y)-(x))+(x))
#define randMutate(x) (randInt(settings::x) % settings::x == 1)

#define toBound(left, right, d)  min(max((left), (d)), (right))

typedef CImg<unsigned char> Image;

namespace settings
{
	int ScreenWidth;
	int ScreenHeight;
	// config
	unsigned int maxPoints = 10;
	unsigned int minPoints = 3;
	int maxAllPoints = 1500;

	unsigned int minPolygons = 10;
	unsigned int maxPolygons = 250;
	int alphaMin = 30;
	int alphaMax = 60;
	int pointMidMovement = 20;
	int pointMinMovement = 3;

	// probability of mutations
	int addPolyMutation = 700;
	int movePolyMutation = 700;
	int delPolyMutation = 1500;

	int addPointMutation = 1500;
	int delPointMutation = 1500;
	int componentMutation = 1500;
	int alphaMutation = 1500;

	int pointMinMutation = 1500;
	int pointMidMutation = 1500;
	int pointMaxMutation = 1500;
}



class DNAPoint
{
private:
	uint16_t x, y;
public:
	DNAPoint() 
	{
		x = randInt(settings::ScreenWidth);
		y = randInt(settings::ScreenHeight);
	}
	DNAPoint(int16_t xx, int16_t yy)
	{
		x = xx;
		y = yy;
	}

	uint16_t getX()
	{
		return x;
	}
	uint16_t getY()
	{
		return y;
	}

	bool mutatePoint()
	{
		bool dirty = false;

		if(randMutate(pointMaxMutation))
		{
			x = randInt(settings::ScreenWidth);
			y = randInt(settings::ScreenHeight);
			dirty = true;
		}
		if(randMutate(pointMidMutation))
		{
			DNAPoint p(randInt(settings::pointMidMovement) * 2 - settings::pointMidMovement, 
				       randInt(settings::pointMidMovement) * 2 - settings::pointMidMovement);

			*this += p;
			dirty = true;
		}
		if(randMutate(pointMinMutation))
		{
			DNAPoint p(randInt(settings::pointMinMovement) * 2 - settings::pointMinMovement, 
				       randInt(settings::pointMinMovement) * 2 - settings::pointMinMovement);

			*this += p;
			dirty = true;
		}
		// out of the bound
		if(x < 0)
			x = 0;
		else if(x > settings::ScreenWidth)
			x = settings::ScreenWidth-1;
		if(y < 0)
			y = 0;
		else if(y > settings::ScreenHeight)
			y = settings::ScreenHeight-1;

		return dirty;
	}
	DNAPoint& operator += (const DNAPoint &a)
	{
		x += a.x;
		y += a.y;
		return *this;
	}

};

class DNAPolygon
{
private:
	typedef list<DNAPoint> PointList;
	typedef PointList::iterator PointIter;
	typedef uint32_t Color;

	PointList _pointList;
	Color _color;

	bool addPoint()
	{
		/*
		if(_pointList.size() <= 1)
		{
			*this = DNAPolygon();
			return true;
		}
		*/
		if(_pointList.size() >= settings::maxPoints)
			return false;

		PointIter _pointIter = _pointList.begin();
		advance(_pointIter, randInt(_pointList.size()-1));
		PointIter prev = _pointIter++, next = _pointIter;

		DNAPoint _newPoint((prev->getX()+next->getX())/2, (prev->getY()+next->getY())/2);
		_pointList.insert(_pointIter, _newPoint);
		return true;
	}
	bool delPoint()
	{
		if(_pointList.size() <= settings::minPoints)
			return false;

		PointIter _pointIter = _pointList.begin();
		advance(_pointIter, randInt(_pointList.size()));
		_pointList.erase(_pointIter);

		return true;
	}

public:
	DNAPolygon()
	{
		//initially the polygon is a triangle
		DNAPoint origin;

		for(unsigned int i = 0; i < settings::minPoints; ++ i)
		{
			_pointList.push_back(DNAPoint( toBound(0, settings::ScreenWidth,  origin.getX()+ randRange(-3, 3)), 
				toBound(0, settings::ScreenHeight, origin.getY()+ randRange(-3, 3)) ));
		}

		_color = 0;
		_color |= (randInt(0x100)<<8);  //blue
		_color |= (randInt(0x100)<<16); //green
		_color |= (randInt(0x100)<<24); //red
		_color |= randRange(settings::alphaMin, settings::alphaMax);       //alpha
	}
	~DNAPolygon()
	{

	}
	
	bool mutatePolygon()
	{
		bool dirty = false;

		if(randMutate(addPointMutation))
			dirty |= addPoint();
		if(randMutate(delPointMutation))
			dirty |= delPoint();
		if(randMutate(componentMutation))
		{
			_color &= 0x00ffffff;
			_color |= (randInt(0x100)<<24);
			dirty = true;
		}
		if(randMutate(componentMutation))
		{
			_color &= 0xff00ffff;
			_color |= (randInt(0x100)<<16);
		}
		if(randMutate(componentMutation))
		{
			_color &= 0xffff00ff;
			_color |= (randInt(0x100)<<8);
			dirty = true;
		}
		if(randMutate(alphaMutation))
		{
			_color &= 0xffffff00;
			_color |= randRange(settings::alphaMin, settings::alphaMax);       //alpha;
			dirty = true;
		}

		PointIter _pointIter = _pointList.begin();
		for(; _pointIter != _pointList.end(); ++ _pointIter)
			dirty |= _pointIter->mutatePoint();

		return dirty;
	}
	void renderPolygon(Image& dest)
	{
		typedef CImg<int> Vertices;
		Vertices _vertices = Vertices(_pointList.size(), 2);
		PointIter _pointIter = _pointList.begin();

		cimg_forX(_vertices, i) 
		{
			_vertices(i,0) = _pointIter->getX(); 
			_vertices(i,1) = _pointIter->getY(); 
			_pointIter ++;
		}
		uint8_t _c[3], _alpha;
		_c[0]  = (unsigned char)(_color>>24);
		_c[1]  = (unsigned char)((_color&0x00ff0000) >>16);
		_c[2]  = (unsigned char)((_color&0x0000ff00) >>8);
		_alpha = (unsigned char)(_color&0x000000ff);
		// memcpy(_c, (unsigned char*)&_color, sizeof(_c));  // seems to be wrong in little endian
		dest.draw_polygon(_vertices, _c, ((float)_alpha/255));
	}
};

class DNAImage
{
private:
	typedef list<DNAPolygon> PolyList;
    typedef PolyList::iterator PolyIter;

	PolyList _polyList;

	bool addPoly()
	{
		if(_polyList.empty())
		{
			_polyList.push_back(DNAPolygon());
			return true;
		}
		if(_polyList.size() >= settings::maxPolygons)
			return false;
		
		PolyIter _polyIter = _polyList.begin();
		advance(_polyIter, randInt(_polyList.size()));
		_polyList.insert(_polyIter, DNAPolygon());

		return true;
	}
	bool delPoly()
	{
		if(_polyList.size() <= settings::minPolygons)
			return false;

		PolyIter _polyIter = _polyList.begin();
		advance(_polyIter, randInt(_polyList.size()));
		_polyList.erase(_polyIter);

		return true;
	}
	bool movePoly()
	{
		if(_polyList.size() <= 1)
			return false;

		PolyIter _polyIter = _polyList.begin();
		advance(_polyIter, randInt(_polyList.size()));
		DNAPolygon p = DNAPolygon(*_polyIter);
		_polyList.erase(_polyIter);

		_polyIter = _polyList.begin();
		advance(_polyIter, randInt(_polyList.size()));
		_polyList.insert(_polyIter, p);

		return true;
	}

public:
	DNAImage()
	{
		// _polyList.push_back(DNAPolygon());
	}
	~DNAImage()
	{

	}
	bool imageMutate()
	{
		bool dirty = false;
		if(randMutate(addPolyMutation))
			dirty |= addPoly();
		if(randMutate(delPolyMutation))
			dirty |= delPoly();
		if(randMutate(movePolyMutation))
			dirty |= movePoly();

		PolyIter _polyIter = _polyList.begin();
		for(; _polyIter != _polyList.end(); ++ _polyIter)
			dirty |= _polyIter->mutatePolygon();

		return dirty;
	}

	void renderImage(Image& dest)
	{
		dest.fill(0);

		PolyIter _polyIter = _polyList.begin();
		for(; _polyIter != _polyList.end(); ++ _polyIter)
			_polyIter->renderPolygon(dest);

	}

};

class Evolve
{
private:
	uint32_t generations;

	Image targetImage;
	Image mutateImage;

	DNAImage nowDNAImage;
	uint64_t opDiff;
	DNAImage opDNAImage;
	uint64_t imageDiff()
	{
		int w = targetImage.width(), h = targetImage.height();
		uint64_t diff = 0;
		for(int x = 0; x < w; ++ x)
		{
			for(int y = 0; y < h; ++ y)
			{
				int dr = targetImage(x, y, 0, 0) - mutateImage(x, y, 0, 0);
				int dg = targetImage(x, y, 0, 1) - mutateImage(x, y, 0, 1);
				int db = targetImage(x, y, 0, 2) - mutateImage(x, y, 0, 2);
				diff += dr*dr + dg*dg + db*db;
			}
		}
		return diff;
	}

	bool compareDNAImage()
	{
		nowDNAImage.renderImage(mutateImage);
		uint64_t diff = imageDiff();
		if(diff < opDiff)
		{
			opDiff = diff;
			return true;
		}
		return false;
	}
public:
	Evolve()
	{
		generations = 300000;
	}
	~Evolve()
	{

	}
	void setTargetImage(string path)
	{
		targetImage = Image(path.c_str());
	}
	Image& getTargetImage()
	{
		return targetImage;
	}
	void setMutateImage()
	{
		int w = targetImage.width(), h = targetImage.height();

		mutateImage = Image(w, h, 1, 3, 0); //parameter: dx, dy, dz, channels, initialed black
	}
	Image& getMutateImage()
	{
		return mutateImage;
	}
	void setGenerations(uint32_t g)
	{
		generations = g;
	}
	void evolve()
	{
		CImgDisplay evolveWin(settings::ScreenWidth, settings::ScreenHeight, "Evoloving");

		opDNAImage = nowDNAImage;

		for(uint32_t i = 0; i < generations; )
		{
			bool dirty = nowDNAImage.imageMutate();  // nowImage mutated, we don't know whether it is good.
			                                         // We should compare its differnce with opDiff
			
			if(!dirty)
				continue;

			i ++;
			if(i%10000 == 0)
				srand(unsigned int(time(NULL)));
			if(i%20 == 0)
				evolveWin.display(getMutateImage());
			/*
			if(i % 100 == 0)
				cout << i << endl;
			if(i % 1000 == 0)
			{
				char* filename;
				filename = new char[50];
				sprintf(filename, "./Gene/mona_poly_%d.bmp", i/1000);
				mutateImage.save_bmp(filename);
			}
			*/
			if(compareDNAImage()) // good mutation
			{
				opDNAImage = nowDNAImage;
			}
			else
			{
				nowDNAImage = opDNAImage;
			}
		}
	}
	void saveImage()
	{
		mutateImage.save_bmp("evolove_poly.bmp");
	}
};

int Evolvemain()
{
	srand(unsigned int(time(NULL)));

	string filename;
	cout << "输入图像路径和名字。只支持24位bmp位图" << endl;
	cin >> filename;

	uint32_t gn;
	cout << "输入进化代数" << endl;
	cin >> gn;


	Evolve mona = Evolve();
	mona.setTargetImage(filename.c_str());
	mona.setMutateImage();
	mona.setGenerations(gn);

	settings::ScreenWidth = mona.getTargetImage().width();
	settings::ScreenHeight = mona.getTargetImage().height();

	//mona.getMutateImage().display("Test");
	
	/* test draw polygon
	unsigned char color[3] = {255, 0, 0};
	Image test(200, 200, 1, 4);
	test.fill(0);
	CImg<unsigned int> point = CImg<unsigned int>(3,2);
	unsigned char p[] = {20, 20, 100, 160, 180, 40};
	int ii = 0;
	cimg_forX(point,ii) 
	{
		point(ii, 0) = p[2*ii];
		point(ii, 1) = p[2*ii+1];
	}
	
	test.draw_polygon(point, color, 0.2);
	test.display("Test draw polygon");
	*/

	mona.evolve();
	mona.saveImage();

	return 0;
}