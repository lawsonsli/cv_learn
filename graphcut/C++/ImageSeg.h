#pragma once
#include <iostream>
#include <cstdio>
#include <queue>
#include <map>
#include <set>
#include <cassert>
#include <cmath>
#include <vector>
#include <tuple>
#include <algorithm>

#include <boost/graph/adjacency_list.hpp>

#include "CImg.h"

#define SOURCE w*h
#define SINK w*h+1

using namespace std;
using namespace boost;
using namespace cimg_library;

//typedef std::tuple<int, int> Point;
typedef std::tuple<int, int, int, int> Rect;

typedef property<vertex_index_t, int> VertexProperty;
typedef property<edge_weight_t, double, property<edge_capacity_t, double> > EdgeProperty;
typedef adjacency_list<vecS, vecS, bidirectionalS, VertexProperty, EdgeProperty> DiGraph;

typedef CImg<unsigned char> Image;

typedef boost::graph_traits<DiGraph>::vertex_descriptor v_desc;
typedef boost::graph_traits<DiGraph>::edge_descriptor e_desc;

class ImageSeg {
private:
	DiGraph graph;
	v_desc graph_src, graph_sink;

	int w, h;
	double fore_mean;
	double back_mean;

	double _diff(int x1, int y1, int x2, int y2) {
		double sigma = 20000, kappa = 2.0;
		double g1 = (img(x1, y1, 0, 0) + img(x1, y1, 0, 1) + img(x1, y1, 0, 2)) / 3.0;
		double g2 = (img(x2, y2, 0, 0) + img(x2, y2, 0, 1) + img(x2, y2, 0, 2)) / 3.0;

		return kappa * exp(-(g1 - g2) * (g1 - g2) / sigma);
	}

	double _conf(int x, int y, string st) {
		double g = (img(x, y, 0, 0) + img(x, y, 0, 1) + img(x, y, 0, 2)) / 3.0;
		double f = -log(abs(g - fore_mean) / (abs(g - fore_mean) + abs(g - back_mean)));
		double b = -log(abs(g - back_mean) / (abs(g - fore_mean) + abs(g - back_mean)));
		if (st == "source") {
			if (b > f)
				return (b - f) / (f + b);
			else
				return 0.0;
		}
		if (st == "sink") {
			if (b < f)
				return (f - b) / (f + b);
			else
				return 0.0;
		}
		return 0.0;
	}

public:
	Image img;

	ImageSeg(string img_path, Rect fore_seed, Rect back_seed) {
		img = Image(img_path.c_str());
		w = img.width(), h = img.height();

		int a, b, c, d;
		std::tie(a, b, c, d) = fore_seed;
		Image fore_img = img.get_crop(a, b, c, d);
		std::tie(a, b, c, d) = back_seed;
		Image back_img = img.get_crop(a, b, c, d);

		fore_mean = fore_img.mean();
		back_mean = back_img.mean();
	}

	~ImageSeg() { };

	void build_graph();
	set<v_desc> max_flow();
};

