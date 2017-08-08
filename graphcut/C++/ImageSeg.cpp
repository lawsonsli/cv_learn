#include "ImageSeg.h"


void ImageSeg::build_graph() {


	for (int x = 0; x < w; ++x) {
		for (int y = 0; y < h; ++y) {
			v_desc v = add_vertex(x * h + y, graph);
		}
	}
	graph_src = add_vertex(SOURCE, graph); //add source
	graph_sink = add_vertex(SINK, graph); //add sink

	EdgeProperty prop;
	for (int k = 0; k < w * h; k++) {
		int x = k / h, y = k % h;
		get_property_value(prop, edge_weight) = 0.0;
		get_property_value(prop, edge_capacity) = _conf(x, y, "source");
		add_edge(graph_src, k, prop, graph);
		
	}
	for (int k = 0; k < w * h; k++) {
		int x = k / h, y = k % h;
		get_property_value(prop, edge_weight) = 0.0;
		get_property_value(prop, edge_capacity) = _conf(x, y, "sink");
		add_edge(k, graph_sink, prop, graph);
	}
	

	for (int k = 0; k < w * h; k++) {
		int x = k / h, y = k % h;
		if (y < h - 1) {
			int right = k + 1;
			get_property_value(prop, edge_weight) = 0.0;
			get_property_value(prop, edge_capacity) = _diff(x, y, x, y + 1);
			add_edge(k, right, prop, graph);
			add_edge(right, k, prop, graph);
		}
		if (x < w - 1) {
			int down = (x + 1) * h + y;
			get_property_value(prop, edge_weight) = 0.0;
			get_property_value(prop, edge_capacity) = _diff(x, y, x + 1, y);

			add_edge(k, down, prop, graph);
			add_edge(down, k, prop, graph);
		}
	}
}

set<v_desc> ImageSeg::max_flow() {
	
	property_map<DiGraph, edge_capacity_t>::type edge_cap = boost::get(edge_capacity, graph);
	property_map<DiGraph, edge_weight_t>::type edge_flow = boost::get(edge_weight, graph);

	int count = 0;
	
	unsigned int size = img.width() * img.height() + 2;
	std::vector<bool> marked_nodes(size);
	std::vector<v_desc> last_node(size);
	std::vector<double> delta(size);
	//std::vector<bool> last_forward(size);

	while (true) {
		printf("%d: ", ++count);

		bool augement_chain = false;
		queue<v_desc> visit_queue;
		visit_queue.push(graph_src); //add source

		std::fill(marked_nodes.begin(), marked_nodes.end(), false);
		marked_nodes[graph_src] = true;

		last_node[graph_src] = w * h + 2; //any number > w*h + 1 is OK
		delta[graph_src] = INFINITY;
		
		while (!visit_queue.empty()) {
			v_desc i = visit_queue.front();
			visit_queue.pop();
			
			graph_traits<DiGraph>::out_edge_iterator out, out_end;
			boost::tie(out, out_end) = out_edges(i, graph);
			for (; out != out_end; out++) {
				v_desc j = target(*out, graph);
				if (marked_nodes[j])
					continue;

				double f = edge_flow[*out];
				double c = edge_cap[*out];
				if (f < c) {
					last_node[j] = i;
					//last_forward[j] = true;
					delta[j] = min(delta[i], c - f);
					visit_queue.push(j);
					marked_nodes[j] = true;
					if (j == graph_sink) {
						augement_chain = true;
						break;
					}
				}
			}

			/*
			graph_traits<DiGraph>::in_edge_iterator in, in_end;
			boost::tie(in, in_end) = in_edges(i, graph);
			for (; in != in_end; in++) {
				v_desc j = source(*in, graph);
				if (marked_nodes[j])
					continue;

				double f = edge_flow[*in];
				double c = edge_cap[*in];
				if (f > 0.0) {
					last_node[j] = i;
					last_forward[j] = false;
					delta[j] = min(delta[i], f);
					visit_queue.push(j);
					marked_nodes[j] = true;
					if (j == graph_sink) {
						augement_chain = true;
						break;
					}
				}
			}*/
			if (augement_chain)
				break;
		}

		// no augement chain exists, return the result
		if (!augement_chain) {
			set<v_desc> marked;
			for (int k = 0; k < marked_nodes.size(); k++) {
				if (marked_nodes[k])
					marked.insert(k);
			}
			return marked;
		}
			

		e_desc e;
		bool found;
		double delta_flow = delta[graph_sink];
		v_desc j = graph_sink, i;
		printf("%zd ", j);
		while (last_node[j] <= w*h + 1) {
			i = last_node[j];
			printf("%zd ", i);
			
			boost::tie(e, found) = edge(i, j, graph);
			edge_flow[e] += delta_flow;
			/*if (last_forward[j]) {
				boost::tie(e, found) = edge(i, j, graph);
				edge_flow[e] += delta_flow;
			}
			else {
				boost::tie(e, found) = edge(j, i, graph);
				edge_flow[e] -= delta_flow;
			}*/
			j = i;
		}
		printf("\n");
	}
}