import copy
import networkx as nx
import numpy as np

class bayesian_network():
	
	def __init__(self,grid_ids,edges):
		self.grid_ids = grid_ids
		self.cpt_table = None
		self.G = None
		self.adj_matrix = None
		self._build_graph(grid_ids,edges)
	

	def _build_graph(self,grid_ids,edges):
		self.G = nx.Graph()
		self.G.add_nodes_from(grid_ids)
		self.G.add_edges_from(edges)
		self.adj_matrix = nx.adjacency_matrix(self.G).todense()

	def fit(self,data):
		a,b = np.where(self.adj_matrix==1)
		cpt_tables = []
		for i,j in zip(a,b):        
			parent_classes = data[self.grid_ids[i]]['current_class'].values
			child_classes = data[self.grid_ids[j]] ['current_class'].values
			cpt_count = {}
			for x,y in zip(parent_classes,child_classes):
				cpt_count[tuple([x,y])] = cpt_count.get(tuple([x,y]),0)+1
			cpt = []
			for key,value in cpt_count.items():
				summ = np.sum([cpt_count[k] for k in cpt_count.keys() if k[0]==key[0]])
				value = value/summ
				cpt.append([(key[0]),(key[1]),value])
			cpt.append([self.grid_ids[i],self.grid_ids[j]])
			cpt_tables.append(cpt)

		self.cpt_table = cpt_tables

	def predict(self,start,end,prio):
		path = list(nx.all_simple_paths(self.G,start,end, cutoff=None))[0]
		# print(path)
		for n in range(len(path)-1): 
			# print(path[n+1])
			cpt = [cpt[:-1] for cpt in self.cpt_table if cpt[-1] == [path[n],path[n+1]]][0]
			temp = copy.copy(prio)
			child = np.zeros(prio.shape)
			for item in cpt:
				if len(prio.shape)>1:
					child[:,item[1]] += temp[:,item[0]]*item[2]
				elif len(prio.shape)==1:
					child[item[1]] += temp[item[0]]*item[2]
				else:
					print("no data input")
					return None
			prio = child
	    	
		return prio


