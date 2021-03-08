##### This is a class. It contains the methods for creating and
##### minor manipulations of graphs.
#####
##### The graph is stored as an adjacency list. This is represented by a dict() structure,
##### where the keys are vertices and the values are sets with the corresponding neighborhood lists.
#####
##### Original code by : C. Seshadhri, Jan 2015
##### Modified by : Deeparnab Chakrabarty, Apr 2019

import itertools
import string
import time as timer


class graph(object):

    #### Initializing empty graph
    ####

    # Globals

    def __init__(self):
        self.vertices = set()  # Vertices are stored in a set
        self.adj_list = dict()  # Initial adjacency list is empty dictionary
        self.weights = dict()  # Initialize weights list for edges; default is 1
        self.visited = dict()
        self.time = 0
        self.first = dict()
        self.last = dict()
        self.Fcomp = dict()
        self.fcomp = 0
        self.root = dict()
        self.SCCs = list()
        self.num_SCC = 0

    #### Stats
    ####

    def num_vertices(self):  # Returns number of vertices
        return len(self.vertices)

    def num_edges(self):
        s = 0
        for v in self.vertices:
            s = s + len(self.adj_list[v])
        return s

    #### Checks if (node) is vertex of graph. Output is 1 (yes) or 0 (no).
    ####

    def isVertex(self, node):
        if node in self.vertices:  # Check if node is vertex
            return 1
        return 0  # Vertex not present!

    #### Checks if (node1, node2) is edge of graph. Output is 1 (yes) or 0 (no).
    ####

    def isEdge(self, node1, node2):
        if node1 in self.vertices:  # Check if node1 is vertex
            if node2 in self.adj_list[node1]:  # Then check if node2 is neighbor of node1
                return 1  # Edge is present!

        #        if node2 in self.vertices:               # Check if node2 is vertex
        #            if node1 in self.adj_list[node2]:    # Then check if node1 is neighbor of node2
        #                return 1                         # Edge is present!

        return 0  # Edge not present!

    ##### Return weight of edge (u,v) if present. Assumed to be directed
    #####

    def weight(self, node1, node2):
        if self.isEdge(node1, node2):
            return self.weights[(node1, node2)]
        else:
            print((node1, node2), "is not an edge")
            return None

    #### Add vertex (v)
    ####

    def Add_Vertex(self, node):
        if not self.isVertex(node):
            self.vertices.add(node)
            self.adj_list[node] = set()

    #### Add directed edge (node1, node2). Can add weights as (node1, node2, w). Default w = 1
    ####

    def Add_Edge(self, node1, node2, wt=1):
        if node1 == node2:  # Self loop, so do nothing
            return
        if not self.isVertex(node1):
            self.Add_Vertex(node1)
        if not self.isVertex(node2):
            self.Add_Vertex(node2)

        nbrs = self.adj_list[node1]  # nbrs is neighbor list of node1
        if node2 not in nbrs:  # Check if node2 already neighbor of node1
            nbrs.add(node2)  # Add node2 to this list

        self.weights[(node1, node2)] = wt  # add weight

    #### Add undirected, simple edge (node1, node2) Can add weights as (node1, node2, w). Default w = 1
    #### Calls above twice

    def Add_Und_Edge(self, node1, node2, w=1):
        self.Add_Edge(node1, node2, w)
        self.Add_Edge(node2, node1, w)

    #### Read a graph from a file with list of edges. Arguments are fname (file name), sep (separator), flag (undirected/directed). Looks for file fname.
    #### Default is DIRECTED. Set flag = "u" for UNDIRECTED
    #### Assumes that line looks like:
    ####
    #### node1 sep node2 sep weight
    ####
    ####
    #### If sep is not set, then it is just whitespace.
    #### IMPORTANT: if the first character of line is # (hash), this line is assumed to be a comment and ignored.

    def Read_Edges(self, fname, sep=None, flag=None):
        self.vertices = set()
        self.adj_list = dict()  # purge graph before reading

        f_input = open(fname, 'r')  # Open file
        list_edges = f_input.readlines()  # Read lines as list

        for each_edge in list_edges:  # Loop of each line/edge
            edge = each_edge.strip()  # Remove whitespace from edge
            if len(edge) == 0:  # If empty line, move to next line
                continue
            if edge[0] == '#':  # line starts with #, and is comment
                continue  # this is comment, so move to next line
            tokens = edge.split(sep)  # Split by sep to get tokens (nodes)
            if (len(tokens) > 2):
                w = tokens[2]
            else:
                w = 1
            if (flag == "u"):  # If flag set to undirected
                self.Add_Und_Edge(tokens[0], tokens[1], w)
            else:
                self.Add_Edge(tokens[0], tokens[1],
                              w)  # Default is directed. Add directed edge given by first two tokens

    #### Read labels on vertices of a graph
    #### Assumes that line looks like:
    ####
    #### node sep label
    ####
    #### If node not found in G.vertices, then will return an error
    ####
    #### If sep is not set, then it is just whitespace.
    #### IMPORTANT: if the first character of line is # (hash), this line is assumed to be a comment and ignored.

    def Read_Vertex_Labels(self, fname, sep=None):
        self.label = dict()

        f_input = open(fname, 'r')  # Open file
        list_pairs = f_input.readlines()  # Read lines as list

        for each_line in list_pairs:
            pair = each_line.strip()  # Remove whitespace
            if (len(pair) == 0):  # if empty line or start with #, ignore
                continue
            if pair[0] == '#':
                continue
            token = pair.split(sep)
            if (self.isVertex(token[0])):
                self.label[token[0]] = token[1]
            else:
                print("Error: Vertex not in Graph")
                return -1

    ####
    #### For some applications you may need the incidence lists to be ordered in a certain order sigma. The following code does that

    def Reorder_Edges(self, sg):
        for v in self.vertices:
            self.adj_list[v] = sorted(self.adj_list[v], key=lambda x: sg.index(x))

    #### Calculates maximum out degree
    def max_degree(self):
        max = 0
        for v in self.vertices:
            if len(self.adj_list[v]) > max:
                max = len(self.adj_list[v])
        return max

    #### Depth First Search

    def DFS(self):

        for v in self.vertices:
            self.visited[v] = 0

        global topstack
        topstack = []
        path = []

        for v in sorted(self.vertices):
            if self.visited[v] == 0:
                self.DFSHelper(v, path)


        print("first", self.first)
        return path

    def DFSHelper(self, vertex, path):
        global time

        stack = []
        stack.append(vertex)

        while len(stack) != 0:  # not empty
            u = stack[-1]
            stack.pop()
            if self.visited[u] == 0:  # not visited
                self.visited[u] = 1
                path.append(u)
                #topstack.append(u)


                # add all unvisited neighbors
                for n in sorted(self.adj_list[u], reverse=True):
                    # print("i am vertex ", u, " and my neighbors are ", sorted(self.adj_list[u]))
                    # print(sorted(self.adj_list[u]), n)
                    if self.visited[n] == 0:
                        stack.append(n)
                        #topstack.append(n)

                topstack.append(u)





        #print(topstack)

    def DFSwithFirst(self):
        for v in self.vertices:
            self.visited[v] = 0
            self.first[v] = 0
            self.last[v] = 0
            self.root[v] = 0
            self.Fcomp[v] = 0

        # print("sorted", sorted(self.vertices))
        for v in sorted(self.vertices):
            if self.visited[v] == 0:
                self.fcomp += 1
                self.root[self.fcomp] = v
                self.DFSwithFirstHelper(v)
        print(self.visited)

        print(self.first)
        print(self.last)

    def DFSwithFirstHelper(self, vertex):

        # stack = []
        # stack.append(vertex)

        self.visited[vertex] = 1
        self.Fcomp[vertex] = self.fcomp
        self.time += 1
        self.first[vertex] = self.time

        for n in sorted(self.adj_list[vertex]):
            if self.visited[n] == 0:
                # Add edge to forest
                # self.Fcomp.Add_Edge(vertex, n)
                self.DFSwithFirstHelper(n)

        self.time += 1
        self.last[vertex] = self.time

    def DFS_SCC(self, isOrder, order):
        global time

        if (isOrder):
            self.Reorder_Edges(order)

        for v in self.vertices:
            self.visited[v] = 0
            self.first[v] = 0
            self.last[v] = 0
            self.root[v] = 0
            self.Fcomp[v] = 0

        self.SCCs.append(0)

        time = 0  # reset counter

        for v in sorted(self.vertices):
            if self.visited[v] == 0:
                self.fcomp = self.fcomp + 1  # adds to tree count
                self.root[self.fcomp] = v  # sets root of current tree

                self.num_SCC += 1
                self.SCCs.append(list())
                self.DFS_SCC_Helper(v)

    def DFS_SCC_Helper(self, v):
        global time
        stack = []
        stack.append(v)
        while len(stack) != 0:
            time += 1
            u = stack[-1]
            self.Fcomp[u] = self.fcomp
            if self.visited[u] == 0:  # unvisisted
                self.visited[u] = 1
                self.first[u] = time
                self.SCCs[self.num_SCC].append(u)  # add vertex to current SCC
                for n in sorted(self.adj_list[u], reverse=True):  # reverse helps with stack ordering
                    if self.visited[n] == 0:  # unvisited
                        stack.append(n)
            else:  # this vertex has been seen at least once and shouldn't be in stack
                stack.pop()  # remove from stack because already seen
                if self.visited[u] == 1:  # visited once
                    self.visited[u] = 2
                    self.last[u] = time

    def findSCC(self):
        global time
        # run DFS in forward order
        self.DFS_SCC(False, None)
        # resorted vertices in decreasing order of lasts
        pi = sorted(self.last, key=self.last.get)
        # reverse the graph
        grev = graph()
        for v in self.vertices:
            grev.Add_Vertex(v)
            for u in self.adj_list[v]:
                grev.Add_Edge(u, v)

                # run DFS again
        for v in grev.vertices:
            grev.visited[v] = 0
            grev.first[v] = 0
            grev.last[v] = 0

        grev.SCCs.append([0])

        time = 0  # reset counter

        # run in pi order
        while len(pi) != 0:
            v = pi.pop()
            if grev.visited[v] == 0:
                grev.num_SCC += 1
                grev.SCCs.append(list())
                grev.DFS_SCC_Helper(v)

        # return number of SCC and  num vertices in max SCC
        biggest = 0
        for scc in grev.SCCs:
            if len(scc) > biggest:
                biggest = len(scc)

        print("num of SCC: ", grev.num_SCC)
        print("biggest SCC: ", biggest)

    def Top_Order(self):
        #Run DFS
        self.DFS()
        return topstack[::-1]
        # print(topstack[::-1])

        #new_stack = []
        # for n in topstack:
        #     new_stack.append(int(n))
        # sort = sorted(new_stack, reverse=True)
        # print(sort)

        # Re-order? Not sure if this works or not but what ever
        #self.Reorder_Edges(path)



    def Longest_Path(self):
        sigma = self.Top_Order()
        print(sigma)


#### Problem 1a
facebookGraph = graph()
facebookGraph.Read_Edges("facebooksample.txt")
# print("Number of vertices: ", facebookGraph.num_vertices())
# print("Number of edges: ", facebookGraph.num_edges())
# print("Maximum out degree: ", facebookGraph.max_degree())

#### Problem 1b
graph2 = graph()
graph2.Read_Edges("graph2.txt")
#print(graph2.DFS())

#### Problem 1c
graph3 = graph()
# graph3.Read_Edges("graph3.txt")
# graph3.DFSwithFirst()

#### Problem 1d
graph2.findSCC()

# start = timer.time()
# facebookGraph.findSCC()
# end = timer.time()
# print("Time taken for facebook: ", end-start)

# epin= graph()
# epin.Read_Edges("epinions.txt")
# start = timer.time()
# epin.findSCC()
# end = timer.time()
# print("Time taken for epinions: ", end-start)

#
# google = graph()
# google.Read_Edges("CS31/googlemaps.txt")
# start = timer.time()
# google.findSCC()
# end = timer.time()
# print("Time taken for google maps: ", end - start)

# Part e
fb = graph()
fb.Read_Edges("facebooksample.txt")
fb.Longest_Path()
