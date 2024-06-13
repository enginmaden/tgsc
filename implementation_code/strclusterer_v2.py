import numpy as np
import time
import warnings
import threading
import copy
import sys

from threading import Thread, Lock
from sentence_transformers import SentenceTransformer
from finch import FINCH 
from collections import defaultdict
from fastrp import *

# python version=3.11.4
# scikit-learn=1.1.3
###################################################
#Clusterer with window size as a parameter where it is independent of batch size parameter
###################################################
# To model tweets
class tweet:
    def __init__(self, id, cluster, data, vector1,vector2, ass_cluster):
        self.id = id
        self.data = data
        self.cluster = cluster
        self.vector1 = vector1
        self.vector2 = vector2
        self.ass_cluster=ass_cluster


        
# To generate vectors using SBERT embeddings from array:Documents
def generate_sbert_emb(tws_processed,start,end):            
    #data=list(getattr(tw,'data') for tw in tweets)
    #print(data)
    #sentence_embeddings = sbert_model.encode(data)
    #beginInd=0
    #endInd=0
    thread_count=2
    threads = list()
    #while endInd<len(tweets):
        #endInd+=(int)(len(tws_processed)/thread_count)
        #print(beginInd,"-",endInd,"-",len(tweets))
        #tws_part=tws_processed[beginInd:endInd]
    #print("index",(int)(len(tws_processed)/2))
    tws_part1=copy.deepcopy(tws_processed[0:(int)(len(tws_processed)/2)])
    tws_part2=copy.deepcopy(tws_processed[(int)(len(tws_processed)/2):])
    #print(tws_part1[2].id)
    #print(tws_part2[2].id)
    #beginInd=endInd
    t1=threading.Thread(target=generate_sbert_emb_thread,args=(tws_part1,))
    threads.append(t1)
    t1.start()
    t2=threading.Thread(target=generate_sbert_emb_thread,args=(tws_part2,))
    threads.append(t2)
    t2.start()

    t1.join()
    t2.join()
    print("generating vectors completed")
    print("sizes",len(tws_part1),"---",len(tws_part2))
    #print(tws_part1[2].id)
    #print(tws_part1[1].vector1)
    #print(tws_part2[2].id)
    #print(tws_part2[1].vector1)
    
    tws_processed=tws_part1+tws_part2
    ind=0
    print("start end tweets.length twws_processed.length----",start,end,len(tweets),len(tws_processed))
    for i in range (start,end):
        tweets[i].vector1=tws_processed[ind].vector1
        ind+=1
        

        #t.join(timeout=30)
 #   for thread in threads:
 #       thread.join(timeout=30,)

def generate_sbert_emb_thread(tws_part):
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    for tw in tws_part:
        vector1=sbert_model.encode([tw.data])[0]
        #add cluster value to the end of vector for FINCH
        #with lock:
        tw.vector1=np.append(vector1,tw.cluster)
        #print("vector:",tw.vector1)

def process_tweets(tweets,offset):
    # To model the edges in graph produced by using tweets having common words 
    class edge:
        def __init__(self, tw1_id, tw2_id, weight):
            self.tw1_id = tw1_id
            self.tw2_id = tw2_id
            self.weight = weight
    # execute FINCH in generated vectors
    print("Executing FINCH for generated vectors")
    print_time()
    tweets_tmp=copy.deepcopy(tweets)
    print("printing vectors----------------------------")

    data=np.array(list(getattr(tw,'vector1') for tw in tweets_tmp))
    c, num_clust, req_c = FINCH(data, distance='euclidean')
    print("FINCH completed")
    print_time()
    
    #generate a map-dictionary for clusters
    clusters=defaultdict(list)
    #use 3rd dimension-clustering partition of FINCH result
    #add tweet to related cluster

    for i in range(len(c)):
            #clusters[c[i][1]].append(tweets[i])
            #print("appending clusters[]",c[i][1],"i",i)
            if len(c[i])>=3:
                clusters[c[i][2]].append(tweets_tmp[i])
            else:
                clusters[c[i][0]].append(tweets_tmp[i])

    #print clusters
    #print("-----------printing clusters after first FINCH------------")

##    for i in range(len(clusters)):
##        print("-----------cluster: ",i,"-----------------------------")
##        for j in range(len(clusters[i])):
##            print(clusters[i][j].id," : ",clusters[i][j].data)

    
    print("generating graphs for clusters")
    print_time()
    graphs = defaultdict(list)
    # generate graphs for each cluster
    for i in range(len(clusters)):
        for tw1 in clusters[i]:
            for tw2 in clusters[i]:
                if tw1 != tw2:
                    tw1_tokens=tw1.data.split()
                    tw2_tokens=tw2.data.split()
                    result=np.intersect1d(tw1_tokens,tw2_tokens)
                    if(len(result) > 0):
                        graphs[i].append(edge(tw1.id,tw2.id,len(result)))
    print("graph generation is completed")
    print_time()

    #print("---------printing graphs after first FINCH clustering-----------")
    #for i in range(len(graphs)):
    #     for edge in graphs[i]:
    #         line=str(edge.tw1_id)+'-('+str(edge.weight)+')-'+str(edge.tw2_id)
    #         print(line.replace(" ",""))
    #print("---------printing graphs after first FINCH clustering completed-----------")
    
    vectorAssignedNodes=[]
    #dimensions as parameter for fastRP
    dim=128
    print("Applying FASTR for generated graphs to generate embeddings")
    print_time()
    for i in range(len(graphs)):
        #print("-----------graph: ",i,"-----------------------------")
        #Generate adjacency matrix for FASTRP, rows and colums are documents-nodes and values-data are edge values
        adjRows=np.array([])
        adjCols=np.array([])
        adjData=np.array([])
        #map edge numbers to a new list starting with 0 to length(graph[i])
        edgeMap=defaultdict(list)
        tmpEdgeInd=0
        for edge in graphs[i]:
            if edgeMap[edge.tw1_id]==[]:
                edgeMap[edge.tw1_id]=tmpEdgeInd
                tmpEdgeInd+=1

            adjRows=np.append(adjRows,edgeMap[edge.tw1_id])
            if edgeMap[edge.tw2_id]==[]:
                edgeMap[edge.tw2_id]=tmpEdgeInd
                tmpEdgeInd+=1

            adjCols=np.append(adjCols,edgeMap[edge.tw2_id])
            adjData=np.append(adjData,edge.weight)
        if len(graphs[i])==0:
            continue;
        A=csc_matrix((adjData, (adjRows, adjCols)), shape=(len(graphs[i]), len(graphs[i])))
        conf = {
                'projection_method': 'sparse',
                'input_matrix': 'trans',
                'weights': [1.0, 1.0, 7.81, 45.28],
                'normalization': False,
                'dim': dim,
                'alpha': -0.628,
                'C': 1.0
        }
        U = fastrp_wrapper(A, conf)
        
        #assign generated embedding as vector to the tweet
        for i in list(edgeMap.keys()):
            ind=int(i)
            tweets_tmp[ind-offset].vector2=np.append(U[edgeMap[ind]],tweets_tmp[ind-offset].cluster)
            vectorAssignedNodes.append(ind-offset)

    print("FASTRP for graphs are completed")
    print_time()
    print("Executing second FINCH for clustering graphs")
    print_time()
    # execute second FINCH in generated vectors
    for i in range(len(tweets_tmp)):
        if i not in vectorAssignedNodes:
            tweets_tmp[i].vector2=[]
            for j in range(dim):
                tweets_tmp[i].vector2=np.append(tweets_tmp[i].vector2,0)
            tweets_tmp[i].vector2=np.append(tweets_tmp[i].vector2,tweets_tmp[i].cluster)

    data=np.array(list(getattr(tw,'vector2') for tw in tweets_tmp))
    c, num_clust, req_c = FINCH(data, distance='euclidean')
    print("Executing second FINCH completed for clustering graphs")
    print_time()

    
    #generate a map-dictionary for clusters
    clusters=defaultdict(list)
    #use 3rd dimension-clustering partition of FINCH result
    #add tweet to related cluster
    for i in range(len(c)):
            if len(c[i])>=3:
                clusters[c[i][2]].append(tweets_tmp[i])
                tweets_tmp[i].ass_cluster=c[i][2]
            else:
                clusters[c[i][0]].append(tweets_tmp[i])
                tweets_tmp[i].ass_cluster=c[i][0]
            


    print("Processing tweets completed")
    print_time()
    
    
    return clusters
##########################################################################
###################process the first window of tweets#####################
def process_first_window(file_in):
    #Read tweets from file and generate an array consisting of these tweets
    window = [next(file_in) for x in range(window_size)]
    for line in window:
        splitted=line.split(':')
        cluster=(splitted[2].split(",")[0].replace(" ",""))
        id=int((splitted[1].split(',')[0].replace('"','').replace(" ","")))-1
        data=(splitted[3].replace("\"","").replace("}",""))
        tweets.append(tweet(id,cluster,data,' ',' ',-1))
    # Generate vectors from embedding with S-BERT
    print("generating vectors for tweets with S_BERT")
    print_time()
    generate_sbert_emb(tweets,0,(window_size))
    print("vector generation completed")
    print_time()
    
    return process_tweets(tweets,0)
##########################################################################
#####merge the results of last two window clustering operations#######
def merge_clusters(curr_clusters,prev_clusters):
    #if number of clusters decreases or the same, assign the labels by increasing the previous labels with the difference
    #Ex: prev_clusters: 2,3,4 curr_clusters:0,1,2 --> curr_clusters:2,3,4
    #Ex: prev_clusters: 2,3,4 curr_clusters:0,1 --> curr_clusters:3,4
    diff=len(curr_clusters)-len(prev_clusters)

    print("printing current cluster labels--------------------")
    for i in range(len(curr_clusters)):
        for j in range(len(curr_clusters[i])):
            print("id: ", curr_clusters[i][j].id, " cluster label: ",curr_clusters[i][j].ass_cluster)
    print("end of printing current cluster labeles------------")
    print("printing previous cluster labels--------------------")
    for i in range(len(prev_clusters)):
        for j in range(len(prev_clusters[i])):
            print("id: ", prev_clusters[i][j].id, " cluster label: ",prev_clusters[i][j].ass_cluster)
    print("end of printing previous cluster labeles------------")


    if diff <= 0:
        for i in range(len(curr_clusters)):
            for j in range(len(curr_clusters[i])):
                curr_clusters[i][j].ass_cluster=int(prev_clusters[i][0].ass_cluster)+(-1 * diff)
    #if the number of clusters has increased assign the previous cluster labels to the current window and assign new labels
    #for the new clusters
    #Ex: prev_clusters: 2,3,4 curr_clusters:0,1,2,3 --> curr_clusters:2,3,4,5
    else:
        for i in range(len(prev_clusters)):
            for j in range(len(curr_clusters[i])):
                curr_clusters[i][j].ass_cluster=prev_clusters[i][0].ass_cluster
        last_cluster=int(curr_clusters[len(prev_clusters)-1][0].ass_cluster)
        #assign new labels for the remaining clusters                
        for i in range(diff):
            for j in range(len(curr_clusters[i+len(prev_clusters)])):
                curr_clusters[i+len(prev_clusters)][j].ass_cluster=last_cluster+diff

#####merge previous and current clusters with respect to the similarity matrix##########
def merge_clusters_with_sim(curr_clusters,prev_clusters,next_cluster_no):
    data=generate_sim_matrix(curr_clusters,prev_clusters)
    index_vals=[0 for x in range(len(curr_clusters))] #index/rows will be the current cluster labels
    column_vals=[0 for x in range(len(prev_clusters))] #columns will be the previous cluster labels
    for i in range(len(curr_clusters)):
        index_vals[i]=i
    for i in range(len(prev_clusters)):
        #print(prev_clusters[i][0].ass_cluster, "-",prev_clusters[i][0].id,"-",prev_clusters[i][0].data)
        if prev_clusters[i][0].ass_cluster not in column_vals:
            column_vals[i]=prev_clusters[i][0].ass_cluster
    print("printing data for similarity matrix------------------")
    print(data)
    print("end of printing data for similarity matrix-----------")
    sim_matrix=pd.DataFrame(data, index=index_vals, columns=column_vals)
    
    print("printing current cluster labels--------------------")
    for i in range(len(curr_clusters)):
        for j in range(len(curr_clusters[i])):
            print("id: ", curr_clusters[i][j].id, " cluster label: ",curr_clusters[i][j].ass_cluster," orig.cluster: ",curr_clusters[i][j].cluster)
    print("end of printing current cluster labeles------------")
    print("printing previous cluster labels--------------------")
    for i in range(len(prev_clusters)):
        for j in range(len(prev_clusters[i])):
            print("id: ", prev_clusters[i][j].id, " cluster label: ",prev_clusters[i][j].ass_cluster," orig.cluster: ",prev_clusters[i][j].cluster)
    print("end of printing previous cluster labeles------------")
    

    #print("----printing similarity matrix-----")
    #print(sim_matrix)
    #print("---- end of printing similarity matrix-----")
    #keep the mappings for current clusters and previous clusters    
    mappings=defaultdict(list)
    remove_cols=[]
    remove_rows=[]
    #Iterate while sim_matrix has a change
    while True:
        changed=False
        for row in sim_matrix.index:
            row_max=max(sim_matrix.loc[row])
            #col_max=sim_matrix[sim_matrix.idxmax(axis=1)[row]].max()
            col_max=sim_matrix.idxmax(axis=1)[row].max()

            print("row_max: ",row_max," col_max:",col_max)
            
            #if the maximum of the row->curr_cluster=maximum of the column->prev_cluster
            if row_max==col_max:
                #assign the labels of prev_cluster to the curr_cluster
                #add the assignment to mapping
                column=sim_matrix.idxmax(axis=1)[row]
                #if this row isn't mapped to any column before and column is not mapped to any row before
                if row not in remove_rows and column not in remove_cols:
                    mappings[row]=column
                remove_rows.append(row)
                remove_cols.append(column)
                changed=True
        #print(sim_matrix) 
        #remove mapped rows from sim_matrix
        print("remove rows", remove_rows)
        sim_matrix=sim_matrix.drop(remove_rows)
        #remove mapped cols from sim_matrix
        print("remove cols",remove_cols)
        print(sim_matrix)
        sim_matrix=sim_matrix.drop(remove_cols,axis=1)
        if changed is False:
            break
        remove_cols=[]
        remove_rows=[]
    #map the remaining curr_clusters to prev_clusters with max_value
    if not sim_matrix.empty:
        for row in sim_matrix.index:
            if sim_matrix.shape[1]==0:
                break;
            #find the column with max value
            col=sim_matrix.idxmax(axis=1)[row]
            print("row: ",row,"col: ",col)
            mappings[row]=col
            #drop the column from the sim_matrix
            sim_matrix=sim_matrix.drop(col,axis=1)
            #drop the row from the sim_matrix
            sim_matrix=sim_matrix.drop(row)
            print(sim_matrix)
            print("printing mappings")
            print(mappings)
            
            
    #assign the mapped previous clusters to the current clusters
    for index in mappings:
        for j in range (len(curr_clusters[index])):
            curr_clusters[index][j].ass_cluster=mappings[index]
    #if the number of current_clusters is greater than previous clusters
    #and if there are unmapped clusters, mark them as new clusters with new labels
    print("next cluster no",next_cluster_no)
    for i in range(len(curr_clusters)):
        if i not in mappings:
            for j in range(len(curr_clusters[i])):
                curr_clusters[i][j].ass_cluster=next_cluster_no
            print("i: ",i," next cluster no:",next_cluster_no)
            next_cluster_no=next_cluster_no+1

    print("printing current cluster labels after merge--------------------")
    for i in range(len(curr_clusters)):
        for j in range(len(curr_clusters[i])):
            print("id: ", curr_clusters[i][j].id, " cluster label: ",curr_clusters[i][j].ass_cluster," orig.cluster: ",curr_clusters[i][j].cluster)
    print("printing current cluster labels after merge ends--------------------")
    print_batch(curr_clusters,file_out)
    return next_cluster_no
        
###calculate the number of common tweets in current clusters and previous clusters as a matrix###
def generate_sim_matrix(curr_clusters, prev_clusters):
    matrix=[]
    for i in range(len(curr_clusters)):
        row_values=[]
        for j in range(len(prev_clusters)):
            row_values.append(calculate_sim_value(curr_clusters[i],prev_clusters[j]))
        matrix.append(row_values)
    return matrix
###calculate the number of common tweets of two clusters#####################
def calculate_sim_value(cluster1,cluster2):
    result=0
    for i in range(len(cluster1)):
        for j in range(len(cluster2)):
            if cluster1[i].id==cluster2[j].id:
                result=result+1
                continue
    return result
    
        
#print clustering results of given batch###
def print_batch(batch,file):
    for i in range(len(batch)):
        for j in range(len(batch[i])):
            #print(batch[i][j].id,"-",batch[i][j].ass_cluster,"-",batch[i][j].cluster,file=file)
            result_tweets[batch[i][j].id]=str(batch[i][j].id)+"-"+str(batch[i][j].ass_cluster)+"-"+str(batch[i][j].cluster)
        
####print time####
def print_time():
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    print(current_time)
####main function######
warnings.filterwarnings("ignore")
print_time()
# creating list
tweets = []
graph = []
result_tweets = defaultdict(list)
#batch_size=1000
#window_size=5 # in terms of batches
#open input in reading mode
#argv[1]->input file
#argv[2]->win_size
#argv[3]->batch_size
input_file=sys.argv[1]
window_size=int(sys.argv[2])
batch_size=int(sys.argv[3])
file_in = open(input_file, 'r')
file_out = open('results'+input_file+'_finch_3_win_'+str(window_size)+'_batch_'+str(batch_size)+'_multi.txt', 'w')
print("clustering started", " win size-batch size: ",window_size,"-",batch_size, "finch partition no:",3)

print_time()
#keep results of first window
print_time()
prev_clusters=process_first_window(file_in)
print(prev_clusters[0][0])
print_batch(prev_clusters,file_out)
next_cluster_no=len(prev_clusters)
print("next cluster no",next_cluster_no)
last_processed_ind=batch_size
removed_batch_count=0

print("processing first window completed")

print("printing first window cluster labels--------------------")
for i in range(len(prev_clusters)):
    for j in range(len(prev_clusters[i])):
        print("id: ", prev_clusters[i][j].id, " cluster label: ",prev_clusters[i][j].ass_cluster)
print("end of printing first window cluster labeles------------")


while True:
    try:
        batch = []
        while True:
            try:
                batch.append(next(file_in));
                if len(batch) == batch_size:
                    break
            except StopIteration:
                break
        print(batch)
        print("batch length: ",len(batch),"batch_size: ", batch_size)
        if len(batch) < batch_size:
            break
        for line in batch:
            splitted=line.split(':')
            cluster=(splitted[2].split(",")[0].replace(" ",""))
            id=int((splitted[1].split(',')[0].replace('"','').replace(" ","")))-1
            data=(splitted[3].replace("\"","").replace("}",""))
            tweets.append(tweet(id,cluster,data,' ',' ',-1))
        #print out results of the removed batch
        #print_batch(tweets[:batch_size],file_out)
        tweets=tweets[batch_size:]
        removed_batch_count+=1
        print("processing tweets ",last_processed_ind,"->",last_processed_ind+(window_size))
        print_time()
        # Generate vectors from embedding with S-BERT for the new incoming batch        
        generate_sbert_emb(tweets[(-1 * batch_size):],(len(tweets) - batch_size),len(tweets))

        #keep results of curent window
        curr_clusters=process_tweets(tweets,removed_batch_count*batch_size)
        print("processing batch completed")
        print_time()
        print("merging clusters")
        #compare and merge the results of previous and current windows
        #merge_clusters(curr_clusters,prev_clusters)
        """
        print("printing previous cluster labels before merge--------------------")
        for i in range(len(prev_clusters)):
            for j in range(len(prev_clusters[i])):
                print("id: ", prev_clusters[i][j].id, " cluster label: ",prev_clusters[i][j].ass_cluster)
        print("end of printing previous cluster labels before merge------------")
        """
        next_cluster_no=merge_clusters_with_sim(curr_clusters,prev_clusters,next_cluster_no)
        print_time()
        print("merge completed")
        #update the previous window results for the next iteration
        prev_clusters=copy.deepcopy(curr_clusters)
        last_processed_ind+=batch_size
        print_time()
        print("end of iteration")
        
    except StopIteration:
        break


#proces remaining instances
print("processing last batch...")
if len(batch) > 0:        
    for line in batch:
        splitted=line.split(':')
        cluster=(splitted[2].split(",")[0].replace(" ",""))
        id=int((splitted[1].split(',')[0].replace('"','').replace(" ","")))-1
        data=(splitted[3].replace("\"","").replace("}",""))
        tweets.append(tweet(id,cluster,data,' ',' ',-1))
    tweets=tweets[batch_size:]
    removed_batch_count+=1
    print("processing tweets ",last_processed_ind,"->",last_processed_ind+window_size+len(batch))
    print_time()
    # Generate vectors from embedding with S-BERT for the new incoming batch        
    generate_sbert_emb(tweets[(-1 * batch_size):],(len(tweets) - len(batch)),len(tweets))

    #keep results of curent window
    curr_clusters=process_tweets(tweets,removed_batch_count*batch_size)
    print("processing batch completed")
    print_time()
    print("merging clusters")
    next_cluster_no=merge_clusters_with_sim(curr_clusters,prev_clusters,next_cluster_no)
    print_time()
    print("merge completed")
    #update the previous window results for the next iteration
    prev_clusters=copy.deepcopy(curr_clusters)
    last_processed_ind+=len(batch)
    print_time()
    print("end of iteration")



#print assgined clusters in order
for i in range(len(result_tweets)):
    print(result_tweets[i],file=file_out)

file_in.close()
file_out.close()
