import sys
from sklearn.metrics.cluster import normalized_mutual_info_score,v_measure_score,adjusted_rand_score
from sklearn.metrics import accuracy_score,homogeneity_score,completeness_score
from numpy import loadtxt
print(sys.argv[1])
lines = loadtxt(sys.argv[1], comments="#", delimiter="-", unpack=False)
assigned_clusters=lines[:,1]
actual_clusters=lines[:,2]
print("nmi score:", normalized_mutual_info_score(actual_clusters,assigned_clusters ))
print("v_measure:", v_measure_score(actual_clusters,assigned_clusters ))
print("adjusted_rand_score", adjusted_rand_score(actual_clusters,assigned_clusters ))
print("accuracy_score", accuracy_score(actual_clusters,assigned_clusters))
print("homogenetiy_score", homogeneity_score(actual_clusters,assigned_clusters))
print("completeness_score", completeness_score(actual_clusters,assigned_clusters))
