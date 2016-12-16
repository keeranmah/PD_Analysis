
import pandas as pd
import numpy as np
from sklearn import decomposition
import pymc3 as pm
import scipy.stats as stats
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import seaborn as sn
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy.optimize import fmin_powell
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix
import scipy.stats as st
from sklearn.utils import resample

def load_all_data():
	na_list = ["Don't Know", "No Experience", "Not Applicable"]
	df_all = pd.read_csv('Phenotype_ML_All.csv', na_values = na_list)
	return df_all

def pca_alldata(df_all):
	X = df_all.drop('Unnamed: 0', axis = 1)
	X = np.array(X)
	X = StandardScaler().fit_transform(np.array(X))
	pca = decomposition.PCA(n_components=10, whiten = True)
	pca.fit(X)
	z = []
	z = pca.explained_variance_ratio_.cumsum()
	g = pca.components_
	h = pca.explained_variance_
	X = pca.transform(X)
	df_X = pd.DataFrame(X)
	df_X.to_csv('Xpca.csv',index=False)
	plt.figure(1, figsize=(4, 3))
	plt.axes([.2, .2, .7, .7])
	plt.plot(z, linewidth=2)
	plt.axis('tight')
	plt.xlabel('Number of Components')
	plt.ylabel('Exp Var Ratio')
	plt.savefig('PCA.png')
	#X = StandardScaler().fit_transform(np.array(X))
	X = pd.DataFrame(X)
	X=X.rename(columns = {0:'One',1:'Two',2:'Three',3:'Four',4:'Five',5:'Six',6:'Seven',7:'Eight',8:'Nine',9:'Ten'})	
	Xp = X[:24]
	Xe = X[24:34]
	Xpp = X[34:]
	Xn = X[24:]
	return X, Xp, Xn, Xe, Xpp, z, g, h
	
def resample_ten(Xp,Xn,n=5):
	Xp['PD'] = 1
	Xn['PD'] = 0
	X = pd.concat([Xp,Xn],axis=0,ignore_index=True)
	cols = ['PD','One','Two','Three','Four','Five','Six','Seven','Eight','Nine','Ten']
	X = X[cols]
	i = 1
	XS=X
	while i <= n:
		S=resample(XS,random_state=71)
		XS=pd.concat([XS,S],axis = 0,ignore_index=True)
		i = i + 1
	return X, XS
	
def GLM_ten(XS,Xn,n=2000):
	niter = n
	with pm.Model() as model:
		pm.glm.glm('PD ~ One + Two + Three + Four + Five + Six + Seven + Eight + Nine + Ten', X, family=pm.glm.families.Binomial())
		start_MAP = pm.find_MAP(fmin=fmin_powell, disp=False)
		trace = pm.sample(niter, start = start_MAP, step=pm.NUTS(), random_seed=71, progressbar=True)
	pm.summary(trace)
	stat_metrics = pm.stats.df_summary(trace)
	stat_metrics.to_csv('Stat_Ten',index=False)
	df_trace = pm.trace_to_dataframe(trace)
	pd.scatter_matrix(df_trace[-1000:], diagonal='kde')
	plt.savefig('Trace_All.png')
	p,intercept,One,Two,Three,Four,Five,Six,Seven,Eight,Nine,Ten=df_trace[-niter//4:].mean()
	return X,df_trace,trace

def predict_ten(one,two,three,four,five,intercept, df_trace):
	p,intercept,One,Two,Three,Four,Five=df_trace[-niter//4:].mean()
	v = intercept + one*One + two*Two + three*Three + four*Four + five*Five
	return np.exp(v)/(1+np.exp(v))
	
def plot_five(X, df_trace, trace):
	plt.traceplot(trace)
	plt.savefig('Convergence_All.png')
	ps = predict_five(df.One, df.Two, df.Three, df.Four, df.Five)
	m = ps.mean()
	ps[ps<m] = int(0)
	ps[ps>=m] = int(1)
	cpd = confusion_matrix(ps, df.PD)
	return cpd, ps

def RFC_five(df):
	y = df.pop('PD')
	X = np.array(df)
	Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=71, stratify=y,train_size=.9)
	rfc = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=71, class_weight = 'balanced')
	rfc.fit(Xtrain, ytrain)
	rfc_result = rfc.predict(Xtest)
	rfc_mse_split = mean_squared_error(rfc_result, ytest)
	cpdsplit = confusion_matrix(rfc_result, ytest)
	rfc.fit(X, y)
	rfc_result = rfc.predict(X)
	rfc_mse = mean_squared_error(rfc_result, y)
	cpd = confusion_matrix(rfc_result, y)
	return cpd, rfc_mse, cpdsplit, rfs_mse_split	
	