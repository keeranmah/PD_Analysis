
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
from sklearn.utils import 

def load_all_data():
	na_list = ["Don't Know", "No Experience", "Not Applicable"]
	df_all = pd.read_csv('Phenotype_ML_All.csv', na_values = na_list)
	return df_all

def pca_alldata(df_all):
	X = df_all.drop('Unnamed: 0', axis = 1)
	X = np.array(X)
	#X = StandardScaler().fit_transform(np.array(X))
	pca = decomposition.PCA(n_components=5, whiten = True)
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
	plt.show()
	plt.clf()
	X = StandardScaler().fit_transform(np.array(X))
	X = pd.DataFrame(X)
	X=X.rename(columns = {0:'One',1:'Two',2:'Three',3:'Four',4:'Five'})	
	Xp = X[:26]
	Xe = X[25:36]
	Xpp = X[36:]
	Xn = X[26:]
	return X, Xp, Xn, Xe, Xpp, z, g, h

def GLM_five(Xp,Xn,n=2000):
	Xp['PD'] = 1
	Xn['PD'] = 0
	X = pd.concat([Xp,Xn],axis=0,ignore_index=True)
	cols =  ['PD','One','Two','Three','Four','Five']
	X = X[cols]
	niter = n
	with pm.Model() as model:
		pm.glm.glm('PD ~ One + Two + Three + Four + Five', X, family=pm.glm.families.Binomial())
		start_MAP = pm.find_MAP(fmin=fmin_powell, disp=False)
		trace = pm.sample(niter, start = start_MAP, step=pm.NUTS(), random_seed=71, progressbar=True)
	pm.summary(trace)
	df_trace = pm.trace_to_dataframe(trace)
	pd.scatter_matrix(df_trace[-1000:], diagonal='kde')
	plt.savefig('Trace_All.png')
	plt.show()
	plt.clf()
	return X,df_trace,trace

def predict_five(one,two,three,four,five,intercept, df_trace):
	p,intercept,One,Two,Three,Four,Five=df_trace[-niter//4:].mean()
	v = intercept + one*One + two*Two + three*Three + four*Four + five*Five
	return np.exp(v)/(1+np.exp(v))
	
def plot_five(X, df_trace, trace):
	plt.traceplot(trace)
	plt.savefig('Convergence_All.png')
	plt.show()
	plt.clf()
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
	
	
#perform prediction on two compressed measurements
def load_compressed_data():
	na_list = ["Don't Know", "No Experience", "Not Applicable"]
	df= pd.read_csv('Kluger_ML.csv', na_values = na_list)
	return df

def seg_compdata(df):
	C = df.drop(['Dx (class)'], axis=1)
	C_pd = C[:24]
	C_et = C[24:34]
	C_ppd = C[34:]
	C_npd = C[25:]
	#Xp = StandardScaler().fit_transform(np.array(C_pd))
	#Xn = StandardScaler().fit_transform(np.array(C_npd))
	#Xpp = StandardScaler().fit_transform(np.array(C_ppd))
	#Xet = StandardScaler().fit_transform(np.array(C_et))
	Xp=pd.DataFrame(C_pd)
	Xn=pd.DataFrame(C_npd)
	Xpp=pd.DataFrame(C_ppd)
	Xet=pd.DataFrame(C_et)
	Xp.rename(columns={0:'One',1:'Two'},inplace=True)
	Xn.rename(columns={0:'One',1:'Two'},inplace=True)
	Xpp.rename(columns={0:'One',1:'Two'},inplace=True)
	Xet.rename(columns={0:'One',1:'Two'},inplace=True)
	return C, Xp, Xn, Xpp, Xet
	
def GLM_two(Xp,Xn,n=2000):
	Xp['PD'] = 1
	Xn['PD'] = 0
	X = pd.concat([Xp,Xn],axis=0,ignore_index=True)
	cols = ['PD','One','Two']
	X = X[cols]
	niter = n
	with pm.Model() as logistic_model:
		pm.glm.glm('PD ~ One + Two', X, family=pm.glm.families.Binomial())
		start_MAP = pm.find_MAP(fmin=fmin_powell, disp=False)
		trace = pm.sample(niter, start = start_MAP, step=pm.NUTS(), random_seed=71, progressbar=True)
	pm.summary(trace)
	df_trace = pm.trace_to_dataframe(trace)
	pd.scatter_matrix(df_trace[-1000:], diagonal='kde')
	plt.savefig('Trace.png')
	plt.show()
	p,intercept,One,Two=df_trace[-niter//4:].mean()
	return X,intercept,One,Two,trace, df_trace

def predict_two(one, two, One, Two, intercept):
    v = intercept + one*One + two*Two
    return np.exp(v)/(1+np.exp(v))

def plot_two(X, trace, One, Two, intercept):
	xs = np.linspace(X.One.min(), X.One.max(), 100)
	ys = np.linspace(X.Two.min(), X.Two.max(), 100)
	P, Q = np.meshgrid(xs, ys)
	Z = predict_two(P, Q, One, Two, intercept)
	ps = predict_two(X.One, X.Two, One, Two, intercept)
	m = .5
	plt.figure(figsize=(6,6))
	plt.contour(P, Q, Z, levels =[m], colors = 'blue')
	colors = ['lime' if i else 'yellow' for i in X.PD]
	errs = ((ps < m) & X.PD) |((ps >= m) & (1-X.PD))
	plt.scatter(X.One[errs], X.Two[errs], facecolors='red', s=150)
	plt.scatter(X.One, X.Two, facecolors=colors, edgecolors='k', s=50, alpha=1);
	plt.xlabel('One', fontsize=16)
	plt.ylabel('Two', fontsize=16)
	plt.title('PD classification by two measurements', fontsize=16)
	plt.tight_layout()
	plt.savefig('PD2.png')
	plt.show()
	#plt.traceplot(trace)
	#plt.savefig('Convergence.png')
	#plt.show()
	ps[ps<m] = int(0)
	ps[ps>=m] = int(1)
	cpd = confusion_matrix(ps, X.PD)
	return df, ps, cpd
	
def RFC_two(X):
	y = X.pop('PD')
	X = np.array(X)
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
	return cpd, rfc_mse, cpdsplit, rfc_mse_split

def calc_sample(ps,alpha=.98,beta=.90):
	za = st.norm.ppf(alpha)
	zb = st.norm.ppf(beta)
	m = ps.mean()
	mua = ps[ps<m].mean()
	mub = ps[ps>=m].mean()
	sigma = (ps[ps<m].std()+ps[ps>=m].std())/2
	size = ((zb-za)*(sigma))
	
