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

def load_compressed_data():
	na_list = ["Don't Know", "No Experience", "Not Applicable"]
	df= pd.read_csv('Kluger_ML.csv', na_values = na_list)
	return df

def seg_compdata(df):
	C = df.drop(['Dx (class)'], axis=1)
	C_pd = C[:24]
	C_et = C[24:34]
	C_ppd = C[34:]
	C_npd = C[24:]
	Xp = StandardScaler().fit_transform(np.array(C_pd))
	Xn = StandardScaler().fit_transform(np.array(C_npd))
	Xpp = StandardScaler().fit_transform(np.array(C_ppd))
	Xet = StandardScaler().fit_transform(np.array(C_et))
	Xp=pd.DataFrame(Xp)
	Xn=pd.DataFrame(Xn)
	Xpp=pd.DataFrame(Xpp)
	Xet=pd.DataFrame(Xet)
	Xp.rename(columns={0:'One',1:'Two'},inplace=True)
	Xn.rename(columns={0:'One',1:'Two'},inplace=True)
	Xpp.rename(columns={0:'One',1:'Two'},inplace=True)
	Xet.rename(columns={0:'One',1:'Two'},inplace=True)
	return Xp, Xn, Xpp, Xet

def resample_two(Xp,Xn,n=5):
	Xp['PD'] = 1
	Xn['PD'] = 0
	X = pd.concat([Xp,Xn],axis=0,ignore_index=True)
	cols = ['PD','One','Two']
	X = X[cols]
	i = 1
	XS=X
	while i <= n:
		S=resample(XS,random_state=71)
		XS=pd.concat([XS,S],axis = 0,ignore_index=True)
		i = i + 1
	return X, XS
		
def GLM_two(XS,n=2000):
	niter = n
	with pm.Model() as logistic_model:
		pm.glm.glm('PD ~ One + Two', XS, family=pm.glm.families.Binomial())
		start_MAP = pm.find_MAP(fmin=fmin_powell, disp=True)
		trace = pm.sample(niter, start = start_MAP, step=pm.NUTS(), random_seed=71, progressbar=True)
	stat_metrics = pm.stats.df_summary(trace)
	df_trace = pm.trace_to_dataframe(trace)
	pd.scatter_matrix(df_trace[-1000:], diagonal='kde')
	plt.savefig('Trace.png')
	p,intercept,One,Two=df_trace[-niter//4:].mean()
	return intercept,One,Two,trace, df_trace

def predict_two(one, two, One, Two, intercept):
    v = intercept + one*One + two*Two
    return np.exp(v)/(1+np.exp(v))
	
def plot_two(X, One, Two, intercept):
	xs = np.linspace(X.One.min(), X.One.max(), 100)
	ys = np.linspace(X.Two.min(), X.Two.max(), 100)
	P, Q = np.meshgrid(xs, ys)
	Z = predict_two(P, Q, One, Two, intercept)
	ps = predict_two(X.One, X.Two, One, Two, intercept)
	m = ps.mean()
	plt.figure(figsize=(6,6))
	plt.contour(P, Q, Z, levels =[m], colors = 'blue')
	colors = ['lime' if i else 'yellow' for i in X.PD]
	errs = ((ps < m) & X.PD) |((ps >= m) & (1-X.PD))
	plt.scatter(X.One[errs], X.Two[errs], facecolors='red', s=150)
	plt.scatter(X.One, X.Two, facecolors=colors, edgecolors='k', s=50, alpha=1);
	plt.xlabel('One', fontsize=16)
	plt.ylabel('Two', fontsize=16)
	plt.title('Classification by Raw Sample', fontsize=16)
	plt.tight_layout()
	plt.savefig('PD_Single.png')
	#plt.show()
	plt.traceplot(trace)
	plt.savefig('Convergence_Single.png')
	#plt.show()
	pps=ps
	ps[ps<m] = int(0)
	ps[ps>=m] = int(1)
	cpd = confusion_matrix(ps, X.PD)
	cpd=pd.DataFrame(cpd)
	cpd.rename(columns={0:'Actual_Negative',1:'Actual_Positive'},inplace=True)
	cpd['Predicted']=['Negative','Positive']
	cm = cpd[['Predicted','Actual_Negative','Actual_Positive']]
	cm.to_csv('Confusion Matrix_NoBS', index=False)
	return ps, pps, cm
	
def RFC_two(X):
	y = XS.pop('PD')
	X = np.array(XS)
	Xtrain,Xtest,ytrain,ytest=train_test_split(X,y,random_state=71, stratify=y,train_size=.9)
	rfc = RandomForestClassifier(n_estimators=10, n_jobs=-1, random_state=71, class_weight = 'balanced')
	rfc.fit(Xtrain, ytrain)
	rfc_result = rfc.predict(Xtest)
	rfc_mse_split = mean_squared_error(rfc_result, ytest)
	#cpdsplit = confusion_matrix(rfc_result, ytest)
	rfc.fit(X, y)
	rfc_result = rfc.predict(X)
	rfc_mse = mean_squared_error(rfc_result, y)
	cmrfc = confusion_matrix(rfc_result, y)
	cmrfc=pd.DataFrame(cmrfc)
	cmrfc.rename(columns={0:'Actual_Negative',1:'Actual_Positive'},inplace=True)
	cmrfc['Predicted']=['Negative','Positive']
	cm = cmrfc[['Predicted','Actual_Negative','Actual_Positive']]
	cm.to_csv('Confusion Matrix_RFC_BS', index=False)
	rfc_features = rfc.feature_importances_
	%rfc_score_train = rfc.score(xtrain,ytrain)
	%rfc_score_test = rfc.score(xtest,ytest)
	rfc_score_predict = rfc.score(X,y)
	return cpd, rfc_mse, cpdsplit, rfc_mse_split

if __name__ == '__main__':


df = f.load_compressed_data()
C, Xp, Xn, Xpp, Xet = f.seg_compdata(df)
X,intercept,One,Two,trace, df_trace = f.GLM_two(Xp,Xn,n=2000)
df, ps, cpd = f.plot_two(df, trace, One, Two, intercept)




