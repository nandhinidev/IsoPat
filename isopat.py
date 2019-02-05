# Python code to calculate fine grain isotopic pattern

# Import Libraries
import numpy as np
import pandas as pd
from scipy.stats import multinomial
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot

# Functions to calculate the multinomial distribution
# Carbon probability
def cdist(carbon,ci):
    if(carbon>=ci):
        return multinomial.pmf([carbon-ci,ci],carbon,[0.989,0.0107])
    else:
        return 0
# Hydrogen probability
def hdist(hydrogen,hi):
    if(hydrogen>=hi):
        return multinomial.pmf([hydrogen-hi,hi],hydrogen,[0.999885,0.000115])
    else:
        return 0
# Nitrogen probability
def ndist(nitrogen,ni):
    if(nitrogen>=ni):
        return multinomial.pmf([nitrogen-ni,ni],nitrogen,[0.99632,0.00368])
    else:
        return 0
# Oxygen probability
def odist(oxygen,o17i,o18i):
    if(oxygen>=(o17i+o18i)):
        return multinomial.pmf([oxygen-o17i-o18i,o17i,o18i],oxygen,[0.99757,0.00038,0.00205])
    else:
        return 0
# Sulphur probability
def sdist(sulphur,s33i,s34i):
    if(sulphur>=(s33i+s34i)):
        return multinomial.pmf([sulphur-s33i-s34i,s33i,s34i],sulphur,[0.9493,0.0076,0.0429])
    else:
        return 0

# Function to build the isotopic envelepe (Possible combinations of isotopes)
def envelope():
    # Calculate all theoretical possibilities
    df1 = pd.DataFrame({'x':pd.Series(range(0,niso+1)),'key':pd.Series([1]*(niso+1))})
    df2 = df1
    # Build isotopic envelope one element at a time
    chcomb = pd.merge(df1,df2,on='key')
    df = chcomb.drop('key',axis=1)
    df['sum'] = df.apply(sum,axis=1)
    chcombf = df[df['sum'] < (niso+1)].drop('sum',axis=1)
    nit = pd.concat([pd.Series(range(0,niso+1))]*(int(len(chcombf)/niso)))
    new_index = pd.Series(range(0,len(chcombf)))
    chcombf = chcombf.reset_index(drop=True)
    chcombf['key'] = pd.Series(['C']*len(chcombf))
    ndf = pd.DataFrame({'n':pd.Series(range(0,niso+1)),'key':pd.Series(['C']*(niso+1))})
    chncomb = pd.merge(chcombf,ndf,on='key').drop('key',axis=1)
    chncomb['sum'] = chncomb.apply(sum,axis=1)
    chncombf = chncomb[chncomb['sum'] < (niso+1)].drop('sum',axis=1).reset_index(drop=True)
    chncombf['key'] = pd.Series(['C']*len(chncombf))
    odf1 = pd.DataFrame({'ox':pd.Series(range(0,niso+1)),'key':pd.Series(['C']*(niso+1))})
    odf2 = pd.DataFrame({'oy':pd.Series(range(0,niso+1))[::2]}).reset_index(drop=True)
    odf2['key'] = pd.Series(['C']*len(odf2))
    oscomb = pd.merge(odf1,odf2,on='key').drop('key',axis=1)
    oscomb['sum'] = oscomb.apply(sum,axis=1)
    oscombf = oscomb[oscomb['sum'] < (niso+1)].drop('sum',axis=1).reset_index(drop=True)
    oscombf['key'] = pd.Series(['C']*len(oscombf))
    chnocomb = pd.merge(chncombf,oscombf,on='key').drop('key',axis=1)
    chnocomb['sum'] = chnocomb.apply(sum,axis=1)
    chnocombf = chnocomb[chnocomb['sum'] < (niso+1)].drop('sum',axis=1).reset_index(drop=True)
    chnocombf['key'] = pd.Series(['C']*len(chnocombf))
    chnoscomb = pd.merge(chnocombf,oscombf,on='key').drop('key',axis=1)
    chnoscomb['sum'] = np.sum(chnoscomb,axis=1)
    chnoscombf = chnoscomb[chnoscomb['sum'] < (niso+1)]
    chnoscombf.columns = ['ciso','hiso','ntiso','o17iso','o18iso','s33iso','s34iso','sum']
    return chnoscombf

# Calculate Abundances of Isotopes
def pred_int(grid_data,niso):
    new = pd.DataFrame(columns=['mz','Intensity'])
    for i in range(1,(niso+2)):
        if((i-1)==0):
            print("Calculating Monoisotopic Peak...")
        else:
            print(i-1,"isotope...")
        mat = grid_data[grid_data['sum'] == (i-1)].reset_index(drop=True)
        mat['cprob'] = np.vectorize(cdist)(carbon,mat['ciso'])
        mat['cmass'] = mat['ciso'].apply(lambda x: 13.003355 * x + 12 * (carbon-x))
        mat['hprob'] = np.vectorize(hdist)(hydrogen,mat['hiso'])
        mat['hmass'] = mat['hiso'].apply(lambda x: 2.014102 * x + 1.007825 * (hydrogen-x))
        mat['nprob'] = np.vectorize(ndist)(nitrogen,mat['ntiso'])
        mat['nmass'] = mat['ntiso'].apply(lambda x: 15.000109 * x + 14.003074 * (nitrogen-x))
        mat['oprob'] = np.vectorize(odist)(oxygen,mat['o17iso'],mat['o18iso']/2)
        mat['omass'] = mat.apply(lambda x: 16.999132 * x['o17iso'] + 17.999160 * x['o18iso']/2 + 15.994915 * (oxygen-x['o17iso']-x['o18iso']/2),axis=1)
        mat['sprob'] = np.vectorize(sdist)(sulphur,mat['s33iso'],mat['s34iso']/2)
        mat['smass'] = mat.apply(lambda x: 32.971458 * x['s33iso'] + 33.967867 * x['s34iso']/2 + 31.972071 * (sulphur-x['s33iso']-x['s34iso']/2),axis=1)
        prob = mat.apply(lambda x: x['cprob']*x['hprob']*x['nprob']*x['oprob']*x['sprob'],axis=1)
        mass = mat.apply(lambda x: x['cmass']+x['hmass']+x['nmass']+x['omass']+x['smass'],axis=1)
        plist = pd.DataFrame({'mz':mass,'Intensity':prob})
        new = pd.concat([new,plist]).reset_index(drop=True)
    return new
    
# Read elemental composition in the following order: C,H,N,O,S,niso
comp = np.loadtxt('comp.txt',usecols=range(0,6))
carbon = comp[0].astype(int)
hydrogen = comp[1].astype(int)
nitrogen = comp[2].astype(int)
oxygen = comp[3].astype(int)
sulphur = comp[4].astype(int)
niso = comp[5].astype(int)

# Isotopic envelope elemental makeup
isogrid = envelope()
# Calculate Intensities
pair = pred_int(isogrid,niso)
 
# Write Output and plot isotopic pattern
final = pair[pair['Intensity']>0].reset_index(drop=True)
final.sort_values(by=['mz']).reset_index(drop=True)
maxint = final['Intensity'].max()
final['RA'] = final.apply(lambda x: x['Intensity']/maxint * 100,axis=1)
#final.sort_values(by=['RA'],ascending=False).reset_index(drop=True)
isopat = final[final['RA']>1].sort_values(by=['mz']).reset_index(drop=True)
# plt.bar(isopat['mz'],isopat['RA'],width=0.01,color='red')
# plt.xlabel('m/z (Da)')
# plt.ylabel('Relative Intensity')
init_notebook_mode(connected=True)
layout = go.Layout(xaxis=dict(title='m/z (Da)',tickformat='10.3f',ticks='outside'),yaxis=dict(title='Relative Abundance',ticks='outside'))
data = [go.Bar(x=isopat.mz,y=isopat.RA,marker=dict(color='red'))]
fig = go.Figure(data=data,layout=layout)
iplot(fig)
