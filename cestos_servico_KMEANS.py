#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 21:23:45 2019

@author: Victir Faro

Example of How To use Clustering 
"""
import numpy as np 
import pandas as pd 

sao_paulo = [-23.533773 , -46.625290]
qtd_alvos = 1000
n_funcionarios = 8
n_execucoes = 40


array = np.full((qtd_alvos, len(sao_paulo)), sao_paulo , dtype=np.float64)

array = np.full((qtd_alvos, len(sao_paulo)), sao_paulo , dtype=np.float64)

base = pd.DataFrame(array , columns = ['Y','X'])

base['NOISE_X'] = np.random.random((qtd_alvos,1)) / 80
base['NOISE_Y'] = np.random.random((qtd_alvos,1)) / 180

base['SIDE_X'] = np.random.choice([-1, 1], size=1000, p=[.6, .4])
base['SIDE_Y'] = np.random.choice([-1, 1], size=1000, p=[.6, .4])

base['LAT'] = base.Y + (base.NOISE_Y * base.SIDE_Y)
base['LON'] = base.X + (base.NOISE_X * base.SIDE_X)

n_clusters = int(qtd_alvos / n_funcionarios)


from sklearn.cluster import KMeans 

cluster = KMeans(n_clusters = n_clusters ) 

results = cluster.fit_predict(X=base[['LAT','LON']])

base['KMEANS'] = results

mu, sigma = 3., 0.5

debitos = np.random.lognormal(mu, sigma, qtd_alvos)*10

base['VALOR'] = debitos

base['RANK_ELEMENTO'] = base.groupby('KMEANS').VALOR.rank(ascending=False)

resumo = pd.DataFrame(base[base['RANK_ELEMENTO'] <= n_execucoes].groupby('KMEANS').VALOR.sum())
resumo['RANK'] = resumo.VALOR.rank(ascending=False)

resumo['RANK'] = resumo.RANK.astype(np.int32)

resumo['QTD'] = base[base['RANK_ELEMENTO'] <= n_execucoes].groupby('KMEANS').LAT.count()


cestos_selecionados = base[base.KMEANS.isin(resumo[resumo['RANK'] <= n_funcionarios].index.values)]
cestos_selecionados.to_csv('base.csv') # exportando no formato csv
cestos_selecionados.to_excel('base.xls') # exportando no formato xls
