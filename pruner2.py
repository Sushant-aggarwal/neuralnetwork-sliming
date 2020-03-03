import numpy as np
import keras.backend as K
from keras.models import load_model
import math
from kerassurgeon import identify
from kerassurgeon.operations import delete_channels,delete_layer

model=load_model('model.h5')
model.summary()
prunefactor=0.04

def delete(l_b1_w,i):
	global model
	dele=[]
	weight_dict={}
	yo=np.shape(l_b1_w)
	for j in range(yo[1]):
		s=l_b1_w[0][j]
		#if 1>=s:
			#dele.append(j)
		filt=j
		weight_dict[filt]=s
 
		
	#print(dele)
	layer_l=model.layers[i]
	weight_dict_sort=sorted(weight_dict.items(),key=lambda kv:kv[1])
	for l in range(int(len(weight_dict_sort)*prunefactor)):
		dele.append(weight_dict_sort[l][0])

	if len(dele)!=0:
		try:
			model=delete_channels(model,layer_l,dele)
			model.save('prunned.h5')
			del model
			model=load_model('prunned.h5')
		except:
			t=0


count=0
i=1
while(True):
	print(i)
	if(i==5):
		break
	#count+=1
	#print(count)
	l_b1 = model.layers[i+1]
	l_b1_w = l_b1.get_weights()
	l_b2 = model.layers[i+2]
	l_b2_w = l_b2.get_weights()
	l_b0 = model.layers[i]
	l_b0_w = l_b0.get_weights()
	l_b3 = model.layers[i+3]
	l_b3_w = l_b3.get_weights()
	yo0=np.shape(l_b0_w)
	yo1=np.shape(l_b1_w)
	yo2=np.shape(l_b2_w)
	yo3=np.shape(l_b3_w)
	x0=yo0[0]
	x1=yo1[0]
	x2=yo2[0]
	x3=yo3[0]
	if x1==4:
		if x2==4:
			if x3==4:
				delete(l_b3_w,i)
				delete(l_b2_w,i-1)
				delete(l_b1_w,i-2)			
				i+=2
			else:
				
				delete(l_b1_w,i-1)
				delete(l_b2_w,i)
				i+=1

		else:
			try:
				delete(l_b1_w,i)
			except:
				try:
					delete(l_b1_w,i-1)
				except: 
					r=0
	i+=1
model.save('prunned.h5')
