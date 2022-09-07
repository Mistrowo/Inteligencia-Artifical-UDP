import pandas as pd
import bnlearn as bn
 
lect = pd.read_csv('dataset18.csv',         # se omite la ultima columna pq hay 7 parametros y 8 filas 
                          skiprows = 1,
                          names = ['inflacion', 'guerra_ucrania', 'precios_altos', 'escasez','no_carne','no_confort','no_alcohol','omitida' ])


print(lect)

mo = bn.structure_learning.fit(lect)


model_update = bn.parameter_learning.fit(mo, lect)
grafactualizado= bn.plot(model_update, interactive=True, params_interactive={'notebook':True})

#Inferencia 1
consult1 = bn.inference.fit(model_update, variables=['inflacion'], evidence={'precios_altos':1})
consult2 = bn.inference.fit(model_update, variables=['inflacion'], evidence={'precios_altos':0})
#Inferencia 2
consult3 = bn.inference.fit(model_update, variables=['escasez'], evidence={'guerra_ucrania':1})
consult4 = bn.inference.fit(model_update, variables=['escasez'], evidence={'guerra_ucrania':0})
#Inferencia 3
consult5 = bn.inference.fit(model_update, variables=['escasez','guerra_ucrania'], evidence={'precios_altos':1})
consult6 = bn.inference.fit(model_update, variables=['escasez','guerra_ucrania'], evidence={'precios_altos':0})