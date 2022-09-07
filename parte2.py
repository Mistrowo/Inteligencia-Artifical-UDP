

import numpy as np

p=np.array([
            [0.25  , 0.06  , 0.08  , 0.15   ,0.04  , 0.02  , 0.15  , 0.15  ,0.10],
            [0.12  , 0.00  , 0.05 ,  0.24 ,  0.14 ,  0.04 ,  0.27  , 0.07 , 0.07 ],
            [0.15 , 0.15  , 0.10 ,  0.22 ,  0.01  , 0.02  , 0.15 , 0.10  ,0.10   ],
            [0.05  , 0.13  , 0.05  , 0.30  , 0.10   ,0.10   ,0.22 ,  0.05  ,0.00  ],
            [0.18  , 0.20   ,0.07  , 0.20   ,0.15   ,0.05   ,0.05  , 0.05  ,0.05  ],
            [0.20   ,0.10 ,  0.20  , 0.05  , 0.05  , 0.10  , 0.02  , 0.15 , 0.13 ],
            [0.01  , 0.05 ,  0.15 ,  0.14 ,  0.17 ,  0.10  , 0.12 ,  0.10 , 0.16  ],
            [0.17  , 0.15  , 0.07  , 0.07   ,0.15 ,  0.10 ,  0.12 ,  0.09, 0.08],
            [0.13,   0.11 ,  0.13 ,  0.03 ,  0.20 ,  0.20,   0.04 ,  0.15,  0.01]
    ])
n=1000
pn=np.linalg.matrix_power(p,n)
print(pn)

# Resolviendo por matrices A= AT-I) y el vector de ceros terminado en 1
k=len(p)
A=p.transpose()
A=A-np.identity(k, dtype=int)
# la última fila se sustituye por la suma de probabilidadxes
A[-1,:]=np.ones(k,dtype=int)
B=np.zeros(k,dtype=int)
B[-1]=1  # el último
calculopost=np.linalg.solve(A,B)
print('a plazo mayor')
print(calculopost)

