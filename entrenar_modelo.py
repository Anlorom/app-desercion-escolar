import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle

np.random.seed(42)
n = 1500

edad             = np.random.randint(11, 19, n)
grado            = np.random.randint(6, 12, n)
promedio         = np.random.normal(3.2, 0.8, n).clip(1.0, 5.0).round(1)
inasistencias    = np.random.poisson(12, n)
repitencia       = np.random.choice([0,1,2,3], n, p=[0.65,0.25,0.08,0.02])
nivel_se         = np.random.choice([1,2,3,4,5,6], n, p=[0.30,0.30,0.20,0.10,0.07,0.03])
distancia_km     = np.random.exponential(3, n).clip(0.5, 30).round(1)
trabaja          = np.random.choice([0,1], n, p=[0.75,0.25])
acceso_internet  = np.where(nivel_se >= 3,
                    np.random.choice([0,1], n, p=[0.2,0.8]),
                    np.random.choice([0,1], n, p=[0.6,0.4]))
apoyo_familiar   = np.random.choice([1,2,3,4,5], n, p=[0.10,0.15,0.30,0.30,0.15])

riesgo = (
    - promedio        * 1.2
    + inasistencias   * 0.08
    + repitencia      * 0.9
    - nivel_se        * 0.3
    + distancia_km    * 0.05
    + trabaja         * 0.8
    - acceso_internet * 0.4
    - apoyo_familiar  * 0.5
    + np.random.normal(0, 0.8, n)
)
desertor = (riesgo > riesgo.mean()).astype(int)

df = pd.DataFrame({
    'edad': edad, 'grado': grado, 'promedio': promedio,
    'inasistencias': inasistencias, 'repitencia': repitencia,
    'nivel_socioeconomico': nivel_se, 'distancia_km': distancia_km,
    'trabaja': trabaja, 'acceso_internet': acceso_internet,
    'apoyo_familiar': apoyo_familiar, 'desertor': desertor
})

features = ['edad','grado','promedio','inasistencias','repitencia',
            'nivel_socioeconomico','distancia_km','trabaja',
            'acceso_internet','apoyo_familiar']

X = df[features]
y = df['desertor']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

modelo = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42)
modelo.fit(X_train, y_train)

with open('modelo.pkl', 'wb') as f:
    pickle.dump(modelo, f)

print('✅ Modelo guardado como modelo.pkl')
print(f'   Accuracy: {modelo.score(X_test, y_test):.2%}')