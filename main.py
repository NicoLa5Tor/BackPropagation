from Resources.fit import Fit
from Resources.patrons import Patrons
import time

def fit(option = 0):  
    #creacion de objetos
    obj_fit = Fit()
    obj_p = Patrons()
    # Definiciones iniciales
    json_data = obj_p.pt_fit()
    print(json_data)
    if option > 0:
        x = json_data[f'{option}']['entradas'] # Entradas
        yd = json_data[f'{option}']['salida']  # Salida deseada (YD = ln(1.5) + ln(1.2))
    print(x,yd)
    # Pesos iniciales
    wh = [0.3, 0.7, 0.5, 0.9]  # Pesos sinápticos de la capa oculta
    th = [-0.6, -0.8]           # Umbrales de la capa oculta
    w0 = [0.1, 0.4]             # Pesos sinápticos de la capa de salida
    tk = -0.1                   # Umbral de la capa de salida

    # Parámetros de entrenamiento
    alpha = 0.05    # Tasa de aprendizaje
    Ep = 100000      # Número máximo de épocas
    ET = 0.001     # Error total mínimo aceptable

    # Entrenamiento de la red neuronal sin momentum e impresión de pesos
    w0, wh, th, tk, output = obj_fit.train(x, yd, wh, th, w0, tk, alpha, Ep, ET)

    # Resultado final
    print(f"Salida final después del entrenamiento: {output}")
fit(option=1)