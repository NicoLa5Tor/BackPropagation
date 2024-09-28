import math
class Functions:

    def __init__(self) -> None:
        pass
        # Función sigmoide
    def sigmoid(self,x):
        return 1 / (1 + math.exp(-x))

    # Derivada de la función sigmoide
    def sigmoid_derivative(self,x):
        return x * (1 - x)

    # Paso 3: Cálculos en las neuronas ocultas
    def forward_hidden_layer(self,x, wh, th):
        # Neurona j=1
        nethj1 = wh[0] * x[0] + wh[2] * x[1] + th[0]
        yhj1 = self.sigmoid(nethj1)
        
        # Neurona j=2
        nethj2 = wh[1] * x[0] + wh[3] * x[1] + th[1]
        yhj2 = self.sigmoid(nethj2)
        
        return yhj1, yhj2

    # Paso 4: Cálculos en las neuronas de salida
    def forward_output_layer(self,yhj1, yhj2, w0, tk):
        # Neurona k=1
        netok1 = w0[0] * yhj1 + w0[1] * yhj2 + tk
        yok1 = self.sigmoid(netok1)
        
        return yok1

    # Paso 5: Cálculo de errores delta (salida y neuronas ocultas)
    def calculate_deltas(self,yd, yok1, yhj1, yhj2, w0):
        # Cálculo de errores parciales neurona k=1
        Dok1 = (yd - yok1) * yok1 * (1 - yok1)
        
        # Cálculo de errores parciales neurona j=1
        Dhj1x1 = yhj1 * (1 - yhj1) * Dok1 * w0[0]
        Dhj2x1 = yhj2 * (1 - yhj2) * Dok1 * w0[1]
        
        return Dok1, Dhj1x1, Dhj2x1

    # Paso 6: Actualización de los pesos sin momentum
    def update_weights(self,Dok1, Dhj1x1, Dhj2x1, yhj1, yhj2, w0, wh, th, tk, x, alpha):
        # Actualización de los pesos de salida
        w0[0] += alpha * Dok1 * yhj1
        w0[1] += alpha * Dok1 * yhj2
        tk += alpha * Dok1

        # Actualización de los pesos de entrada
        wh[0] += alpha * Dhj1x1 * x[0]
        wh[2] += alpha * Dhj1x1 * x[1]
        wh[1] += alpha * Dhj2x1 * x[0]
        wh[3] += alpha * Dhj2x1 * x[1]

        # Actualización de los umbrales
        th[0] += alpha * (Dhj1x1)
        th[1] += alpha * (Dhj2x1)

        return w0, wh, th, tk

    # Paso 7: Cálculo del error
    def calculate_error(self,Dok1):
        E = 0.5 * (Dok1 ** 2)
        return E