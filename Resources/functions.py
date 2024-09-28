import math
class Functions:

    def __init__(self) -> None:
        pass
    # Función tangente hiperbólica
    def tanh(self,x):
        """
        Calcula la función tangente hiperbólica.
        """
        return math.tanh(x)

    # Derivada de la función tangente hiperbólica
    def tanh_derivative(self,x):
        """
        Calcula la derivada de la función tangente hiperbólica.
        """
        return 1 - math.tanh(x) ** 2

    # Paso 3: Cálculos en las neuronas ocultas (propagación hacia adelante en la capa oculta)
    def forward_hidden_layer(self,x, wh, th, num_neurons_hidden):
        """
        Realiza la propagación hacia adelante en la capa oculta.
        """
        yh = []
        for j in range(num_neurons_hidden):
            net_h = sum(wh[i + j * len(x)] * x[i] for i in range(len(x))) + th[j]
            yh.append(self.tanh(net_h))
        return yh

    # Paso 4: Cálculos en la neurona de salida (propagación hacia adelante en la capa de salida)
    def forward_output_layer(self,yh, w0, tk, num_neurons_hidden):
        """
        Realiza la propagación hacia adelante en la neurona de salida.
        """
        net_output = sum(w0[i] * yh[i] for i in range(num_neurons_hidden)) + tk
        return self.tanh(net_output)

    # Paso 5: Cálculo de los errores delta (retropropagación del error)
    def calculate_deltas(self,yd, yok1, yh, w0):
        """
        Calcula los deltas para ajustar los pesos.
        """
        Dok1 = (yd - yok1) * self.tanh_derivative(yok1)
        Dh = []
        for i in range(len(yh)):
            Dh.append(self.tanh_derivative(yh[i]) * Dok1 * w0[i])
        return Dok1, Dh

    # Paso 6: Actualización de los pesos (ajuste de los pesos y umbrales)
    def update_weights(self,Dok1, Dh, yh, w0, wh, th, tk, x, alpha, num_neurons_hidden):
        """
        Actualiza los pesos y umbrales de la red neuronal.
        """
        for i in range(num_neurons_hidden):
            w0[i] += alpha * Dok1 * yh[i]
        tk += alpha * Dok1

        for j in range(num_neurons_hidden):
            for i in range(len(x)):
                wh[i + j * len(x)] += alpha * Dh[j] * x[i]

        for j in range(num_neurons_hidden):
            th[j] += alpha * Dh[j]

        return w0, wh, th, tk

    # Paso 7: Cálculo del error (error cuadrático)
    def calculate_error(self,Dok1):
        """
        Calcula el error cuadrático para la neurona de salida.
        """
        return 0.5 * (Dok1 ** 2)
  