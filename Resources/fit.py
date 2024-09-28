from .functions import Functions
from .op import Opreration_system
import random
class Fit:
    def __init__(self) -> None:
        self.obj_f = Functions()
        self.obj_op = Opreration_system()
        pass
        # Función de entrenamiento de la red neuronal
    def train(self,patterns, wh, th, w0, tk, alpha, Ep, ET, num_neurons_hidden):
        """
        Entrena la red neuronal con los patrones de entrada/salida.
        """
        aux = """"""
        txt = """"""
        epoch = 0
        data_json = {}
        error_total = float('inf')

        while error_total > ET and epoch < Ep:
            error_total = 0

            for x, yd in patterns:
                yh = self.obj_f.forward_hidden_layer(x, wh, th, num_neurons_hidden)
                yok1 = self.obj_f.forward_output_layer(yh, w0, tk, num_neurons_hidden)
                Dok1, Dh = self.obj_f.calculate_deltas(yd, yok1, yh, w0)
                w0, wh, th, tk = self.obj_f.update_weights(Dok1, Dh, yh, w0, wh, th, tk, x, alpha, num_neurons_hidden)
                error_total += self.obj_f.calculate_error(Dok1)

            epoch += 1

            print(f"Epoch {epoch}: Error total = {error_total}")
            print(f"Pesos capa oculta: {wh}")
            print(f"Umbrales capa oculta: {th}")
            print(f"Pesos capa de salida: {w0}")
            print(f"Umbral capa de salida: {tk}")
            print("-" * 50)
            aux += f"""
            Época {epoch}
            Error total {error_total}
            Pesos de capa oculta {wh}
            umbrales capa oculta {th}
            Pesos capa de salida {w0}
            Umbral capa de salida {tk}
            {"" * 50}
"""
            if f"{epoch}" not in data_json:
                data_json[epoch] = error_total
        txt += f"""
            Entrenamiendo back propagation:
            Duró un total de {epoch} epocas
            Error total {error_total}
            Pesos de capa oculta {wh}
            umbrales capa oculta {th}
            Pesos capa de salida {w0}
            Umbral capa de salida {tk}
            
            Historial:
            {aux}
"""
        self.obj_op.read_historial(data=txt)

        return w0, wh, th, tk, data_json

    # Función para predecir la salida (después del entrenamiento)
    def predict(self,patterns, wh, th, w0, tk, num_neurons_hidden):
        """
        Realiza predicciones utilizando la red neuronal entrenada.
        """
        predictions = []
        for x, yd in patterns:
            yh = self.obj_f.forward_hidden_layer(x, wh, th, num_neurons_hidden)
            yo = self.obj_f.forward_output_layer(yh, w0, tk, num_neurons_hidden)
            predictions.append((yd, yo))
        return predictions



    # Función para inicializar los pesos y umbrales de manera aleatoria entre -1 y 1
    def initialize_weights(self,num_inputs, num_hidden, num_outputs):
        """
        Inicializa los pesos y umbrales de la red neuronal de forma aleatoria.
        """
        wh = [random.uniform(-1, 1) for _ in range(num_inputs * num_hidden)]
        th = [random.uniform(-1, 1) for _ in range(num_hidden)]
        w0 = [random.uniform(-1, 1) for _ in range(num_hidden)]
        tk = random.uniform(-1, 1)
        return wh, th, w0, tk
      # Función para calcular el error cuadrático medio (MSE)
    def calculate_mse(self,predictions):
        """
        Calcula el error cuadrático medio (MSE) de las predicciones.
        """
        error = 0
        for yd, yo in predictions:
            error += (yd - yo) ** 2
        mse = error / len(predictions)
        return mse