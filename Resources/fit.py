from .functions import Functions
from .op import Opreration_system
import random

class Fit:
    def __init__(self) -> None:
        self.obj_f = Functions()
        self.obj_op = Opreration_system()
        pass

    def train(self, patterns, wh, th, w0, tk, alpha, momentum, Ep, ET, num_neurons_hidden, callback=None):
        """
        Entrena la red neuronal con los patrones de entrada/salida usando momentum.
        """
        epoch = 0
        data_json = {}
        error_total = float('inf')

        # Inicializar deltas anteriores para el momentum
        delta_w0_prev = [0] * len(w0)
        delta_wh_prev = [0] * len(wh)
        delta_th_prev = [0] * len(th)
        delta_tk_prev = 0

        while error_total > ET and epoch < Ep:
            error_total = 0

            for x, yd in patterns:
                yh = self.obj_f.forward_hidden_layer(x, wh, th, num_neurons_hidden)
                yok1 = self.obj_f.forward_output_layer(yh, w0, tk, num_neurons_hidden)
                Dok1, Dh = self.obj_f.calculate_deltas(yd, yok1, yh, w0)
                
                # Actualizar pesos con momentum
                w0, wh, th, tk, delta_w0_prev, delta_wh_prev, delta_th_prev, delta_tk_prev = \
                    self.obj_f.update_weights_momentum(
                        Dok1, Dh, yh, w0, wh, th, tk, x, alpha, momentum, num_neurons_hidden,
                        delta_w0_prev, delta_wh_prev, delta_th_prev, delta_tk_prev
                    )
                
                error_total += self.obj_f.calculate_error(Dok1)

            epoch += 1
            data_json[epoch] = error_total
            print(f"Epoch {epoch}: Error total = {error_total}")

            # Llamar al callback con los datos de la época actual
            if callback:
                callback(epoch, error_total)

        return w0, wh, th, tk, data_json


    def predict(self, patterns, wh, th, w0, tk, num_neurons_hidden):
        predictions = []
        for x, yd in patterns:
            yh = self.obj_f.forward_hidden_layer(x, wh, th, num_neurons_hidden)
            yo = self.obj_f.forward_output_layer(yh, w0, tk, num_neurons_hidden)
            predictions.append((yd, yo))
        return predictions

    def initialize_weights(self, num_inputs, num_hidden, num_outputs):
        wh = [random.uniform(-1, 1) for _ in range(num_inputs * num_hidden)]
        th = [random.uniform(-1, 1) for _ in range(num_hidden)]
        w0 = [random.uniform(-1, 1) for _ in range(num_hidden)]
        tk = random.uniform(-1, 1)
        return wh, th, w0, tk

    def calculate_mse(self, predictions):
        error = 0
        for yd, yo in predictions:
            error += (yd - yo) ** 2
        mse = error / len(predictions)
        return mse
