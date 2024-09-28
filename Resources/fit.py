from .functions import Functions
class Fit:
    def __init__(self) -> None:
        self.obj_f = Functions()
        pass
    # Función de entrenamiento de la red neuronal sin momentum e impresión de pesos
    def train(self,x, yd, wh, th, w0, tk, alpha, Ep, ET):
        epoch = 0
        error_total = float('inf')

        while error_total > ET and epoch < Ep:
            # Paso 3: Propagación hacia adelante (capa oculta)
            yhj1, yhj2 = self.obj_f.forward_hidden_layer(x, wh, th)

            # Paso 4: Propagación hacia adelante (capa de salida)
            yok1 = self.obj_f.forward_output_layer(yhj1, yhj2, w0, tk)

            # Paso 5: Cálculo de los deltas
            Dok1, Dhj1x1, Dhj2x1 = self.obj_f.calculate_deltas(yd, yok1, yhj1, yhj2, w0)

            # Paso 6: Actualización de los pesos sin momentum
            w0, wh, th, tk = self.obj_f.update_weights(Dok1, Dhj1x1, Dhj2x1, yhj1, yhj2, w0, wh, th, tk, x, alpha)

            # Paso 7: Cálculo del error
            error_total = self.obj_f.calculate_error(Dok1)
            epoch += 1

            # Imprimir los pesos, umbrales y el error total en cada época
            print(f"Epoch {epoch}: Error total = {error_total}")
            print(f"Pesos capa oculta: {wh}")
            print(f"Umbrales capa oculta: {th}")
            print(f"Pesos capa de salida: {w0}")
            print(f"Umbral capa de salida: {tk}")
            print(f"La salida obtenida es: {yok1}")
            print("-" * 50)

        return w0, wh, th, tk, yok1