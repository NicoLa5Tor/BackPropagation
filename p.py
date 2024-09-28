import math

# Clase para el entrenamiento
class Fit:
    # Función sigmoide
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    # Derivada de la función sigmoide
    def sigmoid_derivative(self, x):
        return x * (1 - x)

    # Paso 3: Propagación hacia adelante en las neuronas ocultas
    def forward_hidden_layer(self, x, wh, th):
        # Neurona j=1
        nethj1 = wh[0] * x[0] + wh[2] * x[1] + th[0]
        yhj1 = self.sigmoid(nethj1)
        
        # Neurona j=2
        nethj2 = wh[1] * x[0] + wh[3] * x[1] + th[1]
        yhj2 = self.sigmoid(nethj2)
        
        return yhj1, yhj2

    # Paso 4: Propagación hacia adelante en las neuronas de salida
    def forward_output_layer(self, yhj1, yhj2, w0, tk):
        # Neurona k=1
        netok1 = w0[0] * yhj1 + w0[1] * yhj2 + tk
        yok1 = self.sigmoid(netok1)
        
        return yok1

    # Paso 5: Cálculo de errores delta (salida y neuronas ocultas)
    def calculate_deltas(self, yd, yok1, yhj1, yhj2, w0):
        # Cálculo de errores parciales neurona k=1
        Dok1 = (yd - yok1) * yok1 * (1 - yok1)
        
        # Cálculo de errores parciales neurona j=1
        Dhj1x1 = yhj1 * (1 - yhj1) * Dok1 * w0[0]
        Dhj2x1 = yhj2 * (1 - yhj2) * Dok1 * w0[1]
        
        return Dok1, Dhj1x1, Dhj2x1

    # Paso 6: Actualización de los pesos con momentum
    def update_weights(self, Dok1, Dhj1x1, Dhj2x1, yhj1, yhj2, w0, wh, th, tk, x, alpha, momentum, prev_updates):
        # Actualización de los pesos de salida con momentum
        delta_w0_0 = alpha * Dok1 * yhj1 + momentum * prev_updates['w0_0']
        delta_w0_1 = alpha * Dok1 * yhj2 + momentum * prev_updates['w0_1']
        delta_tk = alpha * Dok1 + momentum * prev_updates['tk']

        w0[0] += delta_w0_0
        w0[1] += delta_w0_1
        tk += delta_tk

        # Actualización de los pesos de entrada con momentum
        delta_wh_0 = alpha * Dhj1x1 * x[0] + momentum * prev_updates['wh_0']
        delta_wh_2 = alpha * Dhj1x1 * x[1] + momentum * prev_updates['wh_2']
        delta_wh_1 = alpha * Dhj2x1 * x[0] + momentum * prev_updates['wh_1']
        delta_wh_3 = alpha * Dhj2x1 * x[1] + momentum * prev_updates['wh_3']

        wh[0] += delta_wh_0
        wh[2] += delta_wh_2
        wh[1] += delta_wh_1
        wh[3] += delta_wh_3

        # Actualización de los umbrales con momentum
        delta_th_0 = alpha * Dhj1x1 + momentum * prev_updates['th_0']
        delta_th_1 = alpha * Dhj2x1 + momentum * prev_updates['th_1']

        th[0] += delta_th_0
        th[1] += delta_th_1

        # Actualización de las diferencias para momentum
        prev_updates['w0_0'], prev_updates['w0_1'], prev_updates['tk'] = delta_w0_0, delta_w0_1, delta_tk
        prev_updates['wh_0'], prev_updates['wh_1'], prev_updates['wh_2'], prev_updates['wh_3'] = delta_wh_0, delta_wh_1, delta_wh_2, delta_wh_3
        prev_updates['th_0'], prev_updates['th_1'] = delta_th_0, delta_th_1

        return w0, wh, th, tk, prev_updates

    # Paso 7: Cálculo del error
    def calculate_error(self, Dok1):
        return 0.5 * (Dok1 ** 2)

    # Función para entrenar con un solo patrón
    def train_single_pattern(self, x, yd, wh, th, w0, tk, alpha, momentum, prev_updates):
        # Propagación hacia adelante
        yhj1, yhj2 = self.forward_hidden_layer(x, wh, th)
        yok1 = self.forward_output_layer(yhj1, yhj2, w0, tk)

        # Cálculo de los deltas
        Dok1, Dhj1x1, Dhj2x1 = self.calculate_deltas(yd, yok1, yhj1, yhj2, w0)

        # Actualización de los pesos y umbrales
        w0, wh, th, tk, prev_updates = self.update_weights(Dok1, Dhj1x1, Dhj2x1, yhj1, yhj2, w0, wh, th, tk, x, alpha, momentum, prev_updates)

        # Cálculo del error para este patrón
        error = self.calculate_error(Dok1)

        return w0, wh, th, tk, yok1, error

# Clase de patrón (dummy)
class Patrons:
    def pt_fit(self):
        # Simulando un conjunto de patrones con entradas y salidas
        return {
            "1": {"entradas": [1.5, 1.2], "salida": 0.58778666},
            "2": {"entradas": [2.0, 1.8], "salida": 0.7985077},
            "3": {"entradas": [1.0, 2.0], "salida": 0.6931472},
            # Añade más patrones aquí si lo necesitas
        }

# Función principal para entrenar con todos los patrones
def fit():
    # Creación de objetos
    obj_fit = Fit()
    obj_p = Patrons()
    
    # Obtener todos los patrones del objeto Patrons
    json_data = obj_p.pt_fit()
    
    # Pesos iniciales (comunes para todos los patrones)
    wh = [0.3, 0.7, 0.5, 0.9]  # Pesos sinápticos de la capa oculta
    th = [-0.6, -0.8]           # Umbrales de la capa oculta
    w0 = [0.1, 0.4]             # Pesos sinápticos de la capa de salida
    tk = -0.1                   # Umbral de la capa de salida

    # Parámetros de entrenamiento
    alpha = 0.5    # Tasa de aprendizaje
    momentum = 0.9 # Coeficiente de momentum
    Ep = 1000      # Número máximo de épocas
    ET = 0.001     # Error total mínimo aceptable

    # Diccionario para almacenar las actualizaciones previas (para momentum)
    prev_updates = {
        'w0_0': 0, 'w0_1': 0, 'tk': 0,
        'wh_0': 0, 'wh_1': 0, 'wh_2': 0, 'wh_3': 0,
        'th_0': 0, 'th_1': 0
    }

    # Entrenamiento por épocas
    for epoch in range(Ep):
        total_error = 0

        # Recorrer todos los patrones en el JSON
        for option, data in json_data.items():
            x = data['entradas']  # Entradas del patrón actual
            yd = data['salida']   # Salida deseada del patrón actual

            # Entrenar la red neuronal con momentum para el patrón actual
            w0, wh, th, tk, output, error = obj_fit.train_single_pattern(x, yd, wh, th, w0, tk, alpha, momentum, prev_updates)
            
            # Acumular el error total
            total_error += error
        
        # Calcular el error promedio
        avg_error = total_error / len(json_data)
        
        print(f"Época {epoch+1}: Error promedio = {avg_error}")
        
        # Condición de parada: Si el error es menor que el umbral ET, detener el entrenamiento
fit()