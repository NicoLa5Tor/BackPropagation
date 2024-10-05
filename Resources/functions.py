import math

class Functions:
    def __init__(self) -> None:
        pass

    def tanh(self, x):
        return math.tanh(x)

    def tanh_derivative(self, x):
        return 1 - math.tanh(x) ** 2

    def forward_hidden_layer(self, x, wh, th, num_neurons_hidden):
        yh = []
        for j in range(num_neurons_hidden):
            net_h = sum(wh[i + j * len(x)] * x[i] for i in range(len(x))) + th[j]
            yh.append(self.tanh(net_h))
        return yh

    def forward_output_layer(self, yh, w0, tk, num_neurons_hidden):
        net_output = sum(w0[i] * yh[i] for i in range(num_neurons_hidden)) + tk
        return self.tanh(net_output)

    def calculate_deltas(self, yd, yok1, yh, w0):
        Dok1 = (yd - yok1) * self.tanh_derivative(yok1)
        Dh = [self.tanh_derivative(yh[i]) * Dok1 * w0[i] for i in range(len(yh))]
        return Dok1, Dh

    def update_weights_momentum(self, Dok1, Dh, yh, w0, wh, th, tk, x, alpha, momentum, num_neurons_hidden,
                                delta_w0_prev, delta_wh_prev, delta_th_prev, delta_tk_prev):
        # Actualización de w0 (pesos de salida) con momentum
        delta_w0 = [alpha * Dok1 * yh[i] + momentum * delta_w0_prev[i] for i in range(num_neurons_hidden)]
        w0 = [w0[i] + delta_w0[i] for i in range(num_neurons_hidden)]

        # Actualización de umbral de salida tk
        delta_tk = alpha * Dok1 + momentum * delta_tk_prev
        tk += delta_tk

        # Actualización de wh (pesos de la capa oculta) con momentum
        delta_wh = [0] * len(wh)
        for j in range(num_neurons_hidden):
            for i in range(len(x)):
                delta_wh[i + j * len(x)] = alpha * Dh[j] * x[i] + momentum * delta_wh_prev[i + j * len(x)]
                wh[i + j * len(x)] += delta_wh[i + j * len(x)]

        # Actualización de umbrales de la capa oculta th
        delta_th = [alpha * Dh[j] + momentum * delta_th_prev[j] for j in range(num_neurons_hidden)]
        th = [th[j] + delta_th[j] for j in range(num_neurons_hidden)]

        return w0, wh, th, tk, delta_w0, delta_wh, delta_th, delta_tk

    def calculate_error(self, Dok1):
        return 0.5 * (Dok1 ** 2)
