import os
import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import random
import math

# Variables globales para almacenar los resultados del entrenamiento
global_data_json = None
global_results = None
global_mse = None
global_wh = None  # Pesos capa oculta
global_w0 = None  # Pesos capa de salida

def centrar_ventana(ventana, ancho_ventana, alto_ventana):
    ventana.update_idletasks()
    ancho_pantalla = ventana.winfo_screenwidth()
    alto_pantalla = ventana.winfo_screenheight()

    x = int((ancho_pantalla / 2) - (ancho_ventana / 2))
    y = int((alto_pantalla / 2) - (alto_ventana / 2))

    ventana.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

class ScrollableFrame(ctk.CTkFrame):
    def __init__(self, parent, *args, **kwargs):
        super().__init__(parent, *args, **kwargs)

        # Crear un Canvas estándar de Tkinter dentro del Frame de CustomTkinter
        self.canvas = tk.Canvas(self, bg='#2E3440', highlightthickness=0)
        self.scrollbar = ctk.CTkScrollbar(self, orientation="vertical", command=self.canvas.yview)
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        # Empacar el Canvas y la Scrollbar
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)

        # Crear un Frame interno dentro del Canvas
        self.scrollable_frame = ctk.CTkFrame(self.canvas, fg_color='#2E3440')
        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(
                scrollregion=self.canvas.bbox("all")
            )
        )

        # Crear una ventana en el Canvas para el Frame interno
        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")

class Fit:
    def __init__(self):
        pass

    def initialize_weights(self, num_inputs, num_neurons_hidden, num_outputs):
        # Inicialización Xavier para tanh
        limit_hidden = math.sqrt(6 / (num_inputs + num_neurons_hidden))
        limit_output = math.sqrt(6 / (num_neurons_hidden + num_outputs))
        # Pesos entre entrada y capa oculta
        wh = [[random.uniform(-limit_hidden, limit_hidden) for _ in range(num_inputs)] for _ in range(num_neurons_hidden)]
        # Umbrales de la capa oculta
        th = [random.uniform(-limit_hidden, limit_hidden) for _ in range(num_neurons_hidden)]
        # Pesos entre capa oculta y salida
        w0 = [random.uniform(-limit_output, limit_output) for _ in range(num_neurons_hidden)]
        # Umbral de la capa de salida
        tk = random.uniform(-limit_output, limit_output)
        return wh, th, w0, tk

    def train(self, patterns, wh, th, w0, tk, alpha, Ep, ET, num_neurons_hidden, momentum):
        # Inicializar cambios anteriores de pesos para momentum
        delta_wh_prev = [[0.0 for _ in range(len(wh[0]))] for _ in range(len(wh))]
        delta_w0_prev = [0.0 for _ in range(len(w0))]
        delta_th_prev = [0.0 for _ in range(len(th))]
        delta_tk_prev = 0.0

        data_json = {}

        for epoch in range(int(Ep)):
            error_total = 0.0

            for inputs, yd in patterns:
                # Forward pass

                # Calcular la entrada neta a la capa oculta
                net_h = []
                for i in range(num_neurons_hidden):
                    net = sum([wh[i][j] * inputs[j] for j in range(len(inputs))]) - th[i]
                    net_h.append(net)

                # Calcular la salida de la capa oculta usando tanh
                out_h = [self.tanh(net) for net in net_h]

                # Calcular la entrada neta al nodo de salida
                net_o = sum([w0[i] * out_h[i] for i in range(num_neurons_hidden)]) - tk

                # Calcular la salida de la red usando tanh
                yo = self.tanh(net_o)

                # Calcular el error en la salida
                error_output = yd - yo

                # Calcular delta de salida usando derivada de tanh
                delta_output = error_output * self.dtanh(yo)

                # Acumular error total
                error_total += error_output ** 2

                # Backward pass

                # Calcular delta para la capa oculta
                delta_hidden = []
                for i in range(num_neurons_hidden):
                    delta = self.dtanh(out_h[i]) * w0[i] * delta_output
                    delta_hidden.append(delta)

                # Actualizar pesos w0 y umbral tk con momentum
                for i in range(num_neurons_hidden):
                    delta_w0 = alpha * delta_output * out_h[i] + momentum * delta_w0_prev[i]
                    w0[i] += delta_w0
                    delta_w0_prev[i] = delta_w0

                delta_tk = -alpha * delta_output + momentum * delta_tk_prev
                tk += delta_tk
                delta_tk_prev = delta_tk

                # Actualizar pesos wh y umbrales th con momentum
                for i in range(num_neurons_hidden):
                    for j in range(len(inputs)):
                        delta_wh = alpha * delta_hidden[i] * inputs[j] + momentum * delta_wh_prev[i][j]
                        wh[i][j] += delta_wh
                        delta_wh_prev[i][j] = delta_wh

                    delta_th = -alpha * delta_hidden[i] + momentum * delta_th_prev[i]
                    th[i] += delta_th
                    delta_th_prev[i] = delta_th

            # Almacenar error total por época
            data_json[epoch] = error_total

            # Mostrar información de la época en la consola
            print(f"Época {epoch + 1}/{int(Ep)} - Error Total: {error_total:.6f}")

            # Verificar si se alcanzó el error deseado
            if error_total < ET:
                print(f"Error deseado alcanzado en la época {epoch+1}")
                break

        return w0, wh, th, tk, data_json

    def predict(self, patterns, wh, th, w0, tk, num_neurons_hidden):
        predictions = []
        for inputs, yd in patterns:
            # Forward pass

            # Calcular la entrada neta a la capa oculta
            net_h = []
            for i in range(num_neurons_hidden):
                net = sum([wh[i][j] * inputs[j] for j in range(len(inputs))]) - th[i]
                net_h.append(net)

            # Calcular la salida de la capa oculta usando tanh
            out_h = [self.tanh(net) for net in net_h]

            # Calcular la entrada neta al nodo de salida
            net_o = sum([w0[i] * out_h[i] for i in range(num_neurons_hidden)]) - tk

            # Calcular la salida de la red usando tanh
            yo = self.tanh(net_o)

            predictions.append((yd, yo))

        return predictions

    def calculate_mse(self, predictions):
        mse = sum([(yd - yo) ** 2 for yd, yo in predictions]) / len(predictions)
        return mse

    # Función de activación tanh
    def tanh(self, x):
        return math.tanh(x)

    # Derivada de la función tanh
    def dtanh(self, y):
        # y es tanh(x), entonces dtanh/dx = 1 - tanh^2(x) = 1 - y^2
        return 1 - y ** 2

def fit(neuronas=6, alp=0.01, epocas=30000, error=0.001, momentum=0.9):
    obj_fit = Fit()

    # Datos de entrenamiento
    patterns = [
        ((0.144052004, 0.12075527), -1),
        ((0.864953975, 0.25656044), 0),
        ((0.165339509, 0.48834463), 1),
        ((0.797560171, 0.02913177), 0),
        ((0.593582089, 0.13476484), -1),
        ((0.704011446, 0.06116278), 0),
        ((0.905842195, 0.35656061), -1),
        ((0.543030439, 0.39351471), 1),
        ((0.973983333, 0.46497479), -1),
        ((0.308200305, 0.82542819), 1)
    ]

    # Normalizar entradas al rango [-1, 1]
    normalized_patterns = []
    for (inputs, yd) in patterns:
        normalized_inputs = tuple([(x - 0.5) * 2 for x in inputs])
        normalized_patterns.append((normalized_inputs, yd))

    num_neurons_hidden = int(neuronas)
    num_inputs = 2
    num_outputs = 1

    wh, th, w0, tk = obj_fit.initialize_weights(num_inputs, num_neurons_hidden, num_outputs)

    alpha = alp
    Ep = epocas
    ET = error

    # Incluir momentum en el método de entrenamiento
    w0, wh, th, tk, data_json = obj_fit.train(
        normalized_patterns, wh, th, w0, tk, alpha, Ep, ET, num_neurons_hidden, momentum
    )
    predictions = obj_fit.predict(normalized_patterns, wh, th, w0, tk, num_neurons_hidden)

    results = []
    for i, ((inputs, yd), (yd_pred, yo)) in enumerate(zip(normalized_patterns, predictions)):
        result = f"Patrón {i+1}: yd = {yd:.6f}, yo = {yo:.6f}"
        print(result)
        # Incluir patrones de entrada originales en los resultados
        original_inputs = patterns[i][0]
        results.append({
            'pattern_name': f"Patrón {i+1}",
            'inputs': original_inputs,
            'yd': f"{yd:.6f}",
            'yo': f"{yo:.6f}"
        })

    mse = obj_fit.calculate_mse(predictions)
    mse_text = f"Error Cuadrático Medio (MSE): {mse:.6f}"
    print(mse_text)

    # Guardar pesos globalmente para uso en la visualización
    global global_wh, global_w0
    global_wh = wh
    global_w0 = w0

    # Guardar resultados en un archivo de texto
    with open("resultados_entrenamiento.txt", "w") as f:
        f.write("Resultados del Entrenamiento:\n")
        f.write(f"Neurons Hidden: {neuronas}\n")
        f.write(f"Alpha: {alp}\n")
        f.write(f"Momentum: {momentum}\n")
        f.write(f"Épocas: {epocas}\n")
        f.write(f"Error Deseado: {error}\n\n")
        f.write("Resultados por Patrón:\n")
        for res in results:
            f.write(f"{res['pattern_name']}: X1={res['inputs'][0]:.6f}, X2={res['inputs'][1]:.6f}, YD={res['yd']}, YO={res['yo']}\n")
        f.write(f"\n{mse_text}\n")

    return data_json, results, mse

def parse_training_data(data):
    parameters = {}
    data_list = data.strip().split("\n")
    for line in data_list:
        if '=' in line:
            key, value = line.split("=")
            key = key.strip().lower()
            try:
                parameters[key] = float(value.strip())
            except ValueError:
                messagebox.showerror("Error", f"Valor inválido para {key}")
                return None
        else:
            messagebox.showerror("Error", f"La línea '{line}' no tiene el formato correcto.")
            return None
    required_keys = ['alpha', 'momentum', 'épocas', 'error deseado', 'neuronas ocultas']
    if not all(k in parameters for k in required_keys):
        messagebox.showerror("Error", "Faltan parámetros en los datos de entrenamiento.")
        return None
    return parameters

def graficar():
    archivo = "datos_entrenamiento.txt"
    datos_predefinidos = """Alpha = 0.01
Momentum = 0.9
Épocas = 30000
Error deseado = 0.001
Neuronas ocultas = 6
"""

    def abrir_ventana_edicion():
        if not os.path.exists(archivo):
            with open(archivo, 'w') as file:
                file.write(datos_predefinidos)

        ventana_edicion = ctk.CTkToplevel(root)
        ventana_edicion.title("Editar Archivo de Entrenamiento")
        ventana_edicion.geometry("600x500")
        centrar_ventana(ventana_edicion, 600, 500)

        # Hacer que la ventana esté siempre encima
        ventana_edicion.attributes('-topmost', True)

        text_box = ctk.CTkTextbox(ventana_edicion, wrap="word", font=("Arial", 12))
        text_box.pack(fill="both", expand=True, padx=10, pady=10)

        with open(archivo, 'r') as file:
            contenido = file.read()
        text_box.insert(tk.END, contenido)

        def guardar_cambios():
            with open(archivo, 'w') as file:
                contenido = text_box.get("1.0", tk.END)
                file.write(contenido)
            messagebox.showinfo("Guardado", "¡Los cambios se han guardado correctamente!", parent=ventana_edicion)
            ventana_edicion.destroy()

        guardar_button = ctk.CTkButton(
            ventana_edicion, text="GUARDAR CAMBIOS", command=guardar_cambios,
            fg_color="#81A1C1", hover_color="#5E81AC", font=("Arial", 16, "bold")
        )
        guardar_button.pack(pady=20)

    def graficar_errores(epocas, errores, frame):
        if not epocas or not errores:
            print("No hay datos para graficar errores.")
            return

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epocas, errores, marker='o', color='#88C0D0')
        ax.set_title("Épocas vs. Error Total")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Error Total")
        ax.grid(True, color='#4C566A')
        ax.set_facecolor('#2E3440')
        fig.patch.set_facecolor('#2E3440')
        ax.tick_params(colors='#D8DEE9')
        ax.xaxis.label.set_color('#D8DEE9')
        ax.yaxis.label.set_color('#D8DEE9')
        ax.title.set_color('#ECEFF4')
        fig.tight_layout()

        for widget in frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def mostrar_entrenamiento():
        limpiar_interfaz()

        frame_superior = ctk.CTkFrame(root)
        frame_superior.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        texto_label = ctk.CTkLabel(frame_superior, text="BACK PROPAGATION ENTRENAMIENTO",
                                   font=title_font)
        texto_label.pack(pady=10)

        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        data_frame = ctk.CTkFrame(root)
        data_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        data_frame.grid_rowconfigure(0, weight=1)
        data_frame.grid_rowconfigure(1, weight=4)
        data_frame.grid_columnconfigure(0, weight=1)

        top_right_frame = ctk.CTkFrame(data_frame)
        top_right_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        editar_button = ctk.CTkButton(top_right_frame, text="Crear y Editar Archivo", command=abrir_ventana_edicion,
                                      fg_color="#81A1C1", hover_color="#5E81AC", font=button_font)
        editar_button.grid(row=0, column=0, padx=5, pady=5, sticky="e")

        entrenar_button = ctk.CTkButton(top_right_frame, text="Entrenar y Graficar", command=leer_txt_y_entrenar,
                                        fg_color="#88C0D0", hover_color="#81A1C1", font=button_font)
        entrenar_button.grid(row=0, column=1, padx=5, pady=5, sticky="w")

        top_right_frame.grid_columnconfigure(0, weight=1)
        top_right_frame.grid_columnconfigure(1, weight=1)

        global bottom_right_frame
        bottom_right_frame = ctk.CTkFrame(data_frame)
        bottom_right_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        titulo_label = ctk.CTkLabel(bottom_right_frame, text="Gráfico de Épocas vs. Error Total",
                                    font=subtitle_font)
        titulo_label.pack(pady=10)

    def leer_txt_y_entrenar():
        if os.path.exists(archivo):
            with open(archivo, 'r') as file:
                contenido = file.read()
            parameters = parse_training_data(contenido)
            if parameters is None:
                return

            # Usar variables globales para almacenar los resultados del entrenamiento
            global global_data_json, global_results, global_mse
            global_data_json, global_results, global_mse = fit(
                neuronas=int(parameters['neuronas ocultas']),
                alp=parameters['alpha'],
                epocas=int(parameters['épocas']),
                error=parameters['error deseado'],
                momentum=parameters['momentum']
            )
            epocas = list(global_data_json.keys())
            errores = list(global_data_json.values())

            print(f"Épocas entrenadas: {len(epocas)}")
            print(f"Errores registrados: {len(errores)}")

            graficar_errores(epocas, errores, bottom_right_frame)

            # Mostrar mensaje de finalización y abrir el archivo de resultados
            messagebox.showinfo("Entrenamiento Completo", "El entrenamiento ha finalizado correctamente.")
            try:
                os.startfile("resultados_entrenamiento.txt")
            except AttributeError:
                # Para sistemas que no soportan os.startfile, como Linux o Mac
                import subprocess
                subprocess.call(['open', "resultados_entrenamiento.txt"])
        else:
            messagebox.showwarning("Advertencia", "El archivo no existe. Cárgalo o créalo primero.")

    def mostrar_aplicacion():
        limpiar_interfaz()

        frame_superior = ctk.CTkFrame(root)
        frame_superior.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        texto_label = ctk.CTkLabel(frame_superior, text="BACK PROPAGATION APLICACIÓN",
                                   font=title_font)
        texto_label.pack(pady=10)

        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(root)
        main_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=2)  # Peso ajustado para left_frame
        main_frame.grid_columnconfigure(1, weight=30)  # Peso ajustado para right_frame

        left_frame = ctk.CTkFrame(main_frame)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # Crear un ScrollableFrame para las gráficas en el lado derecho
        right_frame = ctk.CTkFrame(main_frame)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        scrollable = ScrollableFrame(right_frame)
        scrollable.pack(fill="both", expand=True)

        # Crear los frames para las gráficas dentro del ScrollableFrame
        architecture_frame = ctk.CTkFrame(scrollable.scrollable_frame, fg_color='#2E3440')
        architecture_frame.pack(fill="both", expand=True, padx=5, pady=5)

        training_plot_frame = ctk.CTkFrame(scrollable.scrollable_frame, fg_color='#2E3440')
        training_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)

        # Usar variables globales para acceder a los resultados del entrenamiento
        if global_results is not None and global_data_json is not None:
            # Crear selector de patrón
            pattern_options = [result['pattern_name'] for result in global_results]
            selected_pattern = tk.StringVar(value=pattern_options[0])

            def update_outputs(*args):
                # Obtener índice del patrón seleccionado
                try:
                    index = pattern_options.index(selected_pattern.get())
                except ValueError:
                    print("Patrón seleccionado no válido.")
                    return
                # Obtener datos correspondientes
                pattern_data = global_results[index]

                # Actualizar las etiquetas
                label_input_x1_value.configure(text=f"{pattern_data['inputs'][0]:.6f}")
                label_input_x2_value.configure(text=f"{pattern_data['inputs'][1]:.6f}")
                label_yd_value.configure(text=pattern_data['yd'])
                label_yo_value.configure(text=pattern_data['yo'])
                label_mse_value.configure(text=f"{global_mse:.6f}")

            # Menú desplegable para seleccionar el patrón
            pattern_label = ctk.CTkLabel(left_frame, text="Seleccione un patrón:",
                                         font=label_font)
            pattern_label.pack(pady=(10, 5))

            # Ajuste del ancho del menú desplegable
            pattern_menu = ctk.CTkOptionMenu(
                left_frame, variable=selected_pattern, values=pattern_options,
                fg_color="#4C566A", button_color="#434C5E",
                button_hover_color="#4C566A", font=label_font,
                width=200  # Especificamos un ancho de 200 píxeles
            )
            pattern_menu.pack(pady=5)

            # Vincular el evento de selección
            selected_pattern.trace('w', update_outputs)

            # Marco para contener las etiquetas de datos y valores verticalmente
            data_frame = ctk.CTkFrame(left_frame)
            data_frame.pack(pady=10, fill="both", expand=True)

            # Organizar etiquetas y valores verticalmente
            labels = ["X1:", "X2:", "YD (Salida deseada):", "YO (Salida obtenida):", "Error Cuadrático Medio (MSE):"]
            value_labels = []

            for i, text in enumerate(labels):
                lbl = ctk.CTkLabel(data_frame, text=text, font=label_font)
                lbl.pack(pady=(5, 0))
                val_lbl = ctk.CTkLabel(data_frame, text="", font=value_font)
                val_lbl.pack(pady=(0, 5))
                value_labels.append(val_lbl)

            label_input_x1_value, label_input_x2_value, label_yd_value, label_yo_value, label_mse_value = value_labels

            # Inicializar salidas
            update_outputs()

            # Crear gráfica con la arquitectura de la red neuronal
            def crear_grafica(frame):
                # Crear figura con estilo de fondo oscuro
                plt.style.use('dark_background')
                fig, ax = plt.subplots(figsize=(8, 6), facecolor='#2E3440')
                ax.set_facecolor('#2E3440')

                # Definir posiciones de neuronas
                num_inputs = 2
                num_hidden = len(global_wh)
                num_outputs = 1

                # Posiciones en el eje x
                x_input = 0
                x_hidden = 2
                x_output = 4

                # Posiciones en el eje y para distribuir las neuronas uniformemente
                y_input = [i * (6 / (num_inputs - 1)) if num_inputs > 1 else 3 for i in range(num_inputs)]
                y_hidden = [i * (6 / (num_hidden - 1)) if num_hidden > 1 else 3 for i in range(num_hidden)]
                y_output = [3]

                # Colores personalizados
                color_input = '#88C0D0'
                color_hidden = '#81A1C1'
                color_output = '#BF616A'
                color_text = '#D8DEE9'
                color_connection = '#5E81AC'

                # Dibujar neuronas de entrada
                for idx, y in enumerate(y_input):
                    circle = plt.Circle((x_input, y), 0.3, color=color_input, ec='black', zorder=4)
                    ax.add_artist(circle)
                    ax.text(x_input - 0.5, y, f"X{idx+1}", fontsize=12, ha='right', va='center', color=color_text)

                # Dibujar neuronas ocultas
                for idx, y in enumerate(y_hidden):
                    circle = plt.Circle((x_hidden, y), 0.3, color=color_hidden, ec='black', zorder=4)
                    ax.add_artist(circle)
                    ax.text(x_hidden, y + 0.5, f"H{idx+1}", fontsize=12, ha='center', color=color_text)

                # Dibujar neuronas de salida
                for idx, y in enumerate(y_output):
                    circle = plt.Circle((x_output, y), 0.3, color=color_output, ec='black', zorder=4)
                    ax.add_artist(circle)
                    ax.text(x_output + 0.5, y, f"Y", fontsize=12, ha='left', va='center', color=color_text)

                # Dibujar conexiones y mostrar pesos
                # Conexiones de entrada a ocultas
                for i, (xi, yi) in enumerate(zip([x_input]*num_inputs, y_input)):
                    for j, (xj, yj) in enumerate(zip([x_hidden]*num_hidden, y_hidden)):
                        weight = global_wh[j][i]
                        ax.plot([xi, xj], [yi, yj], color=color_connection, zorder=1)
                        xm = (xi + xj) / 2
                        ym = (yi + yj) / 2
                        ax.text(xm, ym, f"{weight:.2f}", fontsize=10, color=color_text, zorder=5,
                                bbox=dict(facecolor='#3B4252', edgecolor='none', pad=1))

                # Conexiones de ocultas a salida
                for j, (xj, yj) in enumerate(zip([x_hidden]*num_hidden, y_hidden)):
                    weight = global_w0[j]
                    ax.plot([xj, x_output], [yj, y_output[0]], color=color_connection, zorder=1)
                    xm = (xj + x_output) / 2
                    ym = (yj + y_output[0]) / 2
                    ax.text(xm, ym, f"{weight:.2f}", fontsize=10, color=color_text, zorder=5,
                            bbox=dict(facecolor='#3B4252', edgecolor='none', pad=1))

                # Configurar el gráfico
                ax.axis('off')
                ax.set_xlim(-1, 5)
                ax.set_ylim(-1, 7)
                fig.tight_layout()

                for widget in frame.winfo_children():
                    widget.destroy()

                canvas = FigureCanvasTkAgg(fig, master=frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            # Crear gráfica de arquitectura
            crear_grafica(architecture_frame)

            # Crear gráfica de entrenamiento en Aplicación
            if global_data_json:
                epocas = list(global_data_json.keys())
                errores = list(global_data_json.values())
                graficar_errores(epocas, errores, training_plot_frame)
            else:
                print("No hay datos de entrenamiento disponibles para graficar.")

        else:
            # Si aún no se ha realizado el entrenamiento
            label_info = ctk.CTkLabel(left_frame, text="No se ha entrenado el modelo aún.",
                                      font=label_font)
            label_info.pack(pady=20)

    def limpiar_interfaz():
        for widget in root.grid_slaves():
            if int(widget.grid_info()['row']) != 0:
                widget.destroy()

    def cerrar_ventana():
        root.destroy()

    # Inicializar la ventana principal
    ctk.set_appearance_mode("dark")  # Opciones: "dark", "light", "system"
    ctk.set_default_color_theme("dark-blue")  # Opciones: "blue", "green", "dark-blue"

    root = ctk.CTk()
    root.title("BackP | Juan Moreno - Nicolás Rodríguez Torres")
    root.geometry("1000x700")  # Aumentamos el tamaño para acomodar ambas gráficas

    root.protocol("WM_DELETE_WINDOW", cerrar_ventana)

    centrar_ventana(root, 1200, 800)

    # Definir fuentes
    title_font = ctk.CTkFont(family="Helvetica", size=24, weight="bold")
    subtitle_font = ctk.CTkFont(family="Helvetica", size=18, weight="bold")
    label_font = ctk.CTkFont(family="Helvetica", size=14)
    value_font = ctk.CTkFont(family="Helvetica", size=14, weight="bold")
    button_font = ctk.CTkFont(family="Helvetica", size=14, weight="bold")

    # Botones de selección
    frame_seleccion = ctk.CTkFrame(root)
    frame_seleccion.grid(row=0, column=0, columnspan=2, padx=10, pady=10, sticky="ew")
    frame_seleccion.grid_columnconfigure(0, weight=1)
    frame_seleccion.grid_columnconfigure(1, weight=1)

    boton_aplicacion = ctk.CTkButton(frame_seleccion, text="Aplicación", command=mostrar_aplicacion,
                                     fg_color="#88C0D0", hover_color="#81A1C1", font=button_font)
    boton_aplicacion.grid(row=0, column=0, padx=10, pady=10, sticky="e")

    boton_entrenamiento = ctk.CTkButton(frame_seleccion, text="Entrenamiento", command=mostrar_entrenamiento,
                                        fg_color="#81A1C1", hover_color="#5E81AC", font=button_font)
    boton_entrenamiento.grid(row=0, column=1, padx=10, pady=10, sticky="w")

    # Iniciar con la vista "Aplicación"
    mostrar_aplicacion()

    # Iniciar la aplicación
    root.mainloop()

graficar()
