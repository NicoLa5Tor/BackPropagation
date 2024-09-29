from Resources.fit import Fit
import os
import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variables globales para almacenar los resultados del entrenamiento
global_data_json = None
global_results = None
global_mse = None

def fit(neuronas=2, alp=0.5, epocas=6000, error=0.01, momentum=0.9):
    obj_fit = Fit()
    
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
    
    num_neurons_hidden = int(neuronas)
    num_inputs = 2
    num_outputs = 1
    
    wh, th, w0, tk = obj_fit.initialize_weights(num_inputs, num_neurons_hidden, num_outputs)
    
    alpha = alp
    Ep = epocas
    ET = error

    # Eliminar momentum de la llamada al método de entrenamiento
    w0, wh, th, tk, data_json = obj_fit.train(
        patterns, wh, th, w0, tk, alpha, Ep, ET, num_neurons_hidden
    )
    predictions = obj_fit.predict(patterns, wh, th, w0, tk, num_neurons_hidden)

    results = []
    for i, ((inputs, yd), (yd_pred, yo)) in enumerate(zip(patterns, predictions)):
        result = f"Patrón {i+1}: yd = {yd:.6f}, yo = {yo:.6f}"
        print(result)
        # Incluir patrones de entrada en los resultados
        results.append({
            'pattern_name': f"Patrón {i+1}",
            'inputs': inputs,
            'yd': f"{yd:.6f}",
            'yo': f"{yo:.6f}"
        })

    mse = obj_fit.calculate_mse(predictions)
    mse_text = f"Error Cuadrático Medio (MSE): {mse:.6f}"
    print(mse_text)
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
    datos_predefinidos = """Alpha = 0.5
Momentum = 0.9
Épocas = 1000
Error deseado = 0.001
Neuronas ocultas = 2
"""

    def abrir_ventana_edicion():
        if not os.path.exists(archivo):
            with open(archivo, 'w') as file:
                file.write(datos_predefinidos)

        ventana_edicion = ctk.CTkToplevel(root)
        ventana_edicion.title("Editar Archivo de Entrenamiento")
        ventana_edicion.geometry("600x500")

        text_box = ctk.CTkTextbox(ventana_edicion, wrap="word", font=("Arial", 12))
        text_box.pack(fill="both", expand=True, padx=10, pady=10)

        with open(archivo, 'r') as file:
            contenido = file.read()
        text_box.insert(tk.END, contenido)

        def guardar_cambios():
            with open(archivo, 'w') as file:
                contenido = text_box.get("1.0", tk.END)
                file.write(contenido)
            messagebox.showinfo("Guardado", "¡Los cambios se han guardado correctamente!")
            ventana_edicion.destroy()

        guardar_button = ctk.CTkButton(
            ventana_edicion, text="GUARDAR CAMBIOS", command=guardar_cambios,
            fg_color="#81A1C1", hover_color="#5E81AC", font=("Arial", 16, "bold")
        )
        guardar_button.pack(pady=20)

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
            graficar_errores(epocas, errores)
        else:
            messagebox.showwarning("Advertencia", "El archivo no existe. Cárgalo o créalo primero.")

    def graficar_errores(epocas, errores):
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

        for widget in bottom_right_frame.winfo_children():
            widget.destroy()

        canvas = FigureCanvasTkAgg(fig, master=bottom_right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.N)

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
        main_frame.grid_columnconfigure(0, weight=2)  # Peso ajustado a 2 para left_frame
        main_frame.grid_columnconfigure(1, weight=3)  # Peso ajustado a 3 para right_frame

        left_frame = ctk.CTkFrame(main_frame)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        right_frame = ctk.CTkFrame(main_frame)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Usar variables globales para acceder a los resultados del entrenamiento
        if global_results is not None and global_data_json is not None:
            # Crear selector de patrón
            pattern_options = [result['pattern_name'] for result in global_results]
            selected_pattern = tk.StringVar(value=pattern_options[0])

            def update_outputs(*args):
                # Obtener índice del patrón seleccionado
                index = pattern_options.index(selected_pattern.get())
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

            pattern_menu = ctk.CTkOptionMenu(left_frame, variable=selected_pattern, values=pattern_options,
                                             fg_color="#4C566A", button_color="#434C5E",
                                             button_hover_color="#4C566A", font=label_font)
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

            # Crear gráfica con el error a través de las épocas
            def crear_grafica():
                epocas = list(global_data_json.keys())
                errores = list(global_data_json.values())

                fig, ax = plt.subplots(figsize=(6, 4))
                ax.plot(epocas, errores, marker='o', color='#88C0D0')
                ax.set_title("Épocas vs. Error Cuadrático Medio (MSE)")
                ax.set_xlabel("Épocas")
                ax.set_ylabel("Error Cuadrático Medio (MSE)")
                ax.grid(True, color='#4C566A')
                ax.set_facecolor('#2E3440')
                fig.patch.set_facecolor('#2E3440')
                ax.tick_params(colors='#D8DEE9')
                ax.xaxis.label.set_color('#D8DEE9')
                ax.yaxis.label.set_color('#D8DEE9')
                ax.title.set_color('#ECEFF4')
                fig.tight_layout()

                for widget in right_frame.winfo_children():
                    widget.destroy()

                canvas = FigureCanvasTkAgg(fig, master=right_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

            crear_grafica()
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

    def centrar_ventana(root):
        root.update_idletasks()
        ancho_pantalla = root.winfo_screenwidth()
        alto_pantalla = root.winfo_screenheight()

        ancho_ventana = 850
        alto_ventana = 600

        x = int((ancho_pantalla / 2) - (ancho_ventana / 2))
        y = int((alto_pantalla / 2) - (alto_ventana / 2))

        root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

    # Inicializar la ventana principal
    ctk.set_appearance_mode("dark")  # Opciones: "dark", "light", "system"
    ctk.set_default_color_theme("dark-blue")  # Opciones: "blue", "green", "dark-blue"

    root = ctk.CTk()
    root.title("BackP | Juan Moreno - Nicolás Rodríguez Torres")
    root.geometry("850x600")

    root.protocol("WM_DELETE_WINDOW", cerrar_ventana)

    centrar_ventana(root=root)
    
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