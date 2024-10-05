import os
import json
import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk
import matplotlib.pyplot as plt
from Resources.fit import Fit
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tkinter import ttk  # Importar ttk para la tabla

# Variables globales
salidas = ""

# Variables globales para almacenar datos de entrenamiento
entrenamiento_epocas = []
entrenamiento_error_total = []

# Variables globales para almacenar resultados de la aplicación
aplicacion_patrones = []
aplicacion_Y0 = []
aplicacion_Y0_decimal = []
aplicacion_Y0_binario = []
aplicacion_Ydeseado = []
correct_predictions = []

# Definir los patrones globalmente
patterns = [
    ((0, 0, 0, 0), 0),
    ((0, 0, 0, 1), 1),
    ((0, 0, 1, 0), 1),
    ((0, 0, 1, 1), 0),
    ((0, 1, 0, 0), 1),
    ((0, 1, 0, 1), 0),
    ((0, 1, 1, 0), 0),
    ((0, 1, 1, 1), 1),
    ((1, 0, 0, 0), 1),
    ((1, 0, 0, 1), 0),
    ((1, 0, 1, 0), 0),
    ((1, 0, 1, 1), 1),
    ((1, 1, 0, 0), 0),
    ((1, 1, 0, 1), 1),
    ((1, 1, 1, 0), 1),
    ((1, 1, 1, 1), 0)
]

def get_salidas():
    global salidas
    return salidas

def set_salidas(data):
    global salidas
    salidas = data

def intermediate(data):
    str_data = str(data)
    aux = []
    data_list = str_data.strip().split(f"\n")
    for x in data_list:
        _, d = x.split("=")
        aux.append(float(d.strip()))
    data_json = fit(alp=aux[0], momentum=aux[1], epocas=aux[2], error=aux[3], neuronas=aux[4])
    return data_json

def fit(neuronas=8, alp=0.1, epocas=10000, error=0.01, momentum=0.4, update_callback=None):
    global patterns  # Referenciar los patrones globales
    obj_fit = Fit()

    # Normalizar entradas al rango [-1, 1]
    normalized_patterns = [([(x - 0.5) * 2 for x in inputs], yd) for inputs, yd in patterns]
    num_neurons_hidden = int(neuronas)
    num_inputs = 4
    num_outputs = 1

    wh, th, w0, tk = obj_fit.initialize_weights(num_inputs, num_neurons_hidden, num_outputs)

    # Inicializar deltas anteriores para el momentum
    delta_w0_prev = [0] * len(w0)
    delta_wh_prev = [0] * len(wh)
    delta_th_prev = [0] * len(th)
    delta_tk_prev = 0

    # Entrenar la red con momentum
    epoch = 0
    data_json = {}

    while epoch < epocas:
        error_total = 0
        for x, yd in normalized_patterns:
            yh = obj_fit.obj_f.forward_hidden_layer(x, wh, th, num_neurons_hidden)
            yok1 = obj_fit.obj_f.forward_output_layer(yh, w0, tk, num_neurons_hidden)
            Dok1, Dh = obj_fit.obj_f.calculate_deltas(yd, yok1, yh, w0)

            # Actualizar pesos con momentum
            w0, wh, th, tk, delta_w0_prev, delta_wh_prev, delta_th_prev, delta_tk_prev = obj_fit.obj_f.update_weights_momentum(
                Dok1, Dh, yh, w0, wh, th, tk, x, alp, momentum, num_neurons_hidden,
                delta_w0_prev, delta_wh_prev, delta_th_prev, delta_tk_prev
            )
            error_total += obj_fit.obj_f.calculate_error(Dok1)

        data_json[epoch + 1] = error_total
        if update_callback:
            update_callback(epoch + 1, error_total)  # Actualizar consola con la época y error
        epoch += 1

    predictions = obj_fit.predict(normalized_patterns, wh, th, w0, tk, num_neurons_hidden)
    Y0Binary = []
    Y0Decimal = []
    for i, (yd, yo) in enumerate(predictions):
        Y0Binary.append(yo)
        print(f"Patrón {i + 1}: yd = {yd:.6f}, yo = {yo:.6f}")

    mse = obj_fit.calculate_mse(predictions)
    print(f"Error Cuadrático Medio (MSE): {mse:.6f}")
    Y0Decimal = Y0Binary

    # Crear un dict con epochs como keys y error_total como values
    resultado = {str(epoch): error_total for epoch, error_total in data_json.items()}

    return [1 if x > 0.5 else 0 for x in Y0Binary], Y0Decimal, json.dumps(resultado)

def graficar():
    # Variables globales
    global root, console_textbox

    # Función de validación para permitir solo números y un punto decimal
    def validate_float(new_value):
        if new_value == "" or new_value == ".":
            return True
        try:
            float(new_value)
            return True
        except ValueError:
            return False

    def mostrar_aplicacion():
        limpiar_interfaz()

        # Parte superior para imagen y título
        frame_superior = ctk.CTkFrame(root)
        frame_superior.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Texto sobre la imagen
        texto_label = ctk.CTkLabel(
            frame_superior,
            text="RNA BACK-PROPAGATION (A)",
            font=ctk.CTkFont(size=24, weight="bold")
        )
        texto_label.place(relx=0.5, rely=0.5, anchor="center")

        # Crear un CTkScrollableFrame para el contenido inferior
        main_scrollable_frame = ctk.CTkScrollableFrame(root, width=800, height=600)
        main_scrollable_frame.grid(row=2, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Configurar la grilla del scrollable frame
        main_scrollable_frame.grid_rowconfigure(0, weight=1)  # left_frame
        main_scrollable_frame.grid_rowconfigure(1, weight=1)  # data_frame
        main_scrollable_frame.grid_columnconfigure(0, weight=1)

        # Parte inferior: dos contenedores uno sobre el otro

        # left_frame
        left_frame = ctk.CTkFrame(main_scrollable_frame, height=345)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # data_frame
        data_frame = ctk.CTkFrame(main_scrollable_frame, height=345)
        data_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Dividir el left_frame en dos filas (superior más grande que inferior)
        left_frame.grid_rowconfigure(0, weight=3)  # Fila superior con más peso
        left_frame.grid_rowconfigure(1, weight=1)  # Fila inferior con menos peso
        left_frame.grid_columnconfigure(0, weight=1)  # Aseguramos que las columnas tomen todo el ancho

        # Parte superior del left_frame (Gráfica Error vs Épocas)
        top_left_frame = ctk.CTkFrame(left_frame)
        top_left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        top_left_label = ctk.CTkLabel(
            top_left_frame,
            text="Gráfica Error vs Épocas",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        top_left_label.pack(pady=10)  # Colocar el texto

        # Crear un frame para la gráfica
        grafica_frame = ctk.CTkFrame(top_left_frame)
        grafica_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Parte inferior del left_frame (Pesos Finales)
        bottom_left_frame = ctk.CTkFrame(left_frame)
        bottom_left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        bottom_left_label = ctk.CTkLabel(
            bottom_left_frame,
            text="Pesos Finales",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        bottom_left_label.pack(pady=10)  # Colocar el texto

        # Crear una tabla debajo de Pesos Finales
        tabla_frame = ctk.CTkFrame(bottom_left_frame)
        tabla_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Configurar la tabla usando ttk.Treeview
        columns = ("#Patrón", "Patrón de Entrada", "Y0", "Y0 Decimal", "Y0 Binario")
        tree = ttk.Treeview(tabla_frame, columns=columns, show='headings')

        # Definir los encabezados
        for col in columns:
            tree.heading(col, text=col)
            tree.column(col, anchor="center")

        # Agregar scrollbar a la tabla
        scrollbar = ttk.Scrollbar(tabla_frame, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        tree.pack(fill='both', expand=True)

        # Función para mostrar descripción al seleccionar una fila
        def on_row_select(event):
            selected_item = tree.focus()
            if not selected_item:
                return
            values = tree.item(selected_item, 'values')
            patron = values[1]
            Y0 = int(values[2])  # Convertir a entero
            Y0_decimal = float(values[3])  # Convertir a float
            Y0_binario = int(values[4])  # Convertir a entero
            ydeseado = patterns[int(values[0])][1]  # Obtener el valor deseado

            # Comprobar si la predicción es correcta
            correcto = "Sí" if (Y0 == ydeseado) else "No"

            descripcion = (
                f"Patrón de Entrada: {patron}\n"
                f"Ydeseado: {ydeseado}\n"
                f"Y0: {Y0}\n"
                f"Y0 Decimal: {Y0_decimal}\n"
                f"Y0 Binario: {Y0_binario}\n"
                f"Resultado Correcto: {correcto}"
            )
            messagebox.showinfo("Detalle del Patrón", descripcion)

        # Vincular el evento de selección
        tree.bind("<<TreeviewSelect>>", on_row_select)

        # Mostrar la gráfica si hay datos de entrenamiento
        global entrenamiento_epocas, entrenamiento_error_total
        if entrenamiento_epocas and entrenamiento_error_total:
            # Limpiar la gráfica anterior si existe
            for widget in grafica_frame.winfo_children():
                widget.destroy()

            # Crear una figura de matplotlib
            fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
            ax.plot(entrenamiento_epocas, entrenamiento_error_total, linestyle='-', color='b')  # Línea continua
            ax.set_title("Error Total vs Épocas")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Error Total")
            ax.grid(True)

            # Crear un canvas de matplotlib y agregarlo al frame
            canvas = FigureCanvasTkAgg(fig, master=grafica_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill='both', expand=True)

        # Llenar la tabla con los resultados de la aplicación
        global aplicacion_patrones, aplicacion_Y0, aplicacion_Y0_decimal, aplicacion_Y0_binario, aplicacion_Ydeseado, correct_predictions
        if aplicacion_patrones:
            for i in range(len(aplicacion_patrones)):
                patron_str = ", ".join(map(str, aplicacion_patrones[i]))
                Y0 = aplicacion_Y0[i]
                Y0_decimal = aplicacion_Y0_decimal[i]
                Y0_binario = aplicacion_Y0_binario[i]
                # Aquí se conserva el patrón y el resultado correcto
                tree.insert("", "end", values=(i, patron_str, Y0, Y0_decimal, Y0_binario))

    # Función para mostrar la interfaz de Entrenamiento
    def mostrar_entrenamiento():
        # Parte superior para imagen y título
        def execute_training():
            global entrenamiento_epocas, entrenamiento_error_total
            global aplicacion_patrones, aplicacion_Y0, aplicacion_Y0_decimal, aplicacion_Y0_binario, aplicacion_Ydeseado, correct_predictions
            try:
                # Obtener valores de las entradas
                error = float(error_deseado_entry.get())
                alpha = float(alpha_entry.get())
                epoca = int(epocas_entry.get())
                momentum = float(momentum_entry.get())
                neuronas = int(neuronas_entry.get())

                # Limpiar el TextBox antes de mostrar nuevos resultados
                console_textbox.delete(1.0, tk.END)

                # Callback para actualizar el TextBox en cada época
                def update_console(epoch, error_total):
                    console_textbox.insert(tk.END, f"Epoch {epoch}: Error total = {error_total}\n")
                    console_textbox.see(tk.END)  # Desplazar hacia abajo
                    root.update_idletasks()  # Actualizar la interfaz gráfica

                # Llamar a la función fit y mostrar resultados en el TextBox (consola)
                Y0Binary, Y0Decimal, data_son = fit(alp=alpha, epocas=epoca, error=error, momentum=momentum, neuronas=neuronas, update_callback=update_console)

                # Mostrar resultados finales
                console_textbox.insert(tk.END, "Resultados finales:\n")
                console_textbox.insert(tk.END, f"{Y0Binary}\n")

                # Parsear el JSON recibido
                datos = json.loads(data_son)

                # Extraer épocas y error total
                epocas_list = sorted([int(k) for k in datos.keys()])
                error_total = [datos[str(k)] for k in epocas_list]

                if not epocas_list or not error_total:
                    messagebox.showerror("Error", "Datos insuficientes para graficar.")
                    return

                # Almacenar los datos de entrenamiento en variables globales
                entrenamiento_epocas = epocas_list
                entrenamiento_error_total = error_total

                # Limpiar la gráfica anterior si existe
                for widget in grafica_frame.winfo_children():
                    widget.destroy()

                # Crear una figura de matplotlib
                fig, ax = plt.subplots(figsize=(5, 4), dpi=100)
                ax.plot(epocas_list, error_total, linestyle='-', color='b')  # Línea continua
                ax.set_title("Error Total vs Épocas")
                ax.set_xlabel("Épocas")
                ax.set_ylabel("Error Total")
                ax.grid(True)

                # Crear un canvas de matplotlib y agregarlo al frame
                canvas = FigureCanvasTkAgg(fig, master=grafica_frame)
                canvas.draw()
                canvas.get_tk_widget().pack(fill='both', expand=True)

                # Almacenar los resultados para la aplicación
                aplicacion_patrones = [pat[0] for pat in patterns]
                aplicacion_Y0 = Y0Binary
                aplicacion_Y0_decimal = Y0Decimal
                aplicacion_Y0_binario = [1 if y > 0.5 else 0 for y in Y0Binary]
                aplicacion_Ydeseado = [pat[1] for pat in patterns]
                correct_predictions = [aplicacion_Y0_binario[i] == aplicacion_Ydeseado[i] for i in range(len(patterns))]

            except ValueError:
                messagebox.showerror("Error", "Por favor, ingresa valores válidos.")
            except json.JSONDecodeError:
                messagebox.showerror("Error", "Error al procesar los datos JSON.")
            except Exception as e:
                messagebox.showerror("Error", f"Ocurrió un error: {str(e)}")

        frame_superior = ctk.CTkFrame(root)
        frame_superior.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")
        root.grid_rowconfigure(1, weight=0)  # Fila de la imagen no se expande
        print("Cargando imagen desde:", os.path.abspath("Udec.jpg"))
        # Texto sobre la imagen
        texto_label = ctk.CTkLabel(frame_superior, text="RNA BACK-PROPAGATION (A) ENTRENAMIENTO",
                                   font=ctk.CTkFont(size=24, weight="bold"))
        texto_label.place(relx=0.5, rely=0.5, anchor="center")

        # Parte inferior: dos columnas
        left_frame = ctk.CTkScrollableFrame(root)
        left_frame.grid(row=2, column=0, padx=0, pady=0, sticky="nsew")
        data_frame = ctk.CTkScrollableFrame(root, height=345)
        data_frame.grid(row=2, column=1, padx=0, pady=0, sticky="nsew")

        # Configurar el tamaño de las columnas (izquierda más grande que derecha)
        root.grid_columnconfigure(0, weight=3)  # Columna izquierda más grande
        root.grid_columnconfigure(1, weight=1)  # Columna derecha más pequeña
        root.grid_rowconfigure(2, weight=1)     # Fila 2 se expande

        # Configurar el left_frame
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        # Contenedor para Historial de Entrenamiento
        history_frame = ctk.CTkFrame(left_frame)
        history_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        history_label = ctk.CTkLabel(history_frame, text="Historial de Entrenamiento",
                                       font=ctk.CTkFont(size=16, weight="bold"))
        history_label.pack(pady=5)

        # Crear un TextBox para mostrar el historial
        console_textbox = ctk.CTkTextbox(history_frame, height=150)
        console_textbox.pack(fill='both', expand=True)

        # Parte superior del left_frame (Gráfica Error vs Épocas)
        top_left_frame = ctk.CTkFrame(left_frame)
        top_left_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")
        top_left_label = ctk.CTkLabel(
            top_left_frame,
            text="Gráfica Error vs Épocas",
            font=ctk.CTkFont(size=16, weight="bold")
        )
        top_left_label.pack(pady=10)  # Colocar el texto

        # Crear un frame para la gráfica
        grafica_frame = ctk.CTkFrame(top_left_frame)
        grafica_frame.pack(fill='both', expand=True, padx=10, pady=10)

        # Elementos de entrada en data_frame
        error_deseado_label = ctk.CTkLabel(data_frame, text="Error Deseado:")
        error_deseado_label.pack(pady=5)
        vcmd = (root.register(validate_float), "%P")
        error_deseado_entry = ctk.CTkEntry(data_frame, width=200, placeholder_text="Ej: 0.001")
        error_deseado_entry.pack(pady=5)
        error_deseado_entry.configure(validate="key", validatecommand=vcmd)

        alpha_label = ctk.CTkLabel(data_frame, text="Alpha:")
        alpha_label.pack(pady=5)
        alpha_entry = ctk.CTkEntry(data_frame, width=200, placeholder_text="Ej: 0.5")
        alpha_entry.pack(pady=5)
        alpha_entry.configure(validate="key", validatecommand=vcmd)

        epocas_label = ctk.CTkLabel(data_frame, text="Épocas:")
        epocas_label.pack(pady=5)
        epocas_entry = ctk.CTkEntry(data_frame, width=200, placeholder_text="Ej: 5000")
        epocas_entry.pack(pady=5)
        epocas_entry.configure(validate="key", validatecommand=vcmd)

        momentum_label = ctk.CTkLabel(data_frame, text="Momentum:")
        momentum_label.pack(pady=5)
        momentum_entry = ctk.CTkEntry(data_frame, width=200, placeholder_text="Ej: 0.9")
        momentum_entry.pack(pady=5)
        momentum_entry.configure(validate="key", validatecommand=vcmd)

        neuronas_label = ctk.CTkLabel(data_frame, text="Neuronas:")
        neuronas_label.pack(pady=5)
        neuronas_entry = ctk.CTkEntry(data_frame, width=200, placeholder_text="Ej: 8")
        neuronas_entry.pack(pady=5)
        neuronas_entry.configure(validate="key", validatecommand=vcmd)

        # Botón para ejecutar
        ejecutar_button = ctk.CTkButton(data_frame, text="Ejecutar", command=execute_training)
        ejecutar_button.pack(pady=20)

    # Función para limpiar la interfaz sin borrar los botones de selección
    def limpiar_interfaz():
        for widget in root.grid_slaves():
            if widget.grid_info()['row'] != 0:  # Deja intacta la fila con los botones de selección
                widget.destroy()

    def centrar_ventana(root):
        # Obtén las dimensiones de la pantalla
        ancho_pantalla = root.winfo_screenwidth()
        alto_pantalla = root.winfo_screenheight()

        # Obtén las dimensiones de la ventana
        ancho_ventana = 850
        alto_ventana = 640  # Corregido de 'alto_pantalla' a 'alto_ventana'

        # Calcula la posición de la ventana
        x = int((ancho_pantalla / 2) - (ancho_ventana / 2))
        y = int((alto_pantalla / 2) - (alto_ventana / 2)) - 30

        # Establece la geometría de la ventana
        root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

    # Crear la ventana principal
    ctk.set_appearance_mode("dark")  # Modo oscuro
    ctk.set_default_color_theme("blue")  # Tema azul por defecto

    root = ctk.CTk()
    root.title("Adeline | Juan Moreno - Nicolás Rodríguez Torres")
    root.geometry("850x650")  # Mantener el tamaño original
    centrar_ventana(root=root)

    # Configurar la grilla de la ventana principal
    root.grid_rowconfigure(0, weight=0)  # Fila de los botones de selección
    root.grid_rowconfigure(1, weight=0)  # Fila para frame_superior
    root.grid_rowconfigure(2, weight=1)  # Fila para main_scrollable_frame
    root.grid_columnconfigure(0, weight=1)
    root.grid_columnconfigure(1, weight=1)

    # Botones de selección entre "Aplicación" y "Entrenamiento", siempre visibles
    frame_seleccion = ctk.CTkFrame(root)
    frame_seleccion.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    boton_aplicacion = ctk.CTkButton(frame_seleccion, text="Aplicación", command=mostrar_aplicacion)
    boton_aplicacion.pack(side="left", padx=20)

    boton_entrenamiento = ctk.CTkButton(frame_seleccion, text="Entrenamiento", command=mostrar_entrenamiento)
    boton_entrenamiento.pack(side="left", padx=20)

    # Iniciar con la vista de "Aplicación" como predeterminada
    mostrar_aplicacion()

    # Iniciar la aplicación
    root.mainloop()

# Ejecutar la función principal
graficar()
