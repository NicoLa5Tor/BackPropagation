from Resources.fit import Fit
import os
import customtkinter as ctk
from tkinter import messagebox
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Variables globales
global salidas
salidas = ""

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
    
    w0, wh, th, tk, data_json = obj_fit.train(patterns, wh, th, w0, tk, alpha, Ep, ET, num_neurons_hidden)
    predictions = obj_fit.predict(patterns, wh, th, w0, tk, num_neurons_hidden)

    for i, (yd, yo) in enumerate(predictions):
        print(f"Patrón {i+1}: yd = {yd:.6f}, yo = {yo:.6f}")

    mse = obj_fit.calculate_mse(predictions)
    data_final = f"Error Cuadrático Medio (MSE): {mse:.6f}"
    return data_json

def graficar():
    archivo = "datos_entrenamiento.txt"
    datos_predefinidos = """                         
Alpha = 0.5    
Momentum = 0.9 
Épocas = 1000     
Error deseado = 0.001    
Neuronas ocultas = 2
"""

    def abrir_ventana_edicion():
        if not os.path.exists(archivo):
            with open(archivo, 'w') as file:
                file.write(datos_predefinidos)

        ventana_edicion = tk.Toplevel(root)
        ventana_edicion.title("Editar Archivo de Entrenamiento")
        ventana_edicion.geometry("600x500")

        text_box = tk.Text(ventana_edicion, wrap="word", font=("Arial", 12), height=20)
        text_box.pack(fill="both", expand=True, padx=10, pady=10)

        with open(archivo, 'r') as file:
            contenido = file.read()
        text_box.insert(tk.END, contenido)

        def guardar_cambios():
            with open(archivo, 'w') as file:
                contenido = text_box.get(1.0, tk.END)
                file.write(contenido)
            messagebox.showinfo("Guardado", "¡Los cambios se han guardado correctamente!")
            ventana_edicion.destroy()

        guardar_button = tk.Button(ventana_edicion, text="GUARDAR CAMBIOS", command=guardar_cambios,
                                   bg="green", fg="white", font=("Arial", 16, "bold"), height=2, width=20)
        guardar_button.pack(pady=20)

    def leer_txt_y_entrenar():
        if os.path.exists(archivo):
            with open(archivo, 'r') as file:
                contenido = file.read()
            data_son = intermediate(data=contenido)
            epocas = list(data_son.keys())
            errores = list(data_son.values())
            graficar_errores(epocas, errores)
        else:
            messagebox.showwarning("Advertencia", "El archivo no existe. Cárgalo o créalo primero.")

    def graficar_errores(epocas, errores):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(epocas, errores, marker='o')
        ax.set_title("Épocas vs. Error Total")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Error Total")
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

        texto_label = ctk.CTkLabel(frame_superior, text="BACK PROPAGATION ENTRENAMIENTO", font=ctk.CTkFont(size=24, weight="bold"))
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

        editar_button = ctk.CTkButton(top_right_frame, text="Crear y Editar Archivo", command=abrir_ventana_edicion)
        editar_button.grid(row=0, column=0, padx=2, pady=5, sticky="w")

        entrenar_button = ctk.CTkButton(top_right_frame, text="Entrenar y Graficar", command=leer_txt_y_entrenar)
        entrenar_button.grid(row=0, column=1, padx=2, pady=5, sticky="e")

        top_right_frame.grid_columnconfigure(0, weight=1)
        top_right_frame.grid_columnconfigure(1, weight=1)

        global bottom_right_frame
        bottom_right_frame = ctk.CTkFrame(data_frame)
        bottom_right_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        titulo_label = ctk.CTkLabel(bottom_right_frame, text="Gráfico de Épocas vs. Error Total", font=ctk.CTkFont(size=18))
        titulo_label.pack(pady=10)

    def mostrar_aplicacion():
        limpiar_interfaz()

        frame_superior = ctk.CTkFrame(root)
        frame_superior.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        texto_label = ctk.CTkLabel(frame_superior, text="BACK PROPAGATION APLICACIÓN", font=ctk.CTkFont(size=24, weight="bold"))
        texto_label.pack(pady=10)

        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        main_frame = ctk.CTkFrame(root)
        main_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        main_frame.grid_rowconfigure(0, weight=1)
        main_frame.grid_columnconfigure(0, weight=1)

        left_frame = ctk.CTkFrame(main_frame)
        left_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")
        left_frame.grid_rowconfigure(0, weight=1)
        left_frame.grid_rowconfigure(1, weight=1)
        left_frame.grid_columnconfigure(0, weight=1)

        right_frame = ctk.CTkFrame(main_frame)
        right_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")
        right_frame.grid_rowconfigure(0, weight=1)
        right_frame.grid_columnconfigure(0, weight=1)

        # Crear tabla en el contenedor superior
        def crear_tabla():
            data = [
                ["Patrón 1", "-1.000000", "-0.283774"],
                ["Patrón 2", "0.000000", "-0.804866"],
                ["Patrón 3", "1.000000", "0.911442"],
                ["Patrón 4", "0.000000", "-0.810168"],
                ["Patrón 5", "-1.000000", "-0.799762"],
                ["Patrón 6", "0.000000", "-0.809105"],
                ["Patrón 7", "-1.000000", "-0.794135"],
                ["Patrón 8", "1.000000", "-0.059192"],
                ["Patrón 9", "-1.000000", "-0.767832"],
                ["Patrón 10", "1.000000", "0.918179"],
                ["Error Cuadrático Medio (MSE)", "", "0.374461"]
            ]

            scroll_frame = ctk.CTkScrollableFrame(left_frame)
            scroll_frame.grid(row=0, column=0, padx=5, pady=5, sticky="nsew")

            encabezados = ["Patrón", "YD (Salida deseada)", "YO (Salida obtenida)"]
            # Crear encabezados de la tabla
            for j, header in enumerate(encabezados):
                label = ctk.CTkLabel(scroll_frame, text=header, font=ctk.CTkFont(size=14, weight="bold"))
                label.grid(row=0, column=j, padx=5, pady=5)

            # Llenar los datos de la tabla
            for i, fila in enumerate(data):
                for j, valor in enumerate(fila):
                    label = ctk.CTkLabel(scroll_frame, text=valor, font=ctk.CTkFont(size=12))
                    label.grid(row=i + 1, column=j, padx=5, pady=5)

        crear_tabla()

        # Crear gráfica en el contenedor inferior
        def crear_grafica():
            data_json = {1: 0.25, 2: 0.19, 3: 0.17, 4: 0.15, 5: 0.12, 6: 0.10, 7: 0.08, 8: 0.07, 9: 0.05, 10: 0.04}
            epocas = list(data_json.keys())
            errores = list(data_json.values())

            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(epocas, errores, marker='o')
            ax.set_title("Épocas vs. Error Cuadrático Medio (MSE)")
            ax.set_xlabel("Épocas")
            ax.set_ylabel("Error Cuadrático Medio (MSE)")
            fig.tight_layout()

            # Eliminar gráficos anteriores
            for widget in right_frame.winfo_children():
                widget.destroy()

            # Crear un canvas para la gráfica
            canvas = FigureCanvasTkAgg(fig, master=right_frame)
            canvas.draw()
            canvas.get_tk_widget().pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.N)

        crear_grafica()

    # Función para limpiar la interfaz sin borrar los botones de selección
    def limpiar_interfaz():
        for widget in root.grid_slaves():
            if widget.grid_info()['row'] != 0:
                widget.destroy()

    # Función para cerrar la ventana correctamente
    def cerrar_ventana():
        root.quit()
        root.destroy()

    def centrar_ventana(root):
        ancho_pantalla = root.winfo_screenwidth()
        alto_pantalla = root.winfo_screenheight()

        ancho_ventana = 850
        alto_ventana = 570  # Ajuste de la altura de la ventana

        x = int((ancho_pantalla / 2) - (ancho_ventana / 2))
        y = int((alto_pantalla / 2) - (alto_ventana / 2)) - 30

        root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

    # Crear la ventana principal
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("BackP | Juan Moreno - Nicolás Rodríguez Torres")
    root.geometry("850x570")  # Ajuste de la altura de la ventana

    # Asignar la función de cierre al botón de la "X" en la ventana
    root.protocol("WM_DELETE_WINDOW", cerrar_ventana)

    centrar_ventana(root=root)

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

graficar()
