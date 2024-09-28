from Resources.fit import Fit
import os
import customtkinter as ctk
from tkinter import messagebox
import os
import tkinter as tk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


def intermediate(data):
    str_data = str(data)
    aux =[]
    data_list = str_data.strip().split(f"\n")
    for x in data_list:
        _,d = x.split("=")
        aux.append(float(d.strip()))
    data_json = fit(alp=aux[0],momentum=aux[1],epocas=aux[2],error=aux[3],neuronas=aux[4])
    return  data_json


    
def fit(neuronas = 2,alp = 0.5,epocas = 6000,error = 0.01,momentum = 0.9):
    # Creación de objetos
    obj_fit = Fit()
        # Definiciones iniciales de patrones
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
    # Ajustes
    num_neurons_hidden = int(neuronas)  # Aumentar el número de neuronas en la capa oculta
    print("las neuronas son ",neuronas)

    num_inputs = 2
    num_outputs = 1
    # Inicializar pesos y umbrales aleatoriamente entre -1 y 1
    wh, th, w0, tk = obj_fit.initialize_weights(num_inputs, num_neurons_hidden, num_outputs)
    # Parámetros de entrenamiento
    alpha = alp  # Reducir la tasa de aprendizaje para un ajuste más gradual
    Ep = epocas     # Aumentar el número de épocas para permitir más entrenamiento
    ET = error   # Reducir el error total mínimo aceptable
    # Entrenamiento de la red neuronal
    w0, wh, th, tk, data_json = obj_fit.train(patterns, wh, th, w0, tk, alpha, Ep, ET, num_neurons_hidden)
    # Predicciones
    predictions = obj_fit.predict(patterns, wh, th, w0, tk, num_neurons_hidden)

    # Comparación de yd vs yo
    for i, (yd, yo) in enumerate(predictions):
        print(f"Patrón {i+1}: yd = {yd:.6f}, yo = {yo:.6f}")

    # Calcular y mostrar el error cuadrático medio (MSE)
    mse = obj_fit.calculate_mse(predictions)
    print(f"Error Cuadrático Medio (MSE): {mse:.6f}")
    return data_json


def graficar():
    archivo = "datos_entrenamiento.txt"  # Nombre del archivo a guardar y cargar

    # Datos predefinidos que estarán en el archivo por defecto
    datos_predefinidos = """                         
Alpha = 0.5    
Momentum = 0.9 
Épocas = 1000     
Error deseado = 0.001    
Neuronas ocultas = 2
"""

    # Función para abrir una ventana de edición del archivo
    def abrir_ventana_edicion():
        # Crear el archivo con los datos predefinidos si no existe
        if not os.path.exists(archivo):
            with open(archivo, 'w') as file:
                file.write(datos_predefinidos)

        # Crear una nueva ventana para la edición del archivo
        ventana_edicion = tk.Toplevel(root)
        ventana_edicion.title("Editar Archivo de Entrenamiento")
        ventana_edicion.geometry("600x500")

        # Cuadro de texto en la ventana emergente para editar el archivo
        text_box = tk.Text(ventana_edicion, wrap="word", font=("Arial", 12), height=20)
        text_box.pack(fill="both", expand=True, padx=10, pady=10)

        # Leer el contenido del archivo y mostrarlo en el cuadro de texto
        with open(archivo, 'r') as file:
            contenido = file.read()
        text_box.insert(tk.END, contenido)

        # Función para guardar los cambios hechos en el archivo
        def guardar_cambios():
            with open(archivo, 'w') as file:
                contenido = text_box.get(1.0, tk.END)  # Obtener el contenido actualizado
                file.write(contenido)
            messagebox.showinfo("Guardado", "¡Los cambios se han guardado correctamente!")
            ventana_edicion.destroy()  # Cerrar la ventana al guardar

        # Botón para guardar los cambios en el archivo
        guardar_button = tk.Button(ventana_edicion, text="GUARDAR CAMBIOS", command=guardar_cambios,
                                   bg="green", fg="white", font=("Arial", 16, "bold"), height=2, width=20)
        guardar_button.pack(pady=20)

    # Función para leer e imprimir el contenido del archivo y graficar
    def leer_txt_y_entrenar():
        if os.path.exists(archivo):
            with open(archivo, 'r') as file:
                contenido = file.read()
            data_son = intermediate(data=contenido)
            # Simulación: Usar datos falsos para épocas y errores por ahora
            epocas = list(data_son.keys())  # Las claves son las épocas
            errores = list(data_son.values())  # Los valores son los errores
            
            # Mostrar el contenido del archivo en la consola
            #print("Contenido del archivo:\n", contenido)

            # Graficar en el contenedor vacío
            graficar_errores(epocas, errores)
        else:
            messagebox.showwarning("Advertencia", "El archivo no existe. Cárgalo o créalo primero.")

    # Función para graficar Épocas vs. Error total
    def graficar_errores(epocas, errores):
        fig, ax = plt.subplots(figsize=(6, 4))  # Ajustar el tamaño de la gráfica para que no exceda el contenedor
        ax.plot(epocas, errores, marker='o')
        ax.set_title("Épocas vs. Error Total")
        ax.set_xlabel("Épocas")
        ax.set_ylabel("Error Total")
        fig.tight_layout()

        # Ajustar la gráfica al tamaño del contenedor sin excederlo
        for widget in bottom_right_frame.winfo_children():
            widget.destroy()  # Eliminar gráficos anteriores

        canvas = FigureCanvasTkAgg(fig, master=bottom_right_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, side=tk.TOP, anchor=tk.N)  # Ajustar la gráfica al contenedor

    # Función para mostrar la interfaz de Entrenamiento
    def mostrar_entrenamiento():
        limpiar_interfaz()

        # Parte superior para imagen y título
        frame_superior = ctk.CTkFrame(root)
        frame_superior.grid(row=1, column=0, columnspan=2, padx=10, pady=10, sticky="nsew")

        # Título de la sección
        texto_label = ctk.CTkLabel(frame_superior, text="BACK PROPAGATION ENTRENAMIENTO", font=ctk.CTkFont(size=24, weight="bold"))
        texto_label.pack(pady=10)

        # Configurar la ventana para que las columnas y filas ocupen todo el ancho y alto
        root.grid_rowconfigure(2, weight=1)
        root.grid_columnconfigure(0, weight=1)

        # Crear el contenedor que contendrá los dos sub-contenedores (uno encima del otro)
        data_frame = ctk.CTkFrame(root)
        data_frame.grid(row=2, column=0, padx=10, pady=10, sticky="nsew")
        data_frame.grid_rowconfigure(0, weight=1)
        data_frame.grid_rowconfigure(1, weight=4)
        data_frame.grid_columnconfigure(0, weight=1)

        # Contenedor superior: Botones en la misma fila, menos espaciados
        top_right_frame = ctk.CTkFrame(data_frame)
        top_right_frame.grid(row=0, column=0, padx=10, pady=5, sticky="nsew")

        # Botón para cargar y editar archivo
        editar_button = ctk.CTkButton(top_right_frame, text="Crear y Editar Archivo", command=abrir_ventana_edicion)
        editar_button.grid(row=0, column=0, padx=2, pady=5, sticky="w")  # Reducir espacio entre los botones

        # Botón para entrenar y graficar (en la misma fila)
        entrenar_button = ctk.CTkButton(top_right_frame, text="Entrenar y Graficar", command=leer_txt_y_entrenar)
        entrenar_button.grid(row=0, column=1, padx=2, pady=5, sticky="e")  # Reducir espacio entre los botones

        # Ajustar el grid del top_right_frame para que los botones estén alineados sin mucho espacio
        top_right_frame.grid_columnconfigure(0, weight=1)
        top_right_frame.grid_columnconfigure(1, weight=1)

        # Contenedor inferior: Para mostrar la gráfica
        global bottom_right_frame
        bottom_right_frame = ctk.CTkFrame(data_frame)
        bottom_right_frame.grid(row=1, column=0, padx=10, pady=10, sticky="nsew")

        # Título del gráfico
        titulo_label = ctk.CTkLabel(bottom_right_frame, text="Gráfico de Épocas vs. Error Total", font=ctk.CTkFont(size=18))
        titulo_label.pack(pady=10)

    # Función para limpiar la interfaz sin borrar los botones de selección
    def limpiar_interfaz():
        for widget in root.grid_slaves():
            if widget.grid_info()['row'] != 0:
                widget.destroy()

    # Función para cerrar la ventana correctamente
    def cerrar_ventana():
        root.quit()  # Cierra correctamente el bucle de la ventana
        root.destroy()  # Destruye la ventana

    def centrar_ventana(root):
        ancho_pantalla = root.winfo_screenwidth()
        alto_pantalla = root.winfo_screenheight()

        ancho_ventana = 850
        alto_ventana = 640

        x = int((ancho_pantalla / 2) - (ancho_ventana / 2))
        y = int((alto_pantalla / 2) - (alto_ventana / 2)) - 30

        root.geometry(f"{ancho_ventana}x{alto_ventana}+{x}+{y}")

    # Crear la ventana principal
    ctk.set_appearance_mode("dark")
    ctk.set_default_color_theme("blue")

    root = ctk.CTk()
    root.title("BackP | Juan Moreno - Nicolás Rodríguez Torres")
    root.geometry("850x650")

    # Asignar la función de cierre al botón de la "X" en la ventana
    root.protocol("WM_DELETE_WINDOW", cerrar_ventana)

    centrar_ventana(root=root)

    # Botones de selección entre "Aplicación" y "Entrenamiento", siempre visibles
    frame_seleccion = ctk.CTkFrame(root)
    frame_seleccion.grid(row=0, column=0, columnspan=2, padx=10, pady=10)

    boton_aplicacion = ctk.CTkButton(frame_seleccion, text="Aplicación", command=lambda: limpiar_interfaz())
    boton_aplicacion.pack(side="left", padx=20)

    boton_entrenamiento = ctk.CTkButton(frame_seleccion, text="Entrenamiento", command=mostrar_entrenamiento)
    boton_entrenamiento.pack(side="left", padx=20)

    # Iniciar con la vista de "Entrenamiento" como predeterminada
    mostrar_entrenamiento()

    # Iniciar la aplicación
    root.mainloop()

graficar()

