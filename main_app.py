import gi
import os
import threading
import math
import socket
import pickle
import numpy as np
import time
import sys

gi.require_version("Gtk", "3.0")
gi.require_version('Gst', '1.0')

from gi.repository import GObject
from gi.repository import Gtk, Gdk, Gst, GLib, cairo, Pango
from utils.mfccs import compute_mfccs
from neural_network_class import NeuralNetwork

class LungDiagnosisWindow(Gtk.Window):
    def __init__(self, host):
        # Inicializar la ventana principal
        Gtk.Window.__init__(self, title="Lung Diagnosis")

        # Configurar la ventana para mostrarla en pantalla completa
        self.fullscreen()

        # Crear un contenedor de tipo overlay para superponer widgets
        overlay = Gtk.Overlay()
        self.add(overlay)

        # Crear el contenedor principal por encima del fondo
        self.main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        overlay.add_overlay(self.main_box)

        # Caja para los botones en la parte superior derecha
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL)
        overlay.add_overlay(button_box)
        button_box.set_halign(Gtk.Align.END)  # Alinear a la derecha
        button_box.set_valign(Gtk.Align.START)  # Alinear en la parte superior
        button_box.set_margin_top(20)
        button_box.set_margin_end(20)
        
        self.add_button(button_box, "images/info.png", self.on_info, "Info Button")
        self.add_button(button_box, "images/reboot.png", self.on_reboot, "Reboot Button")
        self.add_button(button_box, "images/shutdown.png", self.on_exit, "Shutdown Button")

        # Crear el botón central "Start"
        self.start_button = Gtk.Button(label="START")
        self.start_button.set_size_request(300, 100)  # Definir tamaño del botón
        self.start_button.get_style_context().add_class("start-button")  # Añadir clase CSS personalizada
        self.start_button.connect("clicked", self.on_start_clicked)

        # Centrar el botón dentro del contenedor principal
        self.main_box.set_valign(Gtk.Align.CENTER)  # Cambiar a CENTER
        self.main_box.set_halign(Gtk.Align.CENTER)  # Cambiar a CENTER
        self.main_box.pack_start(self.start_button, True, True, 0)  # Usar espacio expandible

        # Inicializar variables para la animación
        self.is_beating = False
        self.max_radius = 65  # Radio máximo
        self.min_radius = 20  # Radio mínimo
        self.current_radius = 20  # Radio actual
        self.direction = 1  # 1 para aumentar, -1 para disminuir
        self.pulse_count = 5  # Cantidad de círculos en la animación
        self.drawing_area = None  # Inicializar el área de dibujo como None

        # Aplicar el estilo CSS con la imagen de fondo a la ventana
        self.apply_css()

        self.diseases = {0: "Bronchiectasis", 1: "Bronchiolitis", 2: "COPD", 3: "Healthy", 4: "Pneumonia", 5: "URTI"}
        self.host = host 
        self.port = 12345
        self.client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.network = NeuralNetwork("models/modelrawimp.tflite")
        self.init = True

    def add_button(self, container, icon_path, callback, tooltip_text):
        """Crear un botón con un ícono y agregarlo al contenedor."""
        event_box = Gtk.EventBox()
        icon = Gtk.Image()
        icon.set_from_file(icon_path)
        event_box.add(icon)
        event_box.set_tooltip_text(tooltip_text)
        event_box.connect("button-press-event", callback)
        container.pack_start(event_box, False, False, 10)

    def apply_css(self):
        """Aplicar un estilo CSS a la ventana con una imagen de fondo."""
        css_provider = Gtk.CssProvider()

        # Definir el estilo CSS para la ventana con una imagen de fondo y el botón "Start"
        css = """
        window {
            background-color: rgba(255, 255, 255, 1);
            background-image: url("images/background.jpg"); /* Ruta de la imagen de fondo */
            background-size: 60%;  /* Escalar la imagen para cubrir toda la ventana */
            background-repeat: no-repeat;  /* No repetir la imagen */
            background-position: bottom;  /* Centrar la imagen */
        }
        eventbox {
            background-color: transparent; /* Fondo transparente para botones */
            border: none;
            box-shadow: none;
        }
        eventbox:hover {
            background-color: #f5f5f5; /* Fondo gris claro al pasar el ratón */
        }
        dialog {
            background-color: #2E2E2E; /* Fondo del cuadro de diálogo */
            border-radius: 10px;
        }
        dialog label {
            color: white; /* Color del texto dentro del cuadro de diálogo */
            font-size: 14px;
        }
        /* Estilo personalizado para el botón "Start" */
        .start-button {
            font-size: 30px;  /* Tamaño de fuente grande para el botón */
            color: white;
            background: rgba(0, 85, 255, 0.5); /* Color de fondo azul con transparencia */
            border-radius: 15px;  /* Bordes redondeados */
            padding: 20px;  /* Espaciado interno */
            border: 2px solid rgba(0, 51, 187, 0.5); /* Borde sólido con transparencia */
        }
        .start-button:hover {
            background: rgba(0, 44, 204, 0.5); /* Color de fondo azul al hacer hover */
        }
        /* Estilos para el cuadro de resultados */
        .results-frame {
            background-color: rgba(255, 255, 255, 0.5); /* Fondo blanco con algo de transparencia */
            border: 2px solid rgba(0, 85, 255, 0.7); /* Borde azul */
            border-radius: 5px; /* Bordes redondeados */
            padding: 5px; /* Espaciado interno */
            box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.2); /* Sombra suave */
        }
        /* Estilos para la etiqueta de resultados */
        .results-label {
            font-size: 30px; /* Tamaño de fuente */
            color: #111; /* Color gris oscuro */
            font-weight: bold; /* Negrita */
        }
        .accept-button {
            font-size: 30px;  /* Tamaño de fuente grande para el botón */
            color: white;
            background: rgba(0, 85, 255, 0.5); /* Color de fondo azul con transparencia */
            border-radius: 15px;  /* Bordes redondeados */
            padding: 10px;  /* Espaciado interno */
            border: 2px solid rgba(0, 51, 187, 0.5); /* Borde sólido con transparencia */
        }
        .accept-button:hover {
            background: rgba(0, 44, 204, 0.5); /* Color de fondo azul al hacer hover */
        }
        .result-content {
            font-family: Sans;
            font-weight: bold;
            font-size: 14px;
        }
        """
        
        # Cargar el CSS en el proveedor
        css_provider.load_from_data(css.encode('utf-8'))

        # Obtener el contexto de la pantalla y aplicar el estilo CSS a la ventana
        screen = Gdk.Screen.get_default()
        style_context = Gtk.StyleContext()
        style_context.add_provider_for_screen(
            screen, css_provider, Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
        )

    def on_start_clicked(self, widget):
        """Acción cuando se presiona el botón "Start"."""
        if self.init:
           print("Conectando con " + self.host)
           self.client.connect((self.host,self.port))
           self.init = False
        if not self.is_beating:
            self.is_beating = True
            # Crear el área de dibujo dinámicamente
            if self.drawing_area is None:
                self.drawing_area = Gtk.DrawingArea()
                self.drawing_area.set_size_request(300, 300)  # Ajustar tamaño según necesidad
                self.main_box.pack_start(self.drawing_area, False, False, 20)
                self.drawing_area.connect("draw", self.on_draw)
                self.drawing_area.show()  # Mostrar el área de dibujo

            GLib.timeout_add(50, self.animate_pulse)  # Llamar a animate_pulse cada 50 ms
            threading.Thread(target=self.receive_and_inference).start()

    def animate_pulse(self):
        """Animar el efecto de pulso cambiando el radio de los círculos."""
        # Cambiar el radio
        if self.drawing_area is not None:
            self.current_radius += self.direction * 3  # Cambiar el tamaño de los círculos
            if self.current_radius >= self.max_radius or self.current_radius <= self.min_radius:
                self.direction *= -1  # Invertir dirección al alcanzar el límite
            self.drawing_area.queue_draw()  # Volver a dibujar el área de dibujo
        return self.is_beating  # Mantener el timeout activo

    def show_results(self):
        # Detener la animación
        self.is_beating = False
        self.main_box.remove(self.start_button)

        # Eliminar el área de dibujo y los círculos si existen
        if self.drawing_area is not None:
            self.main_box.remove(self.drawing_area)
            self.drawing_area = None

        # Crear un contenedor vertical para los resultados
        results_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=10)
        results_box.set_halign(Gtk.Align.CENTER)  # Centrar horizontalmente
        results_box.set_valign(Gtk.Align.CENTER)  # Centrar verticalmente

        # Etiqueta para mostrar el título de resultados
        result_label = Gtk.Label()
        result_label.set_text("Results of the Diagnosis")
        result_label.get_style_context().add_class("results-label")  # Aplicar clase de estilo (si se usa CSS)
        result_label.set_margin_top(20)
        result_label.set_margin_bottom(10)
        result_label.set_halign(Gtk.Align.CENTER)  # Centrar la etiqueta horizontalmente

        # Agregar la etiqueta de título al contenedor
        results_box.pack_start(result_label, False, False, 0)

        # Crear un frame para los resultados con desplazamiento
        result_frame = Gtk.Frame()
        result_frame.get_style_context().add_class("results-frame")
        result_frame.set_size_request(600, 150)  # Tamaño del frame
        result_frame.set_halign(Gtk.Align.CENTER)  # Centrar el cuadro horizontalmente

        # Crear una ScrolledWindow para el contenido del frame
        scrolled_window = Gtk.ScrolledWindow()
        scrolled_window.set_policy(Gtk.PolicyType.AUTOMATIC, Gtk.PolicyType.AUTOMATIC)  # Activar scroll automático
        scrolled_window.set_min_content_height(150)  # Altura mínima visible

        # Crear un Label dentro del frame para mostrar el diagnóstico
        result_content = Gtk.Label()

        # Construir el texto a mostrar en el Label
        text = ""
        for i in range(len(self.predictions)):
            text += f"Frame {i + 1} -- Accuracy: {self.predictions[i][0]:.2f} -- Prediction: {self.predictions[i][1]}\n"
        text += f"Mean Inference time: {self.avg_time:.4f} s"  # Mostrar tiempo promedio con dos decimales

        result_content.set_text(text)
        font_description = Pango.FontDescription("Sans Bold 14")  # Tipo de letra, estilo y tamaño
        result_content.override_font(font_description)
        
        result_content.set_line_wrap(True)  # Permitir ajuste de línea
        result_content.set_margin_top(20)
        result_content.set_margin_bottom(20)
        result_content.get_style_context().add_class("result-content")

        # Agregar el Label a la ScrolledWindow
        scrolled_window.add(result_content)

        # Agregar la ScrolledWindow al frame de resultados
        result_frame.add(scrolled_window)

        # Agregar el frame de resultados al contenedor vertical
        results_box.pack_start(result_frame, True, True, 0)  # Expandir verticalmente

        # Crear el botón "Aceptar" para cerrar la vista de resultados
        accept_button = Gtk.Button(label="Accept")
        accept_button.get_style_context().add_class("accept-button")  # Aplicar clase de estilo
        accept_button.set_size_request(150, 40)  # Ancho y alto del botón
        accept_button.set_halign(Gtk.Align.CENTER)  # Centrar el botón
        accept_button.connect("clicked", self.on_accept_clicked)

        # Agregar el botón "Aceptar" al contenedor
        results_box.pack_start(accept_button, False, False, 0)

        # Agregar el contenedor de resultados al main_box
        self.main_box.pack_start(results_box, True, True, 0)

        # Mostrar todos los widgets dentro del contenedor results_box
        results_box.show_all()

        return False  # Detener el timeout de GLib si se estaba utilizando en el contexto de una animación

    def on_accept_clicked(self, widget):
        """Acción cuando se presiona el botón "Aceptar"."""
        # Eliminar el cuadro de resultados y el botón "Aceptar"
        if self.main_box.get_children():  # Verifica que haya hijos en main_box
            for child in self.main_box.get_children():
                self.main_box.remove(child)

        # Recrear el botón "Start"
        self.start_button = Gtk.Button(label="START")
        self.start_button.set_size_request(300, 100)
        self.start_button.get_style_context().add_class("start-button")
        self.start_button.connect("clicked", self.on_start_clicked)
        self.main_box.pack_start(self.start_button, False, False, 20)
        self.start_button.show()  # Mostrar el botón "Start"

    def on_draw(self, widget, cr):
        """Dibujar los círculos en el área de dibujo."""
        width = widget.get_allocated_width()
        height = widget.get_allocated_height()

        for i in range(self.pulse_count):
            # Calcular el radio de cada círculo basado en el actual y la posición
            radius = self.current_radius + (i * 20)  # Espaciado entre círculos
            
            # Sombra del contorno (blanca y mínima)
            cr.set_source_rgba(1.0, 1.0, 1.0, 0.3)  # Color blanco con un poco de transparencia
            cr.arc(width / 2, height / 2, radius + 2, 0, 2 * math.pi)  # Círculo de sombra
            cr.fill_preserve()  # Llenar el círculo de sombra
            
            # Dibujar el contorno del círculo
            cr.set_source_rgba(0.0, 0.5, 1.0, 1.0)  # Color azul para el contorno
            cr.set_line_width(2)  # Ancho de línea del contorno
            cr.stroke()  # Dibujar el contorno

    def on_exit(self, widget, event):
        """Cerrar la aplicación."""
        Gtk.main_quit()

    def on_reboot(self, widget, event):
        """Reiniciar la aplicación (Placeholder)."""
        print("Reboot button pressed")

    def on_info(self, widget, event):
        """Mostrar información de la aplicación (Placeholder)."""
        print("Info button pressed")
    
    def receive_and_inference(self):
        tamaño_datos = self.client.recv(8)

        tamaño = int.from_bytes(tamaño_datos, 'big')
        data = b""
        while len(data) < tamaño:
            paquete = self.client.recv(4096)
            if not paquete:
                break
            data += paquete
        
        if data:
            audio_data = pickle.loads(data)
            print(f"Type of data received: {type(audio_data)}")
            print(f"Array shape: {audio_data.shape}")

            inference_times = []
            predictions = []
            for d in audio_data:
                print(d.shape)
                preprocess_start = time.time()
                mfccs = compute_mfccs(d, sample_rate=16000, n_mfcc=20, n_fft=2048, hop_length=512, window='hann',num_filters=128,htk=False)
                preprocess_finish = time.time()
                preprocess_time = preprocess_finish - preprocess_start
                print(f"Preproccess time: {preprocess_time} seconds")

                start_time = time.time()
                self.network.launch_inference(mfccs)
                stop_time = time.time()
                inference_time = stop_time - start_time
                inference_times.append(inference_time)
                print(f"Inference time: {inference_time} seconds")

                classification = self.network.get_results()
                print(classification)
                disease = np.argmax(classification)
                predictions.append((classification[disease], self.diseases[disease]))
                if classification[disease] < 0.8:
                    print("Unclear result, take another sample")
                else:
                    print(self.diseases[disease])
            print("==========================")
            self.avg_time = np.mean(np.array(inference_times))
            self.predictions = predictions
            GLib.idle_add(self.show_results)


if __name__ == '__main__':
    #os.system('sudo udhcpc -i wlan0')
    host = sys.argv[1]
    win = LungDiagnosisWindow(host)
    win.connect("destroy", Gtk.main_quit)
    win.show_all()
    Gtk.main()