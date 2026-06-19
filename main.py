import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import tkinter as tk
from tkinter import scrolledtext

class Arrow3D(FancyArrowPatch):
    """Класс для отрисовки 3D стрелок"""
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def do_3d_projection(self, renderer=None):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, self.axes.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)

class MagnetizationVector:
    """Класс для одного вектора намагниченности"""
    def __init__(self, initial_direction):
        self.initial_magnetization = initial_direction / np.linalg.norm(initial_direction)
        self.magnetization_vector = self.initial_magnetization.copy()
        self.previously_saturated = False
        self.initial_angle = None
        self.mag_at_0_1 = None
        self.field_dir_at_0_1 = None

class BCCVisualizer:
    def __init__(self):
        self.fig = plt.figure(figsize=(16, 8))
        
        # 3D график слева
        self.ax = self.fig.add_subplot(1, 2, 1, projection='3d')
        
        # 2D график справа
        self.ax_plot = self.fig.add_subplot(1, 2, 2)
        self.ax_plot.set_xlabel('Напряженность магнитного поля H, у.е.')
        self.ax_plot.set_ylabel('Намагниченность I, у.е.')
        self.ax_plot.set_title('Кривая намагничивания', fontsize=14, fontweight='bold')
        self.ax_plot.set_xlim(0, 1.5)
        self.ax_plot.set_ylim(0, 1.2)
        self.ax_plot.grid(True, alpha=0.3)
        self.ax_plot.axhline(y=0, color='black', linewidth=0.5)
        self.ax_plot.axvline(x=0.1, color='gray', linestyle='--', alpha=0.5)
        self.ax_plot.axvline(x=1.0, color='gray', linestyle='--', alpha=0.5)
        
        # Линия следа и текущая точка
        self.trace_line, = self.ax_plot.plot([], [], 'b-', linewidth=1.5, alpha=0.7)
        self.current_point, = self.ax_plot.plot([], [], 'ro', markersize=8)
        
        # История для следа
        self.history_H = []
        self.history_proj = []
        
        self.a = 1.0
        self.elev = 30
        self.azim = 45
        
        # Один вектор намагниченности (вдоль +Z)
        self.mag_vector = MagnetizationVector(np.array([0, 0, 1]))
        self.vector_start = np.array([0.5, 0.5, 0.5])
        
        self.field_strength = 0.0
        self.field_theta_deg = 30
        self.field_phi_deg = 60
        self.field_vector = np.array([0.0, 0.0, 0.0])
        self.update_field_vector()
        
        self.edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7)
        ]
        
        self.parallel_groups = [
            [(0, 1), (2, 3), (4, 5), (6, 7)],
            [(1, 2), (3, 0), (5, 6), (7, 4)],
            [(0, 4), (1, 5), (2, 6), (3, 7)]
        ]
        
        self.highlighted_lines = []
        self.mag_artist = None
        
        self.create_bcc_lattice()
        self.draw_magnetization_vector()
        self.field_vector_artist = None
        self.draw_field_vector()
        self.update_magnetization()
        self.update_plot()
        
        self.ax.view_init(elev=self.elev, azim=self.azim)
        self.create_field_controls()
        self.create_direction_buttons()
        self.create_info_button()
        self.setup_axes()
    
    def find_parallel_group(self):
        if self.field_strength == 0:
            return None
        
        field_dir = self.field_vector / np.linalg.norm(self.field_vector)
        vertices = self.atom_positions[:8]
        
        max_cos = -1
        best_group = None
        
        for group in self.parallel_groups:
            edge = group[0]
            edge_vector = vertices[edge[1]] - vertices[edge[0]]
            edge_dir = edge_vector / np.linalg.norm(edge_vector)
            cos_angle = abs(np.dot(field_dir, edge_dir))
            
            if cos_angle > max_cos:
                max_cos = cos_angle
                best_group = group
        
        return best_group
    
    def get_easy_direction(self):
        if self.field_strength == 0:
            return None
        
        vertices = self.atom_positions[:8]
        best_group = self.find_parallel_group()
        
        if best_group is None:
            return None
        
        edge = best_group[0]
        edge_vector = vertices[edge[1]] - vertices[edge[0]]
        edge_dir = edge_vector / np.linalg.norm(edge_vector)
        
        field_dir = self.field_vector / np.linalg.norm(self.field_vector)
        if np.dot(edge_dir, field_dir) < 0:
            edge_dir = -edge_dir
        
        return edge_dir
    
    def vector_to_angles(self, vec):
        """Перевод вектора в углы θ и φ (в градусах)"""
        vec = vec / np.linalg.norm(vec)
        theta = np.degrees(np.arccos(np.clip(vec[2], -1, 1)))
        phi = np.degrees(np.arctan2(vec[1], vec[0]))
        if phi < 0:
            phi += 360
        return theta, phi
    
    def set_field_direction(self, direction):
        """Установка направления поля по вектору"""
        theta, phi = self.vector_to_angles(direction)
        self.field_theta_deg = theta
        self.field_phi_deg = phi
        self.slider_theta.set_val(theta)
        self.slider_phi.set_val(phi)
        self.update_field_vector()
        self.draw_field_vector()
        # Сброс состояний
        self.mag_vector.initial_angle = None
        self.mag_vector.mag_at_0_1 = None
        self.update_magnetization()
        self.update_highlighted_edges()
    
    def update_magnetization(self):
        """Обновление вектора намагниченности"""
        mag = self.mag_vector
        easy_dir = self.get_easy_direction()
        
        if self.field_strength == 0:
            mag.magnetization_vector = mag.initial_magnetization.copy()
            self.draw_magnetization_vector()
            self.update_plot()
            return
        
        if easy_dir is None:
            mag.magnetization_vector = mag.initial_magnetization.copy()
            self.draw_magnetization_vector()
            self.update_plot()
            return
        
        field_dir = self.field_vector / np.linalg.norm(self.field_vector)
        
        # H <= 0.1: поворот к легкому направлению
        if self.field_strength <= 0.1:
            if self.field_strength >= 0.1 - 1e-10:
                mag.magnetization_vector = easy_dir.copy()
                mag.initial_magnetization = easy_dir.copy()
                mag.previously_saturated = True
                mag.mag_at_0_1 = easy_dir.copy()
                mag.field_dir_at_0_1 = field_dir.copy()
                
                cos_phi = np.dot(easy_dir, field_dir)
                mag.initial_angle = np.arccos(np.clip(cos_phi, -1, 1))
            else:
                if mag.previously_saturated:
                    if np.dot(easy_dir, mag.initial_magnetization) > 0.99:
                        mag.magnetization_vector = mag.initial_magnetization.copy()
                    else:
                        mag.previously_saturated = False
                        mag.initial_angle = None
                        mag.mag_at_0_1 = None
                        t = self.field_strength / 0.1
                        mag.magnetization_vector = self.slerp(mag.initial_magnetization, easy_dir, t)
                else:
                    t = self.field_strength / 0.1
                    mag.magnetization_vector = self.slerp(mag.initial_magnetization, easy_dir, t)
        
        # H > 0.1: угол линейно убывает до 0
        else:
            if mag.field_dir_at_0_1 is not None:
                if np.dot(field_dir, mag.field_dir_at_0_1) < 0.999:
                    mag.initial_angle = None
                    mag.mag_at_0_1 = None
            
            if mag.initial_angle is None:
                mag.mag_at_0_1 = mag.magnetization_vector.copy()
                mag.field_dir_at_0_1 = field_dir.copy()
                current_cos = np.dot(mag.magnetization_vector, field_dir)
                mag.initial_angle = np.arccos(np.clip(current_cos, -1, 1))
            
            if self.field_strength <= 1.0:
                target_angle = mag.initial_angle * (1 - (self.field_strength - 0.1) / 0.9)
            else:
                target_angle = 0.0
            
            target_cos = np.cos(target_angle)
            current_cos = np.dot(mag.magnetization_vector, field_dir)
            
            if abs(current_cos - target_cos) > 1e-6:
                if target_angle < 1e-10:
                    mag.magnetization_vector = field_dir.copy()
                else:
                    mag.magnetization_vector = self.rotate_vector_towards(
                        mag.magnetization_vector, field_dir, target_cos
                    )
            
            mag.initial_magnetization = mag.magnetization_vector.copy()
        
        self.draw_magnetization_vector()
        self.update_plot()
    
    def get_projection_on_field(self):
        """Проекция вектора намагниченности на направление поля"""
        if self.field_strength == 0:
            return 0
        field_dir = self.field_vector / np.linalg.norm(self.field_vector)
        return np.dot(self.mag_vector.magnetization_vector, field_dir)
    
    def update_plot(self):
        """Обновление графика со следом"""
        projection = self.get_projection_on_field()
        
        self.history_H.append(self.field_strength)
        self.history_proj.append(projection)
        
        self.trace_line.set_data(self.history_H, self.history_proj)
        self.current_point.set_data([self.field_strength], [projection])
        
        self.fig.canvas.draw_idle()
    
    def rotate_vector_towards(self, vec, target, target_cos):
        vec = vec / np.linalg.norm(vec)
        target = target / np.linalg.norm(target)
        
        current_cos = np.dot(vec, target)
        
        if abs(current_cos - target_cos) < 1e-10:
            return vec.copy()
        
        current_angle = np.arccos(np.clip(current_cos, -1, 1))
        target_angle = np.arccos(np.clip(target_cos, -1, 1))
        rotation_angle = current_angle - target_angle
        
        axis = np.cross(vec, target)
        axis_norm = np.linalg.norm(axis)
        
        if axis_norm < 1e-10:
            return target.copy() if target_cos > current_cos else vec.copy()
        
        axis = axis / axis_norm
        
        cos_a = np.cos(rotation_angle)
        sin_a = np.sin(rotation_angle)
        
        rotated = (vec * cos_a + 
                  np.cross(axis, vec) * sin_a + 
                  axis * np.dot(axis, vec) * (1 - cos_a))
        
        return rotated / np.linalg.norm(rotated)
    
    def slerp(self, v0, v1, t):
        cos_omega = np.dot(v0, v1)
        omega = np.arccos(np.clip(cos_omega, -1, 1))
        
        if omega < 1e-10:
            return v0.copy()
        
        sin_omega = np.sin(omega)
        return (np.sin((1-t)*omega) / sin_omega * v0 + 
                np.sin(t*omega) / sin_omega * v1)
    
    def draw_magnetization_vector(self):
        if self.mag_artist:
            self.mag_artist.remove()
            self.mag_artist = None
        
        vector_length = self.a * 0.8
        vector_end = self.vector_start + self.mag_vector.magnetization_vector * vector_length
        
        self.mag_artist = Arrow3D(
            [self.vector_start[0], vector_end[0]],
            [self.vector_start[1], vector_end[1]],
            [self.vector_start[2], vector_end[2]],
            mutation_scale=20, lw=3, arrowstyle="-|>", 
            color="red", alpha=0.9, label='Вектор намагниченности I'
        )
        self.ax.add_artist(self.mag_artist)
    
    def update_highlighted_edges(self):
        for line in self.highlighted_lines:
            line[0].remove()
        self.highlighted_lines = []
        
        best_group = self.find_parallel_group()
        
        if best_group is not None:
            vertices = self.atom_positions[:8]
            
            for edge in best_group:
                v1 = vertices[edge[0]]
                v2 = vertices[edge[1]]
                
                line = self.ax.plot3D(
                    [v1[0], v2[0]],
                    [v1[1], v2[1]],
                    [v1[2], v2[2]],
                    'yellow', linewidth=4, alpha=0.9
                )
                self.highlighted_lines.append(line)
        
        self.fig.canvas.draw_idle()
    
    def create_bcc_lattice(self):
        vertices = np.array([
            [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
            [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]
        ]) * self.a
        
        center = np.array([[0.5, 0.5, 0.5]]) * self.a
        
        self.atom_positions = np.vstack([vertices, center])
        self.draw_atoms()
        self.draw_edges()
        
    def draw_atoms(self):
        vertices = self.atom_positions[:8]
        center = self.atom_positions[8]
        
        self.vertices_scatter = self.ax.scatter(
            vertices[:, 0], vertices[:, 1], vertices[:, 2], 
            c='black', s=100, alpha=0.8
        )
        
        self.center_scatter = self.ax.scatter(
            center[0], center[1], center[2], 
            c='black', s=200, alpha=0.8
        )
    
    def draw_edges(self):
        vertices = self.atom_positions[:8]
        
        for edge in self.edges:
            self.ax.plot3D(
                [vertices[edge[0]][0], vertices[edge[1]][0]],
                [vertices[edge[0]][1], vertices[edge[1]][1]],
                [vertices[edge[0]][2], vertices[edge[1]][2]],
                'gray', linewidth=1.5, alpha=0.6
            )
        
        center = self.atom_positions[8]
        for vertex in vertices:
            self.ax.plot3D(
                [vertex[0], center[0]],
                [vertex[1], center[1]],
                [vertex[2], center[2]],
                'gray', linewidth=1, alpha=0.3, linestyle='--'
            )
    
    def update_field_vector(self):
        theta_rad = np.radians(self.field_theta_deg)
        phi_rad = np.radians(self.field_phi_deg)
        
        self.field_vector = self.field_strength * np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])
    
    def draw_field_vector(self):
        if self.field_vector_artist:
            self.field_vector_artist.remove()
            self.field_vector_artist = None
        
        if self.field_strength == 0:
            self.fig.canvas.draw_idle()
            return
        
        vector_end = self.vector_start + self.field_vector * self.a * 1.5
        
        self.field_vector_artist = Arrow3D(
            [self.vector_start[0], vector_end[0]],
            [self.vector_start[1], vector_end[1]],
            [self.vector_start[2], vector_end[2]],
            mutation_scale=20, lw=3, arrowstyle="-|>", 
            color="blue", alpha=0.9, label='Вектор напряженности внешнего магнитного поля H'
        )
        self.ax.add_artist(self.field_vector_artist)
        self.fig.canvas.draw_idle()
    
    def create_field_controls(self):
        self.fig.text(0.15, 0.11, 'Внешнее магнитное поле', fontsize=12, fontweight='bold')
        
        ax_strength = plt.axes([0.1, 0.05, 0.2, 0.03])
        self.slider_strength = Slider(ax_strength, 'H', 0.0, 1.5, valinit=self.field_strength)
        self.slider_strength.on_changed(self.update_field_strength)
        
        ax_theta = plt.axes([0.35, 0.05, 0.2, 0.03])
        self.slider_theta = Slider(ax_theta, 'θ°', 0, 180, valinit=self.field_theta_deg)
        self.slider_theta.on_changed(self.update_field_theta)
        
        ax_phi = plt.axes([0.6, 0.05, 0.2, 0.03])
        self.slider_phi = Slider(ax_phi, 'φ°', 0, 360, valinit=self.field_phi_deg)
        self.slider_phi.on_changed(self.update_field_phi)
    
    def create_direction_buttons(self):
        """Кнопки кристаллографических направлений"""
        ax_100 = plt.axes([0.1, 0.01, 0.08, 0.03])
        self.btn_100 = Button(ax_100, '<100>', color='lightblue')
        self.btn_100.on_clicked(lambda event: self.set_field_direction(np.array([1, 0, 0])))
        
        ax_110 = plt.axes([0.19, 0.01, 0.08, 0.03])
        self.btn_110 = Button(ax_110, '<110>', color='lightgreen')
        self.btn_110.on_clicked(lambda event: self.set_field_direction(np.array([1, 1, 0])))
        
        ax_111 = plt.axes([0.28, 0.01, 0.08, 0.03])
        self.btn_111 = Button(ax_111, '<111>', color='lightyellow')
        self.btn_111.on_clicked(lambda event: self.set_field_direction(np.array([1, 1, 1])))
        
        ax_random = plt.axes([0.37, 0.01, 0.08, 0.03])
        self.btn_random = Button(ax_random, 'Случайно', color='lightcoral')
        self.btn_random.on_clicked(self.random_direction)
    
    def create_info_button(self):
        """Кнопка информации"""
        ax_info = plt.axes([0.9, 0.01, 0.05, 0.03])
        self.btn_info = Button(ax_info, 'i', color='lightgray')
        self.btn_info.on_clicked(self.show_info)
    
    def show_info(self, event):
        """Отображение информационного окна"""
        info_window = tk.Tk()
        info_window.title("О программе")
        info_window.geometry("700x600")
        
        info_text = """1. Выберите направление H кнопками или задайте вручную
2. Увеличивайте H, наблюдая, как с этим меняется намагниченность ферромагнетика!

Данная визуализация на примере кристаллической решетки α-железа демонстрирует так называемую магнитокристаллическую анизотропию – явление, заключающееся в том, что «скорость» намагничивания ферромагнетика вдоль разных кристаллографических направлений разная.

В кристаллической решетке существуют так называемые направления легкого намагничивания – кристаллографические направления, вдоль которых намагничивание происходит с наименьшими затратами энергии. В случае α-Fe направлением легкого намагничивания является направление <100>, то есть ребро куба. Труднее всего намагничивание происходит вдоль направления <111>, то есть диагонали куба. Разница в «легкости» намагничивания вдоль разных направлений обусловлена разным магнитным взаимодействием электронов атомов, которые расположены вдоль этого направления.

При приложении слабого поля H вектор намагниченности I ферромагнетика практически без затрат энергии поворачивается в ближайшем направлении легкого намагничивания. В данной визуализации этому магнитному полю соответствует значение 0,1 у.е., а направления легкого намагничивания подсвечиваются желтым. При увеличении поля от 0,1 до 1 у.е. работа магнитного поля затрачивается на поворот вектора намагниченности I от направления легкого намагничивания до направления вектора самого магнитного поля H. Когда I становится сонаправлен с H, наступает магнитное насыщение (в нашем случае этому соответствует поле 1 у.е.). Дальнейшее увеличение поля не приводит к бОльшему намагничиванию ферромагнетика, и его намагниченность не увеличивается (чего не скажешь об индукции!).

Выбрав направление H вдоль разных кристаллографических направлений, а затем увеличивая H, можно увидеть, насколько по-разному происходит намагничивание ферромагнетика. Примерно так и выглядят реальные кривые намагничивания при измерении магнитных свойств образцов ферромагнитных сплавов.

Обращаю внимание, что ввиду технических ограничений и отсутствия заложенных в код физических условий, возможны волшебные скачки намагниченности, а размагничивание образца достигается только... перезапуском программы!"""
        
        text_widget = scrolledtext.ScrolledText(info_window, wrap=tk.WORD, width=80, height=35, font=("Arial", 10))
        text_widget.insert(tk.INSERT, info_text)
        text_widget.configure(state='disabled')
        text_widget.pack(padx=10, pady=10)
        
        info_window.mainloop()
    
    def random_direction(self, event):
        """Случайное направление поля"""
        theta = np.random.uniform(0, 180)
        phi = np.random.uniform(0, 360)
        theta_rad = np.radians(theta)
        phi_rad = np.radians(phi)
        direction = np.array([
            np.sin(theta_rad) * np.cos(phi_rad),
            np.sin(theta_rad) * np.sin(phi_rad),
            np.cos(theta_rad)
        ])
        self.set_field_direction(direction)
    
    def update_field_strength(self, val):
        self.field_strength = val
        self.update_field_vector()
        self.draw_field_vector()
        self.update_magnetization()
        self.update_highlighted_edges()
    
    def update_field_theta(self, val):
        self.field_theta_deg = val
        self.update_field_vector()
        self.draw_field_vector()
        self.mag_vector.initial_angle = None
        self.mag_vector.mag_at_0_1 = None
        self.update_magnetization()
        self.update_highlighted_edges()
    
    def update_field_phi(self, val):
        self.field_phi_deg = val
        self.update_field_vector()
        self.draw_field_vector()
        self.mag_vector.initial_angle = None
        self.mag_vector.mag_at_0_1 = None
        self.update_magnetization()
        self.update_highlighted_edges()
    
    def setup_axes(self):
        self.ax.set_xlabel('X')
        self.ax.set_ylabel('Y')
        self.ax.set_zlabel('Z')
        self.ax.set_xlim(-0.2, 1.2)
        self.ax.set_ylim(-0.2, 1.2)
        self.ax.set_zlim(-0.2, 1.2)
        self.ax.legend(loc='upper right')
    
    def show(self):
        plt.show()

if __name__ == "__main__":
    viz = BCCVisualizer()
    viz.show()