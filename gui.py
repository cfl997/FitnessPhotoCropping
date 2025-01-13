from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget, QLabel, QPushButton, QFileDialog, QMessageBox, QCheckBox
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import os

class ImagePreviewApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("图片预览工具")
        self.setGeometry(100, 100, 800, 600)

        # 初始化全局变量
        self.current_folder = ""

        # 主窗口布局
        main_widget = QWidget()
        main_layout = QHBoxLayout(main_widget)

        # 左侧列表框
        self.listbox = QListWidget()
        self.listbox.setFixedWidth(100)
        self.listbox.itemSelectionChanged.connect(self.on_select)
        main_layout.addWidget(self.listbox)

        # 中间图片预览
        self.preview_label = QLabel("请选择图片")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setFixedSize(600, 800)
        main_layout.addWidget(self.preview_label)

        # 右侧控件
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)

        # 选择文件夹按钮
        self.select_button = QPushButton("选择图片文件夹")
        self.select_button.clicked.connect(self.select_folder)
        right_layout.addWidget(self.select_button)

        # 叠加头像复选框
        self.overlay_var = QCheckBox("叠加头像")
        right_layout.addWidget(self.overlay_var)

        # 开始处理按钮
        self.start_button = QPushButton("开始处理")
        self.start_button.clicked.connect(self.on_process_button_click)
        right_layout.addWidget(self.start_button)

        main_layout.addWidget(right_panel)

        self.setCentralWidget(main_widget)

    def select_folder(self):
        folder_selected = QFileDialog.getExistingDirectory(self, "选择图片文件夹")
        if folder_selected:
            image_files = [f for f in os.listdir(folder_selected) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            self.listbox.clear()
            for file in image_files:
                self.listbox.addItem(file)
            self.current_folder = folder_selected

    def on_select(self):
        selected_items = self.listbox.selectedItems()
        if not selected_items:
            return
        image_file = selected_items[0].text()
        image_path = os.path.join(self.current_folder, image_file)

        # 加载并显示图像
        try:
            pixmap = QPixmap(image_path)
            if not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
                self.preview_label.setPixmap(scaled_pixmap)
            else:
                self.preview_label.setText("无法打开图片")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"无法打开图片: {e}")

    def on_process_button_click(self):
        if not self.current_folder:
            QMessageBox.warning(self, "警告", "请先选择图片文件夹！")
            return

        overlay_head = self.overlay_var.isChecked()
        # 这里可以调用你现有的处理逻辑
        # process_images(self.current_folder, 'out', overlay_head)
        QMessageBox.information(self, "完成", "处理完成！")