import sys
from PyQt5.QtWidgets import QApplication,QMessageBox
from gui import ImagePreviewApp
from image_processing import process_images

class MainApp(ImagePreviewApp):
    def on_process_button_click(self):
        if not self.current_folder:
            QMessageBox.warning(self, "警告", "请先选择图片文件夹！")
            return

        overlay_head = self.overlay_var.isChecked()
        # 调用图像处理逻辑
        process_images(self.current_folder, 'out', overlay_head)
        QMessageBox.information(self, "完成", "处理完成！")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())