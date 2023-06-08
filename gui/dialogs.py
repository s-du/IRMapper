from PySide6 import QtCore, QtGui, QtWidgets
from PySide6 import QtWebEngineWidgets

import os
import numpy as np
import cv2
import traceback
import logging
import open3d as o3d
import win32gui, win32con
import sys
from PIL import Image
# import matplotlib.pyplot as plt

# from multiprocessing import Process
# import threading
# import http.server
# import socketserver

# custom packages
import resources as res
from gui import widgets as wid
from tools import thermal_tools as tt

# paths
ir_xml_path = res.find('other/cam_calib_m2t_opencv.xml')
rgb_xml_path = res.find('other/rgb_cam_calib_m2t_opencv.xml')

# basic logger functionality
log = logging.getLogger(__name__)
handler = logging.StreamHandler(stream=sys.stdout)
log.addHandler(handler)

def show_exception_box(log_msg):
    """Checks if a QApplication instance is available and shows a messagebox with the exception message.
    If unavailable (non-console application), log an additional notice.
    """
    if QtWidgets.QApplication.instance() is not None:
            errorbox = QtWidgets.QMessageBox()
            errorbox.setText("Oops. An unexpected error occured:\n{0}".format(log_msg))
            errorbox.exec_()
    else:
        log.debug("No QApplication instance available.")

class UncaughtHook(QtCore.QObject):
    _exception_caught = QtCore.Signal(object)

    def __init__(self, *args, **kwargs):
        super(UncaughtHook, self).__init__(*args, **kwargs)

        # this registers the exception_hook() function as hook with the Python interpreter
        sys.excepthook = self.exception_hook

        # connect signal to execute the message box function always on main thread
        self._exception_caught.connect(show_exception_box)

    def exception_hook(self, exc_type, exc_value, exc_traceback):
        """Function handling uncaught exceptions.
        It is triggered each time an uncaught exception occurs.
        """
        if issubclass(exc_type, KeyboardInterrupt):
            # ignore keyboard interrupt to support console applications
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
        else:
            exc_info = (exc_type, exc_value, exc_traceback)
            log_msg = '\n'.join([''.join(traceback.format_tb(exc_traceback)),
                                 '{0}: {1}'.format(exc_type.__name__, exc_value)])
            log.critical("Uncaught exception:\n {0}".format(log_msg), exc_info=exc_info)

            # trigger message box show
            self._exception_caught.emit(log_msg)

# create a global instance of our class to register the hook
qt_exception_hook = UncaughtHook()


class AboutDialog(QtWidgets.QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle('What is this app about?')
        self.setFixedSize(300, 300)
        self.layout = QtWidgets.QVBoxLayout()

        about_text = QtWidgets.QLabel('This app was made by Buildwise, to simplify the analysis of thermal images.'
                                      ' Start by loading a folder, '
                                      'then process images.')
        about_text.setWordWrap(True)

        logos1 = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(res.find('img/logo_buildwise2.png'))
        w = self.width()
        pixmap = pixmap.scaledToWidth(100, QtCore.Qt.SmoothTransformation)
        logos1.setPixmap(pixmap)

        logos2 = QtWidgets.QLabel()
        pixmap = QtGui.QPixmap(res.find('img/logo_pointify.png'))
        pixmap = pixmap.scaledToWidth(100, QtCore.Qt.SmoothTransformation)
        logos2.setPixmap(pixmap)

        self.layout.addWidget(about_text)
        self.layout.addWidget(logos1, alignment=QtCore.Qt.AlignCenter)
        self.layout.addWidget(logos2, alignment=QtCore.Qt.AlignCenter)

        self.setLayout(self.layout)


class DialogMeshPreviewOpenFree(QtWidgets.QDialog):
    """
    Dialog that allows to visualize thermal mesh with open3d, with a floating window
    """

    def __init__(self,  pcd_load, np_rgb, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'visu_open_window'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        self.pcd = pcd_load
        self.np_rgb = np_rgb
        self.np_ir_rgb = np.asarray(self.pcd.colors)
        print(self.np_ir_rgb)
        print(self.np_rgb)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=600, height=400)
        self.vis.add_geometry(self.pcd)
        self.opt = self.vis.get_render_option()
        self.ctr = self.vis.get_view_control()
        self.opt.point_size = 1
        self.rgb_mode = False

        self.id = win32gui.FindWindowEx(0, 0, None, "Open3D")
        #win32gui.ShowWindow(self.id, win32con.SW_MAXIMIZE) # maximize open3D window
        if self.id:
            hMenu = win32gui.GetSystemMenu(self.id, 0)
            if hMenu:
                win32gui.DeleteMenu(hMenu, win32con.SC_CLOSE, win32con.MF_BYCOMMAND)


        # add icons
        wid.add_icon(res.find('img/i_plus.png'), self.pushButton_ptplus)
        wid.add_icon(res.find('img/i_min.png'), self.pushButton_ptmin)
        wid.add_icon(res.find('img/i_palette.png'), self.pushButton_style)
        wid.add_icon(res.find('img/i_camera.png'), self.pushButton_captureview)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_vis)
        timer.start(1)

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.create_connections()

    def create_connections(self):
        self.pushButton_style.clicked.connect(self.change_color)
        self.pushButton_ptplus.clicked.connect(self.points_size_plus)
        self.pushButton_ptmin.clicked.connect(self.points_size_min)
        self.pushButton_captureview.clicked.connect(self.capture_view)

    def capture_view(self):
        # use included open3d function
        dest_path = QtWidgets.QFileDialog.getSaveFileName(self, 'Save File As', 'render.jpg', "Image File (*.jpg)")
        if dest_path[0]:
            self.vis.capture_screen_image(dest_path[0], do_render=True)


    def update_vis(self):
        # self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def points_size_plus(self):
        self.opt.point_size +=1
        self.update_vis()

    def points_size_min(self):
        if self.opt.point_size > 0:
            self.opt.point_size -= 1
        self.update_vis()

    def change_color(self):

        #self.opt.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
        # self.update_vis()
        if not self.rgb_mode:
            print('switch to RGB')
            self.pcd.colors = o3d.utility.Vector3dVector(self.np_rgb)
            self.vis.update_geometry(self.pcd)
            self.rgb_mode = True
        else:
            print('switch to IR')
            self.pcd.colors = o3d.utility.Vector3dVector(self.np_ir_rgb)
            self.vis.update_geometry(self.pcd)
            self.rgb_mode = False

        self.vis.poll_events()
        self.vis.update_renderer()

    def enum_windows(self):
        def callback(wnd, data):
            windows.append(wnd)

        windows = []
        win32gui.EnumWindows(callback, None)
        return windows


class DialogMeshPreviewOpen(QtWidgets.QDialog):
    """
    Dialog that allows to visualize thermal mesh with open3D, with integrated window
    """

    def __init__(self,  pcd_load, np_rgb, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'visu_'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        self.pcd = pcd_load
        self.np_rgb = np_rgb
        self.np_ir_rgb = np.asarray(self.pcd.colors)
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window()
        self.vis.add_geometry(self.pcd)
        self.opt = self.vis.get_render_option()
        self.rgb_mode = False

        self.id = win32gui.FindWindowEx(0, 0, None, "Open3D")
        widget = QtWidgets.QWidget()
        widget.setMouseTracking(True)
        self.setMouseTracking(True)
        self.window = QtGui.QWindow.fromWinId(self.id)
        self.windowcontainer = self.createWindowContainer(self.window, widget)

        self.horizontalLayout.addWidget(self.windowcontainer)

        # add icons
        wid.add_icon(res.find('img/i_plus.png'), self.pushButton_ptplus)
        wid.add_icon(res.find('img/i_min.png'), self.pushButton_ptmin)
        wid.add_icon(res.find('img/i_palette.png'), self.pushButton_style)

        timer = QtCore.QTimer(self)
        timer.timeout.connect(self.update_vis)
        timer.start(1)

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

        self.create_connections()
        self.on_top()


    def on_top(self):
        win32gui.BringWindowToTop(self.id)
        win32gui.SetFocus(self.id)

    def mouseMoveEvent(self, event):
        print('bah')
        win32gui.BringWindowToTop(self.id)
        win32gui.SetFocus(self.id)

    def create_connections(self):
        self.pushButton_style.clicked.connect(self.change_color)
        self.pushButton_ptplus.clicked.connect(self.points_size_plus)
        self.pushButton_ptmin.clicked.connect(self.points_size_min)

    def update_vis(self):
        # self.vis.update_geometry()
        self.vis.poll_events()
        self.vis.update_renderer()

    def points_size_plus(self):
        self.opt.point_size +=1
        self.update_vis()

    def points_size_min(self):
        if self.opt.point_size > 0:
            self.opt.point_size -= 1
        self.update_vis()

    def change_color(self):

        #self.opt.point_color_option = o3d.visualization.PointColorOption.ZCoordinate
        # self.update_vis()
        if not self.rgb_mode:
            self.pcd.colors = o3d.utility.Vector3dVector(self.np_rgb)
            self.vis.update_geometry(self.pcd)
            self.rgb_mode = True
        else:
            self.pcd.colors = o3d.utility.Vector3dVector(self.np_ir_rgb)
            self.vis.update_geometry(self.pcd)
            self.rgb_mode = False

        self.vis.poll_events()
        self.vis.update_renderer()

    def enum_windows(self):
        def callback(wnd, data):
            windows.append(wnd)

        windows = []
        win32gui.EnumWindows(callback, None)
        return windows

"""
class MyHandler(http.server.SimpleHTTPRequestHandler):
#Fix for pyinstaller (otherwise http.server causes crashes)
    def log_message(self, format, *args):
        return

class RunnerSignals(QtCore.QObject):
    progressed = QtCore.Signal(int)
    messaged = QtCore.Signal(str)
    finished = QtCore.Signal()

class DialogMeshPreviewPotree(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'visu_potree'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        # add webview
        self.webEngineView = QtWebEngineWidgets.QWebEngineView()
        self.horizontalLayout.addWidget(self.webEngineView)

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def http_launch(self, path):
        
        os.chdir(path)
        PORT = 8080
        Handler = MyHandler

        with socketserver.TCPServer(("", PORT), Handler) as self.httpd:
            print("serving at port", PORT)
            self.httpd.serve_forever()

    def set_folder_to_stream(self, folder):
        os.chdir(folder)

        # Start processing requests
        self.thread = threading.Thread(target=self.http_launch, args=(folder,))
        self.thread.start()
        print(f'streaming this folder: {folder}')

    def stop_server(self):
        self.httpd.shutdown()
        self.httpd.socket.close()
        self.thread.join()

    def closeEvent(self, event):
        print("X is clicked")
        self.stop_server()
"""


class DialogThParams(QtWidgets.QDialog):
    """
    Dialog that allows the user to choose advances thermography options
    """

    def __init__(self, param, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'dialog_options'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)
        self.lineEdit_em.setText(str(param['emissivity']))
        self.lineEdit_dist.setText(str(param['distance']))
        self.lineEdit_rh.setText(str(param['humidity']))
        self.lineEdit_temp.setText(str(param['reflection']))

        # define constraints on lineEdit

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)


class DialogPrepareImages(QtWidgets.QDialog):
    """
    Dialog that allows to process thermal images
    """

    def __init__(self, ir_load_folder, rgb_load_folder, parent=None):
        QtWidgets.QDialog.__init__(self)
        basepath = os.path.dirname(__file__)
        basename = 'preparepictures'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        # add image viewer
        self.viewer = wid.PhotoViewer(self)
        self.horizontalLayout_3.addWidget(self.viewer)

        self.preview_rgb = False

        self.ir_folder = ir_load_folder
        self.rgb_folder = rgb_load_folder
        # list thermal images
        self.ir_imgs = os.listdir(self.ir_folder)
        self.rgb_imgs = os.listdir(self.rgb_folder)
        self.n_imgs = len(self.ir_imgs)

        # comboboxes content
        self.out_of_lim = ['black', 'white', 'red']
        self.out_of_matp = ['k', 'w', 'r']
        self.img_post = ['none', 'smooth', 'sharpen', 'sharpen strong', 'edge (simple)', 'edge (from rgb)']
        self.colormap_list = ['Artic', 'Iron', 'Rainbow', 'Greys_r', 'Greys', 'plasma', 'inferno', 'coolwarm', 'jet',
                              'Spectral_r',
                              'cividis', 'viridis', 'gnuplot2']
        self.view_list = ['thermal normal', 'th. undistorted', 'RGB crop']

        # add content to comboboxes
        self.comboBox.addItems(self.colormap_list)
        self.comboBox.setCurrentIndex(0)
        self.comboBox_img.addItems(self.ir_imgs)

        self.comboBox_colors_low.addItems(self.out_of_lim)
        self.comboBox_colors_low.setCurrentIndex(0)
        self.comboBox_colors_high.addItems(self.out_of_lim)
        self.comboBox_colors_high.setCurrentIndex(1)
        self.comboBox_post.addItems(self.img_post)
        self.comboBox_view.addItems(self.view_list)
        self.advanced_options = False

        # add icons
        wid.add_icon(res.find('img/i_est.png'), self.pushButton_estimate)

        # create validator for qlineedit
        onlyInt = QtGui.QIntValidator()
        onlyInt.setRange(0, 999)
        self.lineEdit_colors.setValidator(onlyInt)
        self.n_colors = 256  # default number of colors
        self.lineEdit_colors.setText(str(256))

        # default thermal options:
        self.thermal_param = {'emissivity': 0.95, 'distance': 5, 'humidity': 50, 'reflection': 25}

        # create simple preview
        self.current_img = 0

        # choose first image for preview
        test_img = self.ir_imgs[self.current_img]
        self.test_img_path = os.path.join(self.ir_folder, test_img)

        # get drone model
        self.drone_model = tt.get_drone_model(self.test_img_path)

        # create temporary folder
        self.preview_folder = os.path.join(self.ir_folder, 'preview')
        if not os.path.exists(self.preview_folder):
            os.mkdir(self.preview_folder)

        # quickly compute temperature delta on first image
        self.tmin, self.tmax = self.compute_delta(self.test_img_path)
        self.lineEdit_min_temp.setText(str(round(self.tmin, 2)))
        self.lineEdit_max_temp.setText(str(round(self.tmax, 2)))

        self.update_img_preview()

        # connections
        self.create_connections()

        # button actions
        self.buttonBox.accepted.connect(self.accept)
        self.buttonBox.rejected.connect(self.reject)

    def create_connections(self):
        # Push buttons
        self.pushButton_estimate.clicked.connect(self.estimate_temp)
        self.pushButton_advanced.clicked.connect(self.define_options)
        self.pushButton_left.clicked.connect(lambda: self.update_img_to_preview('minus'))
        self.pushButton_right.clicked.connect(lambda: self.update_img_to_preview('plus'))
        # Dropdowns
        self.comboBox.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_colors_low.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_colors_high.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_post.currentIndexChanged.connect(self.update_img_preview)
        self.comboBox_img.currentIndexChanged.connect(lambda: self.update_img_to_preview('other'))
        self.comboBox_view.currentIndexChanged.connect(self.update_img_preview)
        # Line edits
        self.lineEdit_min_temp.editingFinished.connect(self.update_img_preview)
        self.lineEdit_max_temp.editingFinished.connect(self.update_img_preview)
        self.lineEdit_colors.editingFinished.connect(self.update_img_preview)

    def define_options(self):
        dialog = DialogThParams(self.thermal_param)
        dialog.setWindowTitle("Choose advanced thermal options")

        if dialog.exec_():
            try:
                self.advanced_options = True
                em = float(dialog.lineEdit_em.text())
                if em < 0.1 or em > 1:
                    raise ValueError
                else:
                    self.thermal_param['emissivity'] = em
                dist = float(dialog.lineEdit_dist.text())
                if dist < 1 or dist > 25:
                    raise ValueError
                else:
                    self.thermal_param['distance'] = dist
                rh = float(dialog.lineEdit_rh.text())
                if rh < 20 or rh > 100:
                    raise ValueError
                else:
                    self.thermal_param['humidity'] = rh
                refl_temp = float(dialog.lineEdit_temp.text())
                if refl_temp < -40 or refl_temp > 500:
                    raise ValueError
                else:
                    self.thermal_param['reflection'] = refl_temp

                self.update_img_preview()

            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Warning",
                                              "Oops! Some of the values are not valid!")
                self.define_options()

    def compute_delta(self, img_path):
        raw_out = img_path[:-4] + '.raw'
        tt.read_dji_image(img_path, raw_out, self.thermal_param)

        fd = open(raw_out, 'rb')
        rows = 512
        cols = 640
        f = np.fromfile(fd, dtype='<f4', count=rows * cols)
        im = f.reshape((rows, cols))
        fd.close()

        comp_tmin = np.amin(im)
        comp_tmax = np.amax(im)

        os.remove(raw_out)

        return comp_tmin, comp_tmax

    def estimate_temp(self):
        ref_pic_name = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file',
                                                             self.ir_folder, "Image files (*.jpg *.JPG *.gif)")
        img_path = ref_pic_name[0]
        if img_path != '':
            tmin, tmax = self.compute_delta(img_path)
            self.lineEdit_min_temp.setText(str(round(tmin, 2)))
            self.lineEdit_max_temp.setText(str(round(tmax, 2)))

        self.update_img_preview()

    def update_img_to_preview(self, direction):
        self.preview_rgb = False
        if direction == 'minus':
            self.current_img -= 1
            self.comboBox_img.setCurrentIndex(self.current_img)
        elif direction == 'plus':
            self.current_img += 1
            self.comboBox_img.setCurrentIndex(self.current_img)
        else:
            self.current_img = self.comboBox_img.currentIndex()

        test_img = self.ir_imgs[self.current_img]
        self.test_img_path = os.path.join(self.ir_folder, test_img)

        self.update_img_preview()

        # change buttons
        if self.current_img == self.n_imgs - 1:
            self.pushButton_right.setEnabled(False)
        else:
            self.pushButton_right.setEnabled(True)

        if self.current_img == 0:
            self.pushButton_left.setEnabled(False)
        else:
            self.pushButton_left.setEnabled(True)

    def update_img_preview(self):
        # fetch user choices
        v = self.comboBox_view.currentIndex()

        if v == 2:
            read_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.current_img])
            self.viewer.setPhoto(QtGui.QPixmap(read_path))

        else:
            # colormap
            i = self.comboBox.currentIndex()
            colormap = self.colormap_list[i]
            try:
                self.n_colors = int(self.lineEdit_colors.text())
            except:
                self.n_colors = 256

            #   temp limits
            try:
                tmin = float(self.lineEdit_min_temp.text())
                tmax = float(self.lineEdit_max_temp.text())

                if tmax > tmin:
                    self.tmin = tmin
                    self.tmax = tmax
                else:
                    raise ValueError

            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Warning",
                                              "Oops! A least one of the temperatures is not valid.  Try again...")
                self.lineEdit_min_temp.setText(str(round(self.tmin, 2)))
                self.lineEdit_max_temp.setText(str(round(self.tmax, 2)))

            #   out of limits color
            i = self.comboBox_colors_low.currentIndex()
            user_lim_col_low = self.out_of_matp[i]
            i = self.comboBox_colors_high.currentIndex()
            user_lim_col_high = self.out_of_matp[i]

            #   post process operation
            k = self.comboBox_post.currentIndex()
            post_process = self.img_post[k]

            print(self.preview_folder)
            dest_path = os.path.join(self.preview_folder, 'preview.JPG')

            read_path = os.path.join(self.rgb_folder, self.rgb_imgs[self.current_img])
            tt.process_one_th_picture(self.thermal_param, self.drone_model, self.test_img_path, dest_path,
                                      self.tmin, self.tmax, colormap, user_lim_col_high,
                                      user_lim_col_low, n_colors=self.n_colors, post_process=post_process,
                                      rgb_path=read_path)

            if v == 1:
                cv_img = cv2.imread(dest_path)

                if post_process != 'edge (from rgb)':
                    undis = tt.undis(cv_img, ir_xml_path)
                    cv2.imwrite(dest_path, undis)

                self.viewer.setPhoto(QtGui.QPixmap(dest_path))
            else:
                self.viewer.setPhoto(QtGui.QPixmap(dest_path))

            # change view if undistort post-process
            if post_process == 'edge (from rgb)':
                self.comboBox_view.setCurrentIndex(1)


class DialogMakeModel(QtWidgets.QDialog):
    def __init__(self, parent=None):
        QtWidgets.QDialog.__init__(self)

        basepath = os.path.dirname(__file__)
        basename = 'makemodel'
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        # Hide some options
        self.checkBox_limit_points.setVisible(False)
        self.label_5.setVisible(False)
        self.lineEdit_maxpoints.setVisible(False)
        self.groupBox_2.setVisible(False)

        self.options_shown = False
        self.do_limit_point = False
        self.do_limit_poly = False
        self.poly_limit = 0
        self.point_limit = 0

        self.quality_list = ['low', 'medium', 'high']
        self.comboBox_quality.addItems(self.quality_list)

        self.texture_list = ['1024', '2048', '4096', '8192']
        self.numb_text = 1
        self.comboBox_texture_size.addItems(self.texture_list)

        QtCore.QTimer.singleShot(0, self.resizeMe)

        # button actions
        self.pushButton_options.clicked.connect(self.toggle_options)
        self.checkBox_limit_points.toggled.connect(self.limit_points)
        self.checkBox_limit_poly.toggled.connect(self.limit_polys)
        self.checkBox_mesh.toggled.connect(self.mesh_options)

        self.buttonBox.accepted.connect(self.return_values)
        self.buttonBox.rejected.connect(self.reject)

    def resizeMe(self):
        self.resize(self.minimumSizeHint())

    def toggle_options(self):
        if not self.options_shown:
            self.checkBox_limit_points.setVisible(True)
            self.label_5.setVisible(True)
            self.lineEdit_maxpoints.setVisible(True)
            self.groupBox_2.setVisible(True)
            self.pushButton_options.setText('Hide advanced options')
            self.options_shown = True
        else:
            self.checkBox_limit_points.setVisible(False)
            self.label_5.setVisible(False)
            self.lineEdit_maxpoints.setVisible(False)
            self.groupBox_2.setVisible(False)
            self.pushButton_options.setText('Show advanced options')
            self.options_shown = False
            QtCore.QTimer.singleShot(0, self.resizeMe)

    def mesh_options(self):
        if self.checkBox_mesh.isChecked():
            self.checkBox_limit_poly.setEnabled(True)
            self.comboBox_texture_size.setEnabled(True)
            self.spinBox.setEnabled(True)
            self.checkBox_ortho.setEnabled(True)
            self.checkBox_pdf.setEnabled(True)
        else:
            self.checkBox_limit_poly.setEnabled(False)
            self.comboBox_texture_size.setEnabled(False)
            self.spinBox.setEnabled(False)
            self.checkBox_ortho.setEnabled(False)
            self.checkBox_pdf.setEnabled(False)
            self.checkBox_ortho.setChecked(True)
            self.checkBox_pdf.setChecked(True)

    def limit_points(self):
        if self.checkBox_limit_points.isChecked():
            self.lineEdit_maxpoints.setEnabled(True)
        else:
            self.lineEdit_maxpoints.setEnabled(False)

    def limit_polys(self):
        if self.checkBox_limit_poly.isChecked():
            self.lineEdit_maxpoly.setEnabled(True)
        else:
            self.lineEdit_maxpoly.setEnabled(False)

    def fill_comb(self, list):
        # fill comboboxes
        self.comboBox_thimg.addItems(list)

    def return_values(self):
        self.quality = self.quality_list[self.comboBox_quality.currentIndex()]
        self.do_mesh = self.checkBox_mesh.isChecked()
        self.ortho = self.checkBox_ortho.isChecked()
        self.pdf = self.checkBox_pdf.isChecked()
        self.img_set_index = self.comboBox_thimg.currentIndex()
        self.texture = self.texture_list[self.comboBox_texture_size.currentIndex()]
        self.numb_text = self.spinBox.value()
        self.do_limit_point = self.checkBox_limit_points.isChecked()
        self.do_limit_poly = self.checkBox_limit_poly.isChecked()
        self.point_limit = self.lineEdit_maxpoints.text()
        self.poly_limit = self.lineEdit_maxpoly.text()


