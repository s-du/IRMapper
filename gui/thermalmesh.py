""" MAIN THERMAL 3D APP """
import os.path
import open3d as o3d
import numpy as np
from pathlib import Path

# import Pyqt packages
from PySide6 import QtCore, QtGui, QtWidgets
#import PySide6.QtWebEngineWidgets, PySide6.QtWebEngineCore
#PySide6.QtWebEngineCore.QWebEngineSettings.WebGLEnabled = True

import json

# import custom packages
import resources as res
from gui import widgets as wid
from tools import thermal_tools as tt
from tools import agisoft_part
from gui import dialogs as dia

# parameters
APP_FOLDER = 'ThermalMesh'
ORIGIN_THERMAL_IMAGES_NAME = 'Original Thermal Images'
RGB_ORIGINAL_NAME = 'Original RGB Images'
RGB_CROPPED_NAME = 'Cropped RGB Images'
ORIGIN_TH_FOLDER = 'img_th_original'
RGB_CROPPED_FOLDER = 'img_rgb'
PROC_TH_FOLDER = 'img_th_processed'
PROC_3D = '3D files'


class ImageSet:
    def __init__(self):
        self.has_rgb = True
        self.nb_ir_imgs = 0
        self.processed_ir_img = []
        self.drone_model = ''


class ThermImage:
    def __init__(self):
        self.rois = []
        self.hot_spots = []


class ThreedDataset:
    def __init__(self):

        self.path = ''
        self.has_pc = False
        self.has_mesh = False
        self.pc_path = ''
        self.pc_folder = ''
        self.mesh_path = ''
        self.files = []
        self.desc = ''
        self.pcd_load = None
        self.np_rgb = None

    def toJSON(self):  # To develop further
        return json.dumps(self, default=lambda o: o.__dict__,
            sort_keys=True, indent=4)


class ThermalWindow(QtWidgets.QMainWindow):
    """
    Main Window class for the Pointify application.
    """

    def __init__(self, parent=None):
        """
        Function to initialize the class
        :param parent:
        """
        super(ThermalWindow, self).__init__(parent)

        # load the ui
        basepath = os.path.dirname(__file__)
        basename = 'thermalmesh2'  # new version, more modern design
        uifile = os.path.join(basepath, 'ui/%s.ui' % basename)
        wid.loadUi(uifile, self)

        # threading
        self.__pool = QtCore.QThreadPool()
        self.__pool.setMaxThreadCount(3)

        # Create model (for the tree structure)
        self.model = QtGui.QStandardItemModel()
        self.treeView_files.setModel(self.model)
        size = QtCore.QSize(50, 50)
        self.treeView_files.setIconSize(size)
        self.selmod = self.treeView_files.selectionModel()

        # add icons
        wid.add_icon(res.find('img/i_folder.png'), self.actionLoadFolder)
        wid.add_icon(res.find('img/i_project.png'), self.actionLoadProject)
        wid.add_icon(res.find('img/i_process.png'), self.actionProcessImages)
        wid.add_icon(res.find('img/i_3d.png'), self.actionGo3D)
        wid.add_icon(res.find('img/i_view.png'), self.actionViewMesh)

        # create json
        self.json_file = ''

        # create empty list of processes
        self.proc_list = []  # list of processed thermal image sets
        self.thermal_process_img_folders = []  # list of folders containing thermal image sets
        self.th_threed_folders = []  # list of folders containing thermal meshes
        self.current_dataset = None
        self.reconstruction_database = []  # storing all reconstruction datasets

        # create overview description
        self.general_overview = ''

        # define useful paths
        self.gui_folder = os.path.dirname(__file__)
        self.list_rgb_paths = ''
        self.list_ir_paths = ''
        self.main_folder = ''
        self.app_folder = ''
        self.original_th_img_folder = ''
        self.rgb_crop_img_folder = ''

        # Read Agisoft license (if defined)
        self.license_loc = res.find('other/license_location.txt') # TODO : use settings.json

        if not os.path.exists(self.license_loc):
            self.license_path = None
        else:
            file1 = open(self.license_loc, "r")
            self.license_path = file1.read()
            file1.close()

        if self.license_path is not None:
            os.environ['agisoft_LICENSE'] = str(Path(self.license_path))
            print('Here it is!!', os.environ['agisoft_LICENSE'])

        # initialize status
        self.update_progress(nb=100, text="Status: Choose images or project!")

        # create connections (signals)
        self.create_connections()

    # GUI Methods_____________________
    def create_connections(self):
        self.actionLoadFolder.triggered.connect(self.load_img_phase1)
        self.actionLoadProject.triggered.connect(self.load_project)
        self.actionProcessImages.triggered.connect(self.go_img_phase1)
        self.actionGo3D.triggered.connect(self.go_mesh_phase1)
        self.actionAbout.triggered.connect(self.show_info)
        self.actionAgisoft_license_path.triggered.connect(self.change_license)

        # self.actionViewMesh.triggered.connect(self.go_visu_potree)  Potree use is possible
        self.actionViewMesh.triggered.connect(self.go_visu)
        self.treeView_files.doubleClicked.connect(self.tree_doubleClicked)
        self.selmod.selectionChanged.connect(self.on_tree_change)

    def change_license(self):
        if self.license_path is not None:
            default_lic_folder, _ = os.path.split(self.license_path)
        else:
            default_lic_folder = r'C:\Program Files\Agisoft\Metashape Pro'

        fname = QtWidgets.QFileDialog.getOpenFileName(self, 'Open file', default_lic_folder, "License files (*.lic)")
        print(f'the following license file will be loaded {fname[0]}')
        self.license_path = fname[0]

        if fname[0] != '':
            os.environ['agisoft_LICENSE'] = str(Path(self.license_path))
            print('Here it is!!', os.environ['agisoft_LICENSE'])

            file1 = open(self.license_loc, "w")
            file1.write(fname[0])
            file1.close()

    def reset_project(self):
        # reset variables
        # create json
        self.json_file = ''

        # create empty list of processes
        self.proc_list = []  # list of processed thermal image sets
        self.thermal_process_img_folders = []  # list of folders containing thermal image sets
        self.th_threed_folders = []  # list of folders containing thermal meshes
        self.current_dataset = None
        self.reconstruction_database = []

        # define useful paths
        self.list_rgb_paths = ''
        self.list_ir_paths = ''
        self.main_folder = ''
        self.app_folder = ''
        self.original_th_img_folder = ''
        self.rgb_crop_img_folder = ''

        # resetoverview
        self.label_infos.setText = 'Overview:'
        self.general_overview = ''

        # reset actions
        self.actionProcessImages.setEnabled(False)
        self.actionGo3D.setEnabled(False)

        # reset tree_view
        self.model = QtGui.QStandardItemModel()
        self.treeView_files.setModel(self.model)
        size = QtCore.QSize(50, 50)
        self.treeView_files.setIconSize(size)
        self.selmod = self.treeView_files.selectionModel()

    def tree_doubleClicked(self):
        indexes = self.treeView_files.selectedIndexes()
        sel_item = self.model.itemFromIndex(indexes[0])
        print(indexes[0])

        if os.path.isfile(sel_item.text()):
            if sel_item.text().endswith('JPG'):
                os.startfile(sel_item.text())

    def write_json(self, dictionary):
        # Serializing json
        json_object = json.dumps(dictionary, indent=4)
        with open(self.json_file, "w") as f:
            f.write(json_object)

    def get_json(self):
        with open(self.json_file, 'r') as f:
            json_object = json.load(f)

        return json_object

    def show_info(self):
        dialog = dia.AboutDialog()
        if dialog.exec_():
            pass

    def update_progress(self, nb=None, text=''):
        self.label_status.setText(text)
        if nb is not None:
            self.progressBar.setProperty("value", nb)

            # hide progress bar when 100%
            if nb >= 100:
                self.progressBar.setVisible(False)
            elif self.progressBar.isHidden():
                self.progressBar.setVisible(True)

    def add_item_in_tree(self, parent, line):
        item = QtGui.QStandardItem(line)
        parent.appendRow(item)
        self.model.sort(0)

    def picture_dropped(self, l, cat_text):
        imageset_item = self.model.findItems(cat_text)
        for url in l:
            if os.path.exists(url):
                print(url)
                icon = QtGui.QIcon(url)
                pixmap = icon.pixmap(72, 72)
                icon = QtGui.QIcon(pixmap)
                item = QtGui.QStandardItem(url)
                item.setIcon(icon)
                item.setStatusTip(url)
                item.setEditable(False)
                imageset_item[0].appendRow(item)

    def on_tree_change(self):
        indexes = self.treeView_files.selectedIndexes()

        if indexes[0].parent().isValid():
            data = indexes[0].parent().data()
        else:
            data = indexes[0].data()

        print(data)

        if PROC_3D in data:
            self.actionViewMesh.setEnabled(True)
            for dataset in self.reconstruction_database:
                if dataset.desc == data:
                    self.current_dataset = dataset
        else:
            self.actionViewMesh.setEnabled(False)


    # thread methods

    # Workflow Methods_____________________
    def load_img_phase1(self):
        ok_load = True
        # warning message (new project)
        if self.list_rgb_paths != '':
            qm = QtWidgets.QMessageBox
            reply = qm.question(self, '', "Are you sure ? It will create a new project", qm.Yes | qm.No)

            if reply == qm.Yes:
                # reset all data
                self.reset_project()

            elif reply == qm.No:
                ok_load = False

        if ok_load:
            folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))

            # warning message (new project)
            if self.list_rgb_paths != '':
                qm = QtWidgets.QMessageBox
                reply = qm.question(self, '', "Are you sure ? It will create a new project", qm.Yes | qm.No)

                if reply == qm.Yes:
                    # reset all data
                    self.reset_project()

            # sort images
            if not folder == '':  # if user cancel selection, stop function
                # add 'original images' in tree view
                self.treeView_files.setEnabled(True)
                self.add_item_in_tree(self.model, ORIGIN_THERMAL_IMAGES_NAME)
                self.add_item_in_tree(self.model, RGB_ORIGINAL_NAME)
                self.add_item_in_tree(self.model, RGB_CROPPED_NAME)
                self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Files')

                self.main_folder = folder
                self.app_folder = os.path.join(folder, APP_FOLDER)

                # update json path
                self.json_file = os.path.join(self.app_folder, 'data.json')

                # create some subfolders for storing images
                self.original_th_img_folder = os.path.join(self.app_folder, ORIGIN_TH_FOLDER)
                self.rgb_crop_img_folder = os.path.join(self.app_folder, RGB_CROPPED_FOLDER)

                # if the subfolders do not exist, create them
                if not os.path.exists(self.app_folder):
                    os.mkdir(self.app_folder)
                if not os.path.exists(self.original_th_img_folder):
                    os.mkdir(self.original_th_img_folder)
                if not os.path.exists(self.rgb_crop_img_folder):
                    os.mkdir(self.rgb_crop_img_folder)

                # update status
                text_status = 'loading images...'
                self.update_progress(nb=0, text=text_status)
                self.list_rgb_paths, self.list_ir_paths = tt.list_th_rgb_images_from_res(self.main_folder)

                # get drone model
                self.drone_model = tt.get_drone_model(self.list_ir_paths[0])

                print(f'Drone model : {self.drone_model}')

                dictionary = {
                    "Drone model": self.drone_model,
                    "Number of image pairs": str(len(self.list_ir_paths)),
                    "rgb_paths": self.list_rgb_paths,
                    "ir_paths": self.list_ir_paths
                }
                self.write_json(dictionary)  # store original images paths in a JSON

                text_status = 'copying thermal images...'
                self.update_progress(nb=10, text=text_status)

                # duplicate thermal images
                tt.copy_list_dest(self.list_ir_paths, self.original_th_img_folder)
                # create folder for cropped/resized rgb

                text_status = 'creating rgb miniatures...'
                self.update_progress(nb=20, text=text_status)

                worker_1 = tt.RunnerMiniature(self.list_rgb_paths, self.drone_model, 20, self.rgb_crop_img_folder, 20, 100)
                worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
                worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

                self.__pool.start(worker_1)
                worker_1.signals.finished.connect(self.load_img_phase2)

    def load_img_phase2(self):
        # add items to treeview
        list_cropped_rgb = []
        for file in os.listdir(self.rgb_crop_img_folder):
            list_cropped_rgb.append(os.path.join(self.rgb_crop_img_folder, file))
        self.picture_dropped(self.list_ir_paths, ORIGIN_THERMAL_IMAGES_NAME)
        self.picture_dropped(self.list_rgb_paths, RGB_ORIGINAL_NAME)
        self.picture_dropped(list_cropped_rgb, RGB_CROPPED_NAME)

        # enable actions for user
        self.actionProcessImages.setEnabled(True)

        # update overview
        #   get json data
        data = self.get_json()
        self.general_overview = self.label_infos.text()
        for i, key in enumerate(data):
            if i<2: # print only the two first lines
                self.general_overview += f'\n {key} : {data[key]}'

        self.label_infos.setText(self.general_overview)

        # update status
        self.update_progress(nb=100, text="Status: You can now process thermal images!")

    def load_project(self):
        try:
            self.app_folder = str(QtWidgets.QFileDialog.getExistingDirectory(self, "Select Directory"))
            if self.app_folder != '':
                _, name_folder = os.path.split(self.app_folder)
                print(name_folder)
                if name_folder == APP_FOLDER:
                    # add image folders
                    self.original_th_img_folder = os.path.join(self.app_folder, ORIGIN_TH_FOLDER)
                    self.rgb_crop_img_folder = os.path.join(self.app_folder, RGB_CROPPED_FOLDER)

                    self.json_file = os.path.join(self.app_folder, 'data.json')

                    # add items to treeview
                    self.treeView_files.setEnabled(True)
                    text_status = 'Loading original thermal images...'
                    self.update_progress(nb=10, text=text_status)

                    self.add_item_in_tree(self.model, ORIGIN_THERMAL_IMAGES_NAME)

                    text_status = 'Loading original rgb images...'
                    self.update_progress(nb=20, text=text_status)

                    self.add_item_in_tree(self.model, RGB_ORIGINAL_NAME)

                    text_status = 'Loading cropped rgb images...'
                    self.update_progress(nb=30, text=text_status)

                    self.add_item_in_tree(self.model, RGB_CROPPED_NAME)
                    self.model.setHeaderData(0, QtCore.Qt.Horizontal, 'Files')

                    # get img paths
                    dict = self.get_json()
                    self.list_rgb_paths = dict['rgb_paths']
                    self.list_ir_paths = dict['ir_paths']

                    list_cropped_rgb = []
                    for file in os.listdir(self.rgb_crop_img_folder):
                        list_cropped_rgb.append(os.path.join(self.rgb_crop_img_folder, file))

                    text_status = 'Loading miniatures (thermal)...'
                    self.update_progress(nb=40, text=text_status)

                    self.picture_dropped(self.list_ir_paths, ORIGIN_THERMAL_IMAGES_NAME)

                    text_status = 'Loading miniatures (rgb)...'
                    self.update_progress(nb=60, text=text_status)

                    self.picture_dropped(self.list_rgb_paths, RGB_ORIGINAL_NAME)

                    text_status = 'Loading miniatures (cropped rgb)...'
                    self.update_progress(nb=80, text=text_status)
                    self.picture_dropped(list_cropped_rgb, RGB_CROPPED_NAME)

                    text_status = 'Loading 3D data...'
                    self.update_progress(nb=90, text=text_status)

                    # get drone model
                    self.drone_model = dict['Drone model']
                    print(f'Drone model : {self.drone_model}')

                    # check existing folder with past processing
                    list_files = os.listdir(self.app_folder)
                    for file in list_files:
                        file_path = os.path.join(self.app_folder, file)
                        if os.path.isdir(file_path) and PROC_TH_FOLDER in file:
                            self.proc_list.append(file)
                            self.thermal_process_img_folders.append(os.path.join(self.app_folder, file))
                            # add image items to tree view
                            tree_desc = f'Thermal processing {len(self.thermal_process_img_folders)}'
                            if 'edge' in file:
                                tree_desc += ' ! not compatible with 3d'
                            self.add_item_in_tree(self.model, tree_desc)

                            list_files = []
                            for file in os.listdir(self.thermal_process_img_folders[-1]):
                                list_files.append(os.path.join(self.thermal_process_img_folders[-1], file))
                            self.picture_dropped(list_files, tree_desc)

                        if os.path.isdir(file_path) and PROC_3D in file:
                            # create a new item of type 'reconstruction class'
                            new_threed_folder = ThreedDataset()
                            self.reconstruction_database.append(new_threed_folder)
                            number = len(self.reconstruction_database)

                            # get data
                            desc = file
                            th_mesh_folder = file_path

                            threed_files = tt.find_files_of_type(th_mesh_folder, types=['obj', 'tif', 'ply'])

                            # see if point cloud and/or mesh in folder
                            opt_pc = False
                            opt_mesh = False
                            for found_file in threed_files:
                                if 'ply' in found_file:
                                    opt_pc = True
                                if 'obj' in found_file:
                                    opt_mesh = True

                            self.reconstruction_database[-1].desc = desc
                            self.reconstruction_database[-1].path = th_mesh_folder
                            self.reconstruction_database[-1].has_pc = opt_pc
                            self.reconstruction_database[-1].has_mesh = opt_mesh
                            self.reconstruction_database[-1].files = threed_files

                            # add elements in treeview
                            self.add_item_in_tree(self.model, desc)
                            self.picture_dropped(threed_files, desc)

                            # point cloud specific operations
                            if opt_pc:
                                # read output point cloud
                                self.reconstruction_database[-1].pc_path = os.path.join(th_mesh_folder,
                                                                                        'thermal_point_cloud.ply')
                                self.reconstruction_database[-1].pcd_load = o3d.io.read_point_cloud(new_threed_folder.pc_path)

                                # read rgb point cloud
                                rgb_ply_path = os.path.join(th_mesh_folder, 'thermal_point_cloud_rgb.ply')
                                rgb_load = o3d.io.read_point_cloud(rgb_ply_path)
                                self.reconstruction_database[-1].np_rgb = np.asarray(rgb_load.colors)

                                pc_folder, _ = os.path.split(self.reconstruction_database[-1].pc_path)
                                self.reconstruction_database[-1].pc_folder = pc_folder

                    # enable action
                    self.actionProcessImages.setEnabled(True)
                    if self.proc_list:
                        for desc in self.proc_list:
                            if not 'edge' in desc:
                                self.actionGo3D.setEnabled(True)

                    # update overview
                    #   get json data
                    data = self.get_json()
                    self.general_overview = self.label_infos.text()
                    for i, key in enumerate(data):
                        if i < 2:  # print only the two first lines
                            self.general_overview += f'\n {key} : {data[key]}'

                    self.label_infos.setText(self.general_overview)

                    # update status
                    self.update_progress(nb=100, text="Status: You can now choose further processing!")
                else:
                    raise ValueError
        except:
            QtWidgets.QMessageBox.warning(self, "Warning",
                                          "It does not look as a ThermalMesh app folder !")

    def go_img_phase1(self):
        # launch corresponding dialog
        dialog = dia.DialogPrepareImages(self.original_th_img_folder, self.rgb_crop_img_folder)
        dialog.setWindowTitle("Choose parameters for thermal images processing")

        if dialog.exec_():
            qm = QtWidgets.QMessageBox
            reply = qm.question(self, '', "Are you sure to process all pictures with these parameters?", qm.Yes | qm.No)

            if reply == qm.Yes:
                try:
                    c = dialog.comboBox.currentIndex()
                    colormap = dialog.colormap_list[c]
                    n_colors = int(dialog.lineEdit_colors.text())

                    #   temp limits
                    tmin = float(dialog.lineEdit_min_temp.text())
                    tmax = float(dialog.lineEdit_max_temp.text())

                    #   out of limits color
                    i = dialog.comboBox_colors_low.currentIndex()
                    user_lim_col_low = dialog.out_of_matp[i]
                    j = dialog.comboBox_colors_high.currentIndex()
                    user_lim_col_high = dialog.out_of_matp[j]

                    #   post process operation
                    k = dialog.comboBox_post.currentIndex()
                    self.post_process = dialog.img_post[k]

                    # create subfolder
                    nummer = len(self.proc_list)
                    desc = f'{PROC_TH_FOLDER}_{colormap}_{str(round(tmin, 0))}_{str(round(tmax, 0))}_{self.post_process}_image set_{nummer}'

                    # append the processing to the list
                    self.proc_list.append(desc)
                    self.thermal_process_img_folders.append(os.path.join(self.app_folder, desc))
                    if not os.path.exists(self.thermal_process_img_folders[-1]):
                        os.mkdir(self.thermal_process_img_folders[-1])

                    # launch image processing
                    rgb_paths = ''
                    if self.post_process == 'edge (from rgb)':
                        rgb_paths = [os.path.join(self.rgb_crop_img_folder, file) for file in
                                     os.listdir(self.rgb_crop_img_folder)]
                    print(rgb_paths)

                    worker_1 = tt.RunnerDJI(self.list_ir_paths, self.thermal_process_img_folders[-1], self.drone_model,
                                            dialog.thermal_param, tmin, tmax, colormap,
                                            user_lim_col_high, user_lim_col_low,
                                            5, 100,
                                            n_colors=n_colors, post_process=self.post_process, rgb_paths = rgb_paths)
                    worker_1.signals.progressed.connect(lambda value: self.update_progress(value))
                    worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

                    self.__pool.start(worker_1)
                    worker_1.signals.finished.connect(self.go_img_phase2)

                except ValueError:
                    QtWidgets.QMessageBox.warning(self, "Warning",
                                                  "Oops! Something went terribly wrong!")
                    self.go_img_phase1()
            else:
                self.go_img_phase2()

    def go_img_phase2(self):

        # add new files in tree view
        tree_desc = f'Thermal processing {len(self.thermal_process_img_folders)}'
        if self.post_process == 'edge (from rgb)':
            tree_desc += ' ! not compatible with 3d'

        self.add_item_in_tree(self.model, tree_desc)

        list_files = []
        for file in os.listdir(self.thermal_process_img_folders[-1]):
            list_files.append(os.path.join(self.thermal_process_img_folders[-1], file))

        self.picture_dropped(list_files, tree_desc)

        # update buttons (allow 3D processing if compatible set)
        if self.proc_list:
            for desc in self.proc_list:
                if not 'edge' in desc:
                    self.actionGo3D.setEnabled(True)

        self.update_progress(nb=100, text="Status: You can now produce 3D data!")

    def go_mesh_phase1(self):
        """
        Function called from the 3D processing button; will create 3D files from image sets
        """
        dialog = dia.DialogMakeModel()
        dialog.setWindowTitle("Choose 3d reconstruction parameters")

        # fill list of possible image set
        possible_sets = []
        possible_sets_names = []
        for im_set in self.thermal_process_img_folders:
            if 'edge' not in im_set:
                _, desc_im_set = os.path.split(im_set)
                possible_sets_names.append(desc_im_set)
                possible_sets.append(im_set)

        dialog.fill_comb(possible_sets_names)

        if dialog.exec_():
            # get reconstruction options
            dialog.return_values()

            # thermal images (processed) to use
            i = dialog.img_set_index
            thermal_img_folder = possible_sets[i]
            ir_list = []
            for file in os.listdir(thermal_img_folder):
                if 'thermal' in file:
                    ir_list.append(os.path.join(thermal_img_folder, file))

            # get user values
            try:
                opt_quality = dialog.quality
                opt_mesh = dialog.do_mesh
                opt_texture = dialog.texture
                nb_text = dialog.numb_text
                opt_ortho = dialog.ortho
                opt_pdf = dialog.pdf
                if dialog.point_limit:
                    point_limit = int(dialog.point_limit)
                else:
                    point_limit = 0
                if dialog.poly_limit:
                    poly_limit = int(dialog.poly_limit)
                else:
                    poly_limit = 0
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Warning",
                                              "Oops! Something wrong with points or poly limits!")
                self.go_mesh_phase1()

            print(f'texture size: {opt_texture} \n'
                  f'quality: {opt_quality} \n'
                  f' output mesh: {opt_mesh} \n'
                  f' output ortho: {opt_ortho} \n'
                  f' output pdf: {opt_pdf} ')

            # new folders
            # create a new item of type 'reconstruction class'
            new_threed_folder = ThreedDataset()
            self.reconstruction_database.append(new_threed_folder)

            number = len(self.reconstruction_database)
            desc = f'{PROC_3D} (reconstruction {number})'
            th_mesh_folder = os.path.join(self.app_folder, desc)

            # do thermal processing
            if not os.path.exists(th_mesh_folder):
                os.mkdir(th_mesh_folder)

            print(f'files to process: {ir_list}, and {self.list_rgb_paths}')
            worker_1 = agisoft_part.RunnerAgisoft(self.drone_model, ir_list, self.list_rgb_paths, th_mesh_folder,
                                                  5, 100, do_precleaning=True, quality=opt_quality,
                                                  opt_make_mesh=opt_mesh, nb_text=nb_text, text_size=opt_texture,
                                                  opt_limit_points=point_limit, opt_limit_poly=poly_limit,
                                                  opt_make_ortho=opt_ortho, opt_make_pdf=opt_pdf)

            worker_1.signals.progressed.connect(lambda value: self.update_progress(nb=value))
            worker_1.signals.messaged.connect(lambda string: self.update_progress(text=string))

            self.__pool.start(worker_1)

            # TODO Add check if reconstruction is successful
            self.reconstruction_database[-1].desc = desc
            self.reconstruction_database[-1].path = th_mesh_folder
            self.reconstruction_database[-1].has_pc = True
            if opt_mesh:
                self.reconstruction_database[-1].has_mesh = True

            worker_1.signals.finished.connect(self.go_mesh_phase2)

    def go_mesh_phase2(self):
        threed_files = tt.find_files_of_type(self.reconstruction_database[-1].path, types=['obj', 'tif', 'ply'])
        self.reconstruction_database[-1].files = threed_files

        # add elements in treeview
        self.add_item_in_tree(self.model, self.reconstruction_database[-1].desc)
        self.picture_dropped(threed_files, self.reconstruction_database[-1].desc)

        # point cloud specific operations
        if self.reconstruction_database[-1].has_pc:
            # read thermal point cloud
            self.reconstruction_database[-1].pc_path = os.path.join(self.reconstruction_database[-1].path,
                                                                    'thermal_point_cloud.ply')
            pc_path = self.reconstruction_database[-1].pc_path
            pc_folder, _ = os.path.split(pc_path)
            self.reconstruction_database[-1].pc_folder = pc_folder

            self.reconstruction_database[-1].pcd_load = o3d.io.read_point_cloud(self.reconstruction_database[-1].pc_path)

            # read rgb point cloud
            rgb_ply_path = os.path.join(self.reconstruction_database[-1].path, 'thermal_point_cloud_rgb.ply')
            rgb_load = o3d.io.read_point_cloud(rgb_ply_path)
            self.reconstruction_database[-1].np_rgb = np.asarray(rgb_load.colors)

            """
            # create potree for thermal point cloud
            output_path = os.path.join(pc_folder, f'viewer')
            tt.potree_render_page(pc_path, output_path)

            # add rgb cloud as data in potree
            output_path2 = os.path.join(output_path, 'pointclouds')
            tt.potree_add_cloud(rgb_ply_path, output_path2)
            """

        self.update_progress(nb=100, text="Status: You can now visualise your data!")

    def go_visu_potree(self):
        dialog = dia.DialogMeshPreviewPotree(self)
        dialog.showMaximized()
        dialog.setWindowTitle("Analyse your cloud")

        to_stream = os.path.join(self.current_dataset.pc_folder, 'viewer')

        try:
            dialog.set_folder_to_stream(to_stream)
        except:
            QtWidgets.QMessageBox.warning(self, "Warning",
                                          "Error starting local server")

        url_to_stream = 'http://localhost:8080/webview.html'
        dialog.webEngineView.setUrl(QtCore.QUrl(url_to_stream))

        if dialog.exec_():
            dialog.stop_server()
            dialog.webEngineView.page().profile().clearHttpCache()


    def go_visu(self):
        dialog = dia.DialogMeshPreviewOpenFree(self.current_dataset.pcd_load, self.current_dataset.np_rgb)
        dialog.setWindowTitle("Visual options")

        # Hide dialog close button, to avoid conflicts with Open3D
        dialog.setWindowFlags((dialog.windowFlags() | QtCore.Qt.CustomizeWindowHint) & ~QtCore.Qt.WindowCloseButtonHint)

        if dialog.exec_():
            dialog.vis.destroy_window()
        else:
            dialog.vis.destroy_window()
