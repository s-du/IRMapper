import os
from shutil import copyfile, copytree
import numpy as np
import subprocess
from pathlib import Path
from PIL import Image
from PIL import ImageFilter
from PySide6 import QtCore, QtGui, QtWidgets

from matplotlib import cm
from matplotlib import pyplot as plt
import matplotlib.colors as mcol
from blend_modes import dodge
import fileinput
import shutil
# from skimage.segmentation import felzenszwalb, slic --> Removed superpixel for the moment
# from skimage.color import label2rgb --> Removed superpixel for the moment
# from skimage.color import label2rgb

import cv2

# custom libraries
import resources as res

# paths
sdk_tool_path = Path(res.find('dji/dji_irp.exe'))
m2t_ir_xml_path = res.find('other/cam_calib_m2t_opencv.xml')
m2t_rgb_xml_path = res.find('other/rgb_cam_calib_m2t_opencv.xml')

m3t_ir_xml_path = res.find('other/cam_calib_m3t_opencv.xml')
m3t_rgb_xml_path = res.find('other/rgb_cam_calib_m3t_opencv.xml')

POTREE_CONV_PATH = res.find('PotreeConverter/PotreeConverter.exe')


class DroneModel():
    def __init__(self,name):
        self.name = name
        if name == 'MAVIC2-ENTERPRISE-ADVANCED':
            self.rgb_xml_path = m2t_rgb_xml_path
            self.ir_xml_path = m2t_ir_xml_path
            sample_rgb_path = res.find('img/M2EA_RGB.JPG')
            sample_ir_path = res.find('img/M2EA_IR.JPG')
            cv_rgb = cv_read_all_path(sample_rgb_path)
            cv_ir = cv_read_all_path(sample_ir_path)
            _, self.dim_undis_ir = undis(cv_ir, self.ir_xml_path)
            print(self.dim_undis_ir)
            _, self.dim_undis_rgb = undis(cv_rgb, self.rgb_xml_path)
            print(self.dim_undis_rgb)
            self.aspect_factor = (self.dim_undis_rgb[0] / self.dim_undis_rgb[1]) / (
                    self.dim_undis_ir[0] / self.dim_undis_ir[1])
            print(self.aspect_factor)
            self.extend = 0.332
            self.x_offset = 50
            self.y_offset = 35
        elif name == 'M3T':
            self.rgb_xml_path = m3t_rgb_xml_path
            self.ir_xml_path = m3t_ir_xml_path
            sample_rgb_path = res.find('img/M3T_RGB.JPG')
            sample_ir_path = res.find('img/M3T_IR.JPG')
            cv_rgb = cv_read_all_path(sample_rgb_path)
            cv_ir = cv_read_all_path(sample_ir_path)
            _, self.dim_undis_ir = undis(cv_ir, self.ir_xml_path)
            _, self.dim_undis_rgb = undis(cv_rgb, self.rgb_xml_path)
            self.aspect_factor = (self.dim_undis_rgb[0] / self.dim_undis_rgb[1]) / (
                    self.dim_undis_ir[0] / self.dim_undis_ir[1])
            self.extend = 0.3504
            self.x_offset = 49
            self.y_offset = 53

# long tasks runner classes
# test with runner
class RunnerSignals(QtCore.QObject):
    progressed = QtCore.Signal(int)
    messaged = QtCore.Signal(str)
    finished = QtCore.Signal()


class RunnerDJI(QtCore.QRunnable):
    def __init__(self, ir_paths, dest_folder, drone_model, param, tmin, tmax, colormap,
                 color_high, color_low, start, stop, n_colors=256, post_process='none', rgb_paths = '', export_tif = False):
        super().__init__()
        self.ir_paths = ir_paths
        self.dest_folder = dest_folder
        self.post_process = post_process
        self.drone_model = drone_model
        self.param = param
        self.tmin = tmin
        self.tmax = tmax
        self.colormap = colormap
        self.color_high = color_high
        self.color_low = color_low
        self.n_colors = n_colors
        self.rgb_paths = rgb_paths
        self.export_tif = export_tif
        self.start = start
        self.stop = stop

        self.signals = RunnerSignals()

    def run(self):
        # create raw outputs for each image
        nb_im = len(self.ir_paths)
        for i, img_path in enumerate(self.ir_paths):
            print(i)
            iter = i * (self.stop - self.start) / nb_im

            self.signals.progressed.emit(self.start + iter)
            self.signals.messaged.emit(f'Processing image {i}/{nb_im} with DJI SDK')

            if i < 9:
                prefix = '000'
            elif 9 < i < 99:
                prefix = '00'
            elif 99 < i < 999:
                prefix = '0'
            _, filename = os.path.split(str(img_path))
            dest_path = os.path.join(self.dest_folder, f'thermal_{prefix}{i}.JPG')

            if self.post_process == 'edge (from rgb)':
                process_one_th_picture(self.param, self.drone_model, img_path, dest_path, self.tmin, self.tmax,
                                       self.colormap, self.color_high, self.color_low, n_colors=self.n_colors,
                                       post_process=self.post_process,
                                       rgb_path=self.rgb_paths[i], export_tif = self.export_tif)
            else:
                process_one_th_picture(self.param, self.drone_model, img_path, dest_path, self.tmin, self.tmax,
                                       self.colormap, self.color_high, self. color_low, n_colors=self.n_colors,
                                       post_process=self.post_process, export_tif = self.export_tif)
            if i == len(self.ir_paths) - 1:
                if not self.export_tif:
                    legend_dest_path = os.path.join(self.dest_folder, 'plot_onlycbar_tight.png')
                    generate_legend(legend_dest_path, self.tmin, self.tmax, self.color_high, self.color_low, self.colormap, self.n_colors)

        self.signals.finished.emit()


class RunnerMiniature(QtCore.QRunnable):
    def __init__(self, list_rgb_paths, drone_model, scale_percent, dest_crop_folder, start, stop):
        super().__init__()
        self.signals = RunnerSignals()
        self.list_rgb_paths = list_rgb_paths
        self.scale_percent = scale_percent
        self.dest_crop_folder = dest_crop_folder
        self.start = start
        self.stop = stop
        self.drone_model = drone_model

    def run(self):
        nb_im = len(self.list_rgb_paths)

        ir_xml_path = self.drone_model.ir_xml_path
        rgb_xml_path = self.drone_model.rgb_xml_path

        for i, rgb_path in enumerate(self.list_rgb_paths):
            iter = i * (self.stop - self.start) / nb_im

            # update progress
            self.signals.progressed.emit(self.start + iter)
            self.signals.messaged.emit(f'Pre-processing image {i}/{nb_im}')
            cv_rgb_img = cv_read_all_path(rgb_path)

            und, _ = undis(cv_rgb_img, rgb_xml_path)
            crop = match_rgb_custom_parameters(und, self.drone_model)
            width = int(crop.shape[1] * self.scale_percent / 100)
            height = int(crop.shape[0] * self.scale_percent / 100)
            dim = (width, height)

            crop = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
            _, file = os.path.split(rgb_path)
            new_name = file[:-4] + 'crop.JPG'

            dest_path = os.path.join(self.dest_crop_folder, new_name)

            cv_write_all_path(crop, dest_path)

        self.signals.finished.emit()


# custom colormaps
def get_custom_cmaps(colormap_name):
    colors = [(25, 0, 150), (94, 243, 247), (100, 100, 100), (243, 116, 27), (251, 250, 208)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    artic_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=256)

    colors = [(0, 0, 0), (144, 15, 170), (230, 88, 65), (248, 205, 35), (255, 255, 255)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    ironbow_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=256)

    colors = [(8, 0, 75), (43, 80, 203), (119, 185, 31), (240, 205, 35), (245, 121, 47), (236, 64, 100),
              (240, 222, 203)]
    colors_scaled = [np.array(x).astype(np.float32) / 255 for x in colors]
    rainbow_cmap = mcol.LinearSegmentedColormap.from_list('my_colormap', colors_scaled, N=256)

    if colormap_name == 'Artic':
        out_colormap = artic_cmap
    elif colormap_name == 'Iron':
        out_colormap = ironbow_cmap
    elif colormap_name == 'Rainbow':
        out_colormap = rainbow_cmap

    return out_colormap


# PATH AND PREPARATION METHODS
def path_info(path):
    """
    Function that reads a path and outputs the foler, the complete filename and the filename without file extension
    @ parameters:
        path -- input path (string)
    """
    folder, file = os.path.split(path)
    extension = file[-4:]
    name = file[:-4]
    return folder, file, name, extension


def rename_from_exif(img_folder):
    pass


def find_files_of_type(folder, types=[]):
    files = [os.path.join(folder, file) for file in os.listdir(folder)]
    output = []

    for file in files:
        for type in types:
            if file.endswith(type):
                output.append(file)

    return output


def sort_image_method1(img_folder, dest_rgb_folder, dest_th_folder, string_to_search):
    """
    this function is adapted to sort all images from a folder, where those images are a mix between thermal and corresponding rgb
    """
    # Sorting images in new folders

    count = 0

    for file in os.listdir(img_folder):
        if count < 9:
            prefix = '000'
        elif 9 < count < 99:
            prefix = '00'
        elif 99 < count < 999:
            prefix = '000'
        if file.endswith('.jpg') or file.endswith('.JPG'):
            if string_to_search in str(file):
                new_file = 'image_' + prefix + str(count) + '.jpg'
                copyfile(os.path.join(img_folder, file), os.path.join(dest_th_folder, new_file))
                count += 1
            else:
                if count == 0:
                    count += 1
                new_file = 'image_' + prefix + str(count) + '.jpg'
                copyfile(os.path.join(img_folder, file), os.path.join(dest_rgb_folder, new_file))


def list_th_rgb_images_from_exif(img_folder):
    list_rgb_paths = []
    list_ir_paths = []
    for file in os.listdir(img_folder):

        path = os.path.join(img_folder, file)
        print(path)
        if file.endswith('.jpg') or file.endswith('.JPG'):
            res = get_resolution(path)
            if res == 640:
                list_ir_paths.append(path)
            else:
                list_rgb_paths.append(path)

    return list_rgb_paths, list_ir_paths


def list_th_rgb_images_from_res(img_folder):
    list_rgb_paths = []
    list_ir_paths = []
    for file in os.listdir(img_folder):
        path = os.path.join(img_folder, file)
        print(path)
        if file.endswith('.jpg') or file.endswith('.JPG'):
            im = Image.open(path)
            w, h = im.size

            if w == 640:
                list_ir_paths.append(path)
            else:
                list_rgb_paths.append(path)

        list_ir_paths.sort()
        list_rgb_paths.sort()

    return list_rgb_paths, list_ir_paths


def list_th_rgb_images(img_folder, string_to_search):
    list_rgb_paths = []
    list_ir_paths = []
    for file in os.listdir(img_folder):
        path = os.path.join(img_folder, file)
        if file.endswith('.jpg') or file.endswith('.JPG'):
            if string_to_search in str(file):
                path = os.path.join(img_folder, file)
                list_ir_paths.append(path)
            else:
                list_rgb_paths.append(path)

    return list_rgb_paths, list_ir_paths


def copy_list_dest(list_paths, dest_folder):
    for path in list_paths:
        _, file = os.path.split(path)
        copyfile(path, os.path.join(dest_folder, file))


def create_undis_folder(list_ir_paths, drone_model, dest_und_folder):
    ir_xml_path = drone_model.ir_xml_path

    for ir_path in list_ir_paths:
        cv_ir_img = cv_read_all_path(ir_path)
        cv_und, _ = undis(cv_ir_img, ir_xml_path)
        _, file = os.path.split(ir_path)
        new_name = file[:-4] + 'undis.JPG'
        dest_path = os.path.join(dest_und_folder, new_name)
        cv_write_all_path(cv_und, dest_path)


def create_rgb_crop_folder(list_rgb_paths, drone_model, scale_percent, dest_crop_folder, progressbar, start, stop):
    nb_im = len(list_rgb_paths)
    print(list_rgb_paths)
    for i, rgb_path in enumerate(list_rgb_paths):
        iter = i * (stop - start) / nb_im
        progressbar.setProperty("value", start + iter)
        cv_rgb_img = cv_read_all_path(rgb_path)

        rgb_xml_path = drone_model.rgb_xml_path

        und, _ = undis(cv_rgb_img, rgb_xml_path)
        crop = match_rgb_custom_parameters(und, drone_model)
        width = int(crop.shape[1] * scale_percent / 100)
        height = int(crop.shape[0] * scale_percent / 100)
        dim = (width, height)

        crop = cv2.resize(crop, dim, interpolation=cv2.INTER_AREA)
        _, file = os.path.split(rgb_path)
        new_name = file[:-4] + 'crop.JPG'

        dest_path = os.path.join(dest_crop_folder, new_name)
        cv2.imwrite(dest_path, crop)

# EXIF Readings
def print_exif(img_path):
    img = Image.open(img_path)
    infos = img.getexif()
    print(infos)


def get_drone_model(img_path):
    img = Image.open(img_path)
    infos = img.getexif()
    model = infos[272]
    return model


def get_resolution(img_path):
    img = Image.open(img_path)
    infos = img.getexif()
    res = infos[256]
    return res


def cv_read_all_path(path):
    """
    Allows reading image from any kind of unicode character (useful for french accents, for example)
    """
    img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
    return img

def cv_write_all_path(img, path):
    is_success, im_buf_arr = cv2.imencode(".JPG", img)
    im_buf_arr.tofile(path)


# LENS RELATED METHODS (GENERAL)
def undis(cv_img, xml_path):
    def read_matrices(xml_path):
        cv_file = cv2.FileStorage(xml_path, cv2.FILE_STORAGE_READ)
        K = cv_file.getNode("Camera_Matrix").mat()
        d = cv_file.getNode("Distortion_Coefficients").mat()
        cv_file.release()

        return K, d

    h, w = cv_img.shape[:2]
    K, d = read_matrices(xml_path)
    newcam, roi = cv2.getOptimalNewCameraMatrix(K, d, (w, h), 1, (w, h))
    dest = cv2.undistort(cv_img, K, d, None, newcam)
    x, y, w, h = roi
    dest = dest[y:y + h, x:x + w]

    height, width, channels = dest.shape
    dim = (width, height)

    return dest, dim


# LENS RELATED METHODS (DRONE SPECIFIC)
def match_rgb_custom_parameters(cv_img, drone_model, resized=False):
    print(cv_img)
    h2, w2 = cv_img.shape[:2]
    new_h = h2 * drone_model.aspect_factor
    ret_x = int(drone_model.extend * w2)
    ret_y = int(drone_model.extend * new_h)
    rgb_dest = cv_img[int(h2 / 2 + drone_model.y_offset) - ret_y:int(h2 / 2 + drone_model.y_offset) + ret_y,
               int(w2 / 2 + drone_model.x_offset) - ret_x:int(w2 / 2 + drone_model.x_offset) + ret_x]

    if resized:
        rgb_dest = cv2.resize(rgb_dest, drone_model.dim_undis_ir, interpolation=cv2.INTER_AREA)

    return rgb_dest


def add_lines_from_rgb(cv_ir_img, cv_match_rgb_img, drone_model, dest_path, exif=None):
    img_gray = cv2.cvtColor(cv_match_rgb_img, cv2.COLOR_BGR2GRAY)

    # Blur the image for better edge detection
    img_blur = cv2.GaussianBlur(img_gray, (3, 3), 0)

    scale = 1
    delta = 0
    ddepth = cv2.CV_16S
    grad_x = cv2.Sobel(img_blur, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)
    grad_y = cv2.Sobel(img_blur, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv2.BORDER_DEFAULT)

    abs_grad_x = cv2.convertScaleAbs(grad_x)
    abs_grad_y = cv2.convertScaleAbs(grad_y)

    edges = cv2.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

    pil_edges = Image.fromarray(edges)
    pil_edges = pil_edges.convert('RGB')
    pil_edges_rgba = pil_edges.convert('RGBA')
    foreground = np.array(pil_edges_rgba)

    # resize
    dim = drone_model.dim_undis_ir
    foreground = cv2.resize(foreground, dim, interpolation=cv2.INTER_AREA)
    foreground_float = foreground.astype(float)  # Inputs to blend_modes need to be floats.

    cv_ir_img = cv2.cvtColor(cv_ir_img, cv2.COLOR_BGR2RGB)
    ir_img = Image.fromarray(cv_ir_img)
    ir_img = ir_img.convert('RGBA')
    background = np.array(ir_img)
    background_float = background.astype(float)

    blended = dodge(background_float, foreground_float, 0.7)

    blended_img = np.uint8(blended)
    blended_img_raw = Image.fromarray(blended_img)
    blended_img_raw = blended_img_raw.convert('RGB')

    if exif is None:
        blended_img_raw.save(dest_path)
    else:
        blended_img_raw.save(dest_path, exif=exif)


# 3D PROCESSING
def potree_add_cloud(cloud_path, pointclouds_potree_folder):
    folder, file, name, _ = path_info(cloud_path)

    output_folder = os.path.join(pointclouds_potree_folder, 'rgb')

    p = subprocess.run(
        [POTREE_CONV_PATH, str(Path(cloud_path)), '-o', str(Path(output_folder))],
        capture_output=True,
        shell=True
    )

    print("the commandline is {}".format(p.args))
    print(p.stdout)


def potree_render_page(cloud_path, output_path):
    """
    Function that renders a point cloud into Potree
    @ parameters:
        cloud_path -- path to the point cloud to be rendered (string)
    @ returns:
        potree_html_file -- path to the newly created html_file
    """

    folder, file, name, _ = path_info(cloud_path)

    p = subprocess.run(
        [POTREE_CONV_PATH, str(Path(cloud_path)), '-o', str(Path(output_path)), '--generate-page', 'webview'],
        capture_output=True,
        shell=True
    )

    print("the commandline is {}".format(p.args))
    print(p.stdout)

    # replace html, js and css
    src = res.find('other/for_potree/webview.html')
    dst = output_path + '/webview.html'
    shutil.copyfile(src, dst)
    src = res.find('other/for_potree/potree.css')
    dst = output_path + '/libs/potree/potree.css'
    shutil.copyfile(src, dst)
    src = res.find('other/for_potree/potree.js')
    dst = output_path + '/libs/potree/potree.js'
    shutil.copyfile(src, dst)
    src = res.find('other/for_potree/sidebar.html')
    dst = output_path + '/libs/potree/sidebar.html'
    shutil.copyfile(src, dst)


def potree_modify_html(potree_html_path):
    """
        Function that modifies the html of a potree page
        @ parameters:
            html_path -- path to the potree html file
        """

    with fileinput.FileInput(potree_html_path, inplace=True) as file:
        for line in file:
            text_to_search = 'viewer.setEDLEnabled(false);'
            replacement_text = 'viewer.setEDLEnabled(true);'
            print(line.replace(text_to_search, replacement_text), end='')

    with fileinput.FileInput(potree_html_path, inplace=True) as file:
        for line in file:
            text_to_search = 'viewer.toggleSidebar();'
            replacement_text = 'viewer.setTopView();'
            print(line.replace(text_to_search, replacement_text), end='')

    with fileinput.FileInput(potree_html_path, inplace=True) as file:
        for line in file:
            text_to_search = 'viewer.setBackground("gradient")'
            replacement_text = 'viewer.setBackground("white")'
            print(line.replace(text_to_search, replacement_text), end='')




# THERMAL PROCESSING
def read_dji_image(img_in, raw_out, param={'emissivity': 0.95, 'distance': 5, 'humidity': 50, 'reflection': 25}):
    dist = param['distance']
    rh = param['humidity']
    refl_temp = param['reflection']
    em = param['emissivity']

    subprocess.run(
        [str(sdk_tool_path), "-s", f"{img_in}", "-a", "measure", "-o", f"{raw_out}", "--measurefmt",
         "float32", "--distance", f"{dist}", "--humidity", f"{rh}", "--reflection", f"{refl_temp}",
         "--emissivity", f"{em}"],
        universal_newlines=True,
        #stdout=subprocess.DEVNULL,
        #stderr=subprocess.STDOUT,
        #shell=True
    )

    image = Image.open(img_in)
    exif = image.info['exif']

    return exif


def process_one_th_picture(param, drone_model, ir_img_path, dest_path, tmin, tmax, colormap, color_high,
                           color_low, n_colors=256, post_process='none', rgb_path='', export_tif = False):
    _, filename = os.path.split(str(ir_img_path))
    new_raw_path = Path(str(ir_img_path)[:-4] + '.raw')

    exif = read_dji_image(str(ir_img_path), str(new_raw_path), param=param)


    # read raw dji output
    fd = open(new_raw_path, 'rb')
    rows = 512
    cols = 640
    f = np.fromfile(fd, dtype='<f4', count=rows * cols)
    im = f.reshape((rows, cols))  # notice row, column format
    fd.close()

    if export_tif:
        dest_path = dest_path[:-4] + '.tiff'
        img_thermal = Image.fromarray(im)
        img_thermal.save(dest_path, exif=exif)

    else:

        # compute new normalized temperature
        thermal_normalized = (im - tmin) / (tmax - tmin)

        # get colormap
        if colormap == 'Artic' or colormap == 'Iron' or colormap == 'Rainbow':
            custom_cmap = get_custom_cmaps(colormap)
        else:
            custom_cmap = cm.get_cmap(colormap, n_colors)

        custom_cmap.set_over(color_high)
        custom_cmap.set_under(color_low)

        thermal_cmap = custom_cmap(thermal_normalized)
        thermal_cmap = np.uint8(thermal_cmap * 255)

        img_thermal = Image.fromarray(thermal_cmap[:, :, [0, 1, 2]])

        if post_process == 'none':
            img_thermal.save(dest_path, exif=exif)
        elif post_process == 'sharpen':
            img_th_sharpened = img_thermal.filter(ImageFilter.SHARPEN)
            img_th_sharpened.save(dest_path, exif=exif)
        elif post_process == 'sharpen strong':
            img_th_sharpened = img_thermal.filter(ImageFilter.SHARPEN)
            img_th_sharpened2 = img_th_sharpened.filter(ImageFilter.SHARPEN)
            img_th_sharpened2.save(dest_path, exif=exif)
        elif post_process == 'edge (simple)':
            img_th_smooth = img_thermal.filter(ImageFilter.SMOOTH)
            img_th_findedge = img_th_smooth.filter(ImageFilter.Kernel((3, 3), (-1, -1, -1, -1, 8,
                                                                               -1, -1, -1, -1), 1, 0))
            img_th_findedge = img_th_findedge.convert('RGBA')
            img_thermal = img_thermal.convert('RGBA')
            foreground = np.array(img_th_findedge)  # Inputs to blend_modes need to be numpy arrays.

            foreground_float = foreground.astype(float)  # Inputs to blend_modes need to be floats.
            background = np.array(img_thermal)
            background_float = background.astype(float)
            print(background_float.shape)
            print(foreground_float.shape)
            blended = dodge(background_float, foreground_float, 0.5)

            blended_img = np.uint8(blended)
            blended_img_raw = Image.fromarray(blended_img)
            blended_img_raw = blended_img_raw.convert('RGB')
            blended_img_raw.save(dest_path, exif=exif)
        elif post_process == 'smooth':
            img_th_smooth = img_thermal.filter(ImageFilter.SMOOTH)
            img_th_smooth.save(dest_path, exif=exif)
        elif post_process == 'edge (from rgb)':
            img_thermal.save(dest_path)
            cv_ir_img = cv_read_all_path(dest_path)

            ir_xml_path = drone_model.ir_xml_path

            cv_ir_img, _ = undis(cv_ir_img, ir_xml_path)
            cv_match_rgb_img = cv_read_all_path(rgb_path)
            add_lines_from_rgb(cv_ir_img, cv_match_rgb_img, drone_model, dest_path, exif=exif)

        # elif post_process == 'superpixel':
        #    img_th_fz = superpixel(img_thermal)
        #    img_th_fz.save(thermal_filename)

    # remove raw file
    os.remove(new_raw_path)


def generate_legend(legend_dest_path, tmin, tmax, color_high, color_low, colormap, n_colors):
    fig, ax = plt.subplots()
    data = np.clip(np.random.randn(10, 10) * 100, tmin, tmax)
    print(data)

    if colormap == 'Artic' or colormap == 'Iron' or colormap == 'Rainbow':
        custom_cmap = get_custom_cmaps(colormap)
    else:
        custom_cmap = cm.get_cmap(colormap, n_colors)

    custom_cmap.set_over(color_high)
    custom_cmap.set_under(color_low)

    cax = ax.imshow(data, cmap=custom_cmap)
    # Add colorbar, make sure to specify tick locations to match desired ticklabels
    if n_colors > 12:
        n_colors = 12
    ticks = np.linspace(tmin, tmax, n_colors + 1, endpoint=True)
    fig.colorbar(cax, ticks=ticks, extend='both')
    ax.remove()

    plt.savefig(legend_dest_path, bbox_inches='tight')


def process_all_th_pictures(param, drone_model, ir_paths, dest_folder, tmin, tmax, colormap, color_high, color_low,
                            n_colors=256,
                            post_process='none', rgb_paths='', export_tif = False):
    """
    this function process all thermal pictures in a folder
    """
    # create raw outputs for each image
    for i, img_path in enumerate(ir_paths):
        print(i)
        if i < 9:
            prefix = '000'
        elif 9 < i < 99:
            prefix = '00'
        elif 99 < i < 999:
            prefix = '0'
        _, filename = os.path.split(str(img_path))
        dest_path = os.path.join(dest_folder, f'thermal_{prefix}{i}.JPG')

        if post_process == 'edge (from rgb)':
            process_one_th_picture(param, drone_model, img_path, dest_path, tmin, tmax, colormap, color_high,
                                   color_low, n_colors=n_colors, post_process=post_process, rgb_path=rgb_paths[i], export_tif = export_tif)
        else:
            process_one_th_picture(param, drone_model, img_path, dest_path, tmin, tmax, colormap, color_high,
                                   color_low, n_colors=n_colors, post_process=post_process, export_tif = export_tif)
        if i == len(ir_paths) - 1:
            legend_dest_path = os.path.join(dest_folder, 'plot_onlycbar_tight.png')
            generate_legend(legend_dest_path, tmin, tmax, color_high, color_low, colormap, n_colors)



