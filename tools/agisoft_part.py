import os
import resources as res
from PySide6 import QtCore, QtGui, QtWidgets
import subprocess
import sys
import pkg_resources


def install_agisoft_module():
    # install Metashape module if necessary
    def install(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    metashape_module = res.find('other/Metashape-2.0.1-cp37.cp38.cp39.cp310.cp311-none-win_amd64.whl')
    install(metashape_module)

# check if module is installed
required = {'metashape'}
installed = {pkg.key for pkg in pkg_resources.working_set}
print(installed)
missing = required - installed
if missing:
    print(r"Ok let's intall Agisoft!")
    install_agisoft_module()

import Metashape

# general parameters
DOWN_SCALE = 2 # downscale factor for image alignment

# paths
basepath = os.path.dirname(__file__)
RGB_IMG_FOLDER = os.path.join(basepath, 'test_dataset', 'rgb')
IR_IMG_FOLDER = os.path.join(basepath, 'test_dataset', 'thermal')

"""
RUNNERS_________________________________________________________________________________________________________
"""


class RunnerSignals(QtCore.QObject):
    progressed = QtCore.Signal(int)
    messaged = QtCore.Signal(str)
    finished = QtCore.Signal()


class RunnerAgisoft(QtCore.QRunnable):
    def __init__(self, drone_model, ir_list, rgb_list, output_folder, start, stop, do_precleaning=True, quality='medium',
                 opt_make_mesh=True, nb_text= 1, text_size=8192, opt_limit_points=0, opt_limit_poly=0,
                 opt_make_ortho=False, opt_make_pdf=False):
        super().__init__()
        self.signals = RunnerSignals()
        self.ir_list = ir_list
        self.rgb_list = rgb_list
        self.output_folder = output_folder
        self.start = start
        self.stop = stop
        self.quality = quality
        self.text_size = text_size
        self.do_precleaning = do_precleaning
        self.opt_make_mesh = opt_make_mesh
        self.opt_make_ortho = opt_make_ortho
        self.opt_make_pdf = opt_make_pdf
        self.drone_model = drone_model
        self.nb_text = nb_text
        self.opt_limit_points = opt_limit_points
        self.opt_limit_poly = opt_limit_poly

    def run(self):
        print('go')
        psx_path = os.path.join(self.output_folder, 'agisoft.psx')
        pc_path = os.path.join(self.output_folder, 'thermal_point_cloud.ply')
        mesh_path = os.path.join(self.output_folder, 'thermal_mesh.obj')
        ortho_path = os.path.join(self.output_folder, 'thermal_ortho.tif')
        pdf_path = os.path.join(self.output_folder, 'thermal_document.pdf')

        # drone model specific data
        if self.drone_model == 'MAVIC2-ENTERPRISE-ADVANCED':
            calib_file = res.find('other/camera_calib_m2t.xml')
        elif self.drone_model == 'M3T':
            calib_file = res.find('other/camera_calib_m3t.xml')

        # compute number of steps

        def pre_cleaning(chk):
            reperr = 0.3
            recunc = 35
            projacc = 10

            f = Metashape.TiePoints.Filter()
            f.init(chk, Metashape.TiePoints.Filter.ReprojectionError)
            f.removePoints(reperr)
            f.init(chk, Metashape.TiePoints.Filter.ReconstructionUncertainty)
            f.removePoints(recunc)
            f.init(chk, Metashape.TiePoints.Filter.ProjectionAccuracy)
            f.removePoints(projacc)
            chk.optimizeCameras()

        step = (self.stop - self.start) / 100
        self.signals.progressed.emit(self.start + step)
        self.signals.messaged.emit('Agisoft reconstruction started')

        # create agisoft document
        doc = Metashape.Document()
        doc.save(path=psx_path)
        # create chunk
        chk = doc.addChunk()
        # add the ir and rgb pictures
        #   loading images

        images = [None] * (len(self.rgb_list) + len(self.ir_list))
        images[::2] = self.ir_list
        images[1::2] = self.rgb_list
        print(f'Here is images struct : {images}')
        # transform to abs path:

        for i, bad_path in enumerate(images):
            images[i] = os.path.normpath(images[i])

        filegroups = [2] * (len(images) // 2)
        # images is alternating list of rgb, ir paths
        # filegroups defines the multi-camera system groups: [2, 2, 2, ....]

        chk.addPhotos(filenames=images, filegroups=filegroups, layout=Metashape.MultiplaneLayout, load_xmp_accuracy=True)
        print('photos added!')

        # check master
        for sensor in chk.sensors:
            print(sensor.label)
            if sensor == sensor.master:
                continue
            print(sensor.label)

        chk.sensors[0].makeMaster()

        # temporarily desactivate thermal images
        for camera in chk.cameras:
            if camera.sensor != sensor.master:
                print(f'here is a thermal image! : {camera.label}')
                camera.enabled = False

        self.signals.progressed.emit(self.start + 5 * step)
        self.signals.messaged.emit('Aligning cameras')

        # align photos (based on rgb data)
        chk.matchPhotos(guided_matching=True, downscale=DOWN_SCALE)
        chk.alignCameras()

        self.signals.progressed.emit(self.start + 15 * step)
        self.signals.messaged.emit('Building depth maps')

        if self.do_precleaning:
            pre_cleaning(chk)

        doc.save(path=psx_path)

        # depth maps
        if self.quality == 'low':
            quality_factor = Metashape.LowFaceCount
            downscale_factor = 8
        elif self.quality == 'medium':
            quality_factor = Metashape.MediumFaceCount
            downscale_factor = 4
        elif self.quality == 'high':
            quality_factor = Metashape.HighFaceCount
            downscale_factor = 2
        chk.buildDepthMaps(downscale=downscale_factor)
        doc.save(path=psx_path)

        # import intrinsic parameters (IR camera)
        user_calib = Metashape.Calibration()
        user_calib.load(calib_file)  # dependent on drone model

        # load calib to thermal sensor
        chk.sensors[1].user_calib = user_calib
        chk.sensors[1].fixed = True

        # ------------------------ OUTPUTS STAGE --------------------------------------
        # RGB -----------------
        # build point cloud
        self.signals.progressed.emit(self.start + 35 * step)
        self.signals.messaged.emit('Building RGB point cloud')



        # reduce size if needed
        chk.buildPointCloud(point_confidence=True)
        if self.opt_limit_points > 0:
            chk.filterPointCloud(point_spacing=self.opt_limit_points)
        doc.save(path=psx_path)

        # filter by confidence TODO: make it work!
        """
        print(chk.point_clouds)
        chk.point_cloud.setConfidenceFilter(0, 3)
        chk.point_cloud.selectPointsByColor([155, 155, 155], tolerance=99)
        chk.point_cloud.assignClassToSelection(target=1)
        chk.dense_cloud.removePoints([1])
        """

        # export RGB point cloud
        chk.exportPointCloud(path=pc_path[:-4] + '_rgb.ply', format=Metashape.PointCloudFormatPLY,
                             crs=Metashape.CoordinateSystem(
                                 'LOCAL_CS["Local CS",LOCAL_DATUM["Local Datum",0],UNIT["metre",1]]'))

        if self.opt_make_mesh:
            # model
            self.signals.progressed.emit(self.start + 45 * step)
            self.signals.messaged.emit('Building mesh')

            chk.buildModel(source_data=Metashape.DataSource.DepthMapsData, face_count=quality_factor)
            if self.opt_limit_poly > 0:
                chk.decimateModel(face_count=self.opt_limit_poly)

            # uv
            self.signals.progressed.emit(self.start + 55 * step)
            self.signals.messaged.emit('Building UV')
            chk.buildUV(mapping_mode=Metashape.GenericMapping, page_count=self.nb_text, texture_size=self.text_size)

            self.signals.progressed.emit(self.start + 65 * step)
            self.signals.messaged.emit('Building RGB Texture')
            # build rgb texture
            chk.buildTexture()

            # save model
            chk.exportModel(path=mesh_path, precision=9, save_texture=True, save_uv=True, save_markers=False,
                            crs=Metashape.CoordinateSystem(
                                'LOCAL_CS["Local CS",LOCAL_DATUM["Local Datum",0],UNIT["metre",1]]'))

            # get texture and change its name
            os.rename(mesh_path[:-4] + '.jpg', mesh_path[:-4] + '_rgb.jpg')

        # INFRARED -----------------
        # change layer index
        chk.sensors[0].layer_index = 1

        # reactivate thermal images
        for camera in chk.cameras:
            if camera.sensor != sensor.master:
                print(f'here is a thermal image! : {camera.label}')
                camera.enabled = True

        # change master
        chk.sensors[1].makeMaster()

        # colorize point cloud
        self.signals.progressed.emit(self.start + 75 * step)
        self.signals.messaged.emit('Colorizing point cloud')
        chk.colorizePointCloud()

        # export point cloud
        chk.exportPointCloud(path=pc_path, format=Metashape.PointCloudFormatPLY, crs=Metashape.CoordinateSystem(
            'LOCAL_CS["Local CS",LOCAL_DATUM["Local Datum",0],UNIT["metre",1]]'))

        if self.opt_make_mesh:
            # build thermal texture
            self.signals.progressed.emit(self.start + 85 * step)
            self.signals.messaged.emit('Building thermal texture')
            chk.buildTexture()

            # export model
            chk.exportModel(path=mesh_path, precision=9, save_texture=True, save_uv=True, save_markers=False,
                            crs=Metashape.CoordinateSystem(
                                'LOCAL_CS["Local CS",LOCAL_DATUM["Local Datum",0],UNIT["metre",1]]'))

            if self.opt_make_ortho:
                chk.buildOrthomosaic(surface_data=Metashape.ModelData, resolution=0.02)
                chk.exportRaster(ortho_path)

            if self.opt_make_pdf:
                # decimate mesh
                try:
                    chk.decimateModel(face_count=20000)
                    chk.buildUV(mapping_mode=Metashape.GenericMapping, page_count=1, texture_size=4096)
                    chk.buildTexture()
                except:
                    pass

                # save pdf
                chk.exportModel(path=pdf_path, precision=9, save_texture=True, save_uv=True, save_markers=False,
                                crs=Metashape.CoordinateSystem(
                                    'LOCAL_CS["Local CS",LOCAL_DATUM["Local Datum",0],UNIT["metre",1]]'))


            self.signals.progressed.emit(self.start + 100 * step)
            self.signals.messaged.emit('Finished')

        self.signals.finished.emit()


