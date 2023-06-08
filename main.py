""" Main entry point to the ThermalMesh application. """

# define authorship information
__authors__ = ['Samuel Dubois']
__author__ = ','.join(__authors__)
__credits__ = []
__copyright__ = 'Copyright (c) Buildwise 2023'
__license__ = ''

from multiprocessing import freeze_support


def main(argv=None):
    """
    Creates the main window for the application and begins the \
    QApplication if necessary.

    :param      argv | [, ..] || None

    :return      error code
    """

    # Define installation path
    install_folder = os.path.dirname(__file__)

    app = None

    # create the application if necessary
    if (not QtWidgets.QApplication.instance()):
        app = QtWidgets.QApplication(argv)
        app.setStyle('Breeze')

    # create the main window
    from gui.thermalmesh import ThermalWindow
    window = ThermalWindow()
    window.setWindowTitle('IR Mapper')
    window.setWindowIcon(QtGui.QIcon(res.find('img/icone.png')))
    window.show()

    # run the application if necessary
    if (app):
        return app.exec_()

    # no errors since we're not running our own event loop
    return 0


if __name__ == '__main__':
    freeze_support()  # To avoid pipinstaller bug. Note: all libraries are imported after freeze_support
    from PySide6 import QtWidgets, QtGui
    # from PySide6 import QtWebEngineWidgets
    import os
    import sys
    import resources as res

    sys.exit(main(sys.argv))
