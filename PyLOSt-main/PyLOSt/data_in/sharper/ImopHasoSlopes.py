# coding=utf-8
'''
Created on Mar 22, 2018

@author: adapa
'''
## Wrapper around Imop_Haso_Slopes
#  Loads SlopesX, SlopesY, Intensity and Pupil data into matlab memory

import re
import numpy as np

# import xml.etree.ElementTree as ET
import xml.etree.cElementTree as ET
from PyLOSt.util import commons

DEG_TO_MRAD = 17.4533


class ImopHasoSlopes:
    # variables
    slopesObj = None
    dimensions = None
    steps = None
    serial_number = None
    isCreatedLocally = False
    HasoData = None

    slopes_x = None
    slopes_y = None
    pupil_data = None
    intensity = None

    motorX = None
    motorY = None
    motorTz = None
    motorRx = None
    motorRy = None
    motorRz = None

    time_stamp = None

    # 1 - Slopes Object
    # 5 - slopes_x, slopes_y, dimensions, steps, serial_number
    def __init__(self, otype, readXML=False, fname=''):
        """
        Reads *.has file as XML

        :param otype: object type
        :param readXML:
        :param fname:
        """
        try:
            if (otype == 'Wrap'):
                self.fname = fname
                if readXML:
                    self.loadXML()
        except Exception as e:
            print('ImopHasoSlopes->__init__()')
            print(e)

    def loadXML(self):
        try:
            with open(self.fname) as f:
                xml = f.read()
            tree = ET.fromstring(re.sub(r"(<\?xml[^>]+\?>)", r"\1<root>", xml) + "</root>")
            root = tree.find('haso_slopes_process_manager')
            # tree                = ET.parse(self.fname)
            # root                = tree.getroot()
            raw_slopes = root.find('raw_slopes')
            metadata = raw_slopes.find('metadata')
            slopes = raw_slopes.find('slopes')

            self.dimensions = commons.uint2D()
            self.dimensions.X = int(slopes.find('size').find('x').text)
            self.dimensions.Y = int(slopes.find('size').find('y').text)
            self.oshape = (self.dimensions.Y, self.dimensions.X)
            self.steps = commons.float2D()
            self.steps.X = float(slopes.find('step').find('x').text)
            self.steps.Y = float(slopes.find('step').find('y').text)
            self.slopes_x = -0.5 * np.fromstring(slopes.find('x_slopes').find('buffer').text, dtype='float32',
                                                 sep='\t').reshape(self.oshape)
            self.slopes_y = -0.5 * np.fromstring(slopes.find('y_slopes').find('buffer').text, dtype='float32',
                                                 sep='\t').reshape(self.oshape)
            self.pupil_data = np.fromstring(slopes.find('pupil').find('buffer').text, dtype='bool', sep='\t').reshape(
                self.oshape)
            self.intensity = np.fromstring(slopes.find('intensity').find('buffer').text, dtype='uint32',
                                           sep='\t').reshape(self.oshape)

            self.serial_number = metadata.find('haso_serial_number').find('crc').text
            self.comments = metadata.find('comments').text
            self.parseComments()
            if metadata.find('acquisition_info'):
                self.time_stamp = metadata.find('acquisition_info').find('acquisition_date').text
                self.exposure_time_us = float(
                    metadata.find('acquisition_info').find('exposure_time_us').find('state').text)
                self.nb_summed_images = int(metadata.find('acquisition_info').find('nb_summed_images').text)
                self.background_removed = metadata.find('acquisition_info').find('background_removed').text
                self.trigger_mode = metadata.find('acquisition_info').find('trigger_mode').text

        except Exception as e:
            print('ImopHasoSlopes->loadXML()')
            print(e)

    def parseComments(self):
        try:
            commArr = re.split('[;,]', self.comments)
            for p in commArr:
                if re.search(r'xpos:', p) is not None:
                    self.motorX = float(re.search(r'xpos:\[\[(.*?)\]\]', p).group(1))

                if re.search('=', p) is not None:
                    dataStr = re.split('[=]', p)[1].strip()
                    data = float(re.search(r'\[\[(.*?)\]\]', dataStr).group(1)) if dataStr.startswith('[[') else float(
                        dataStr)

                    if re.search('X \(tool\)', p) is not None:
                        self.motorY = data
                    elif re.search('X =', p) is not None or re.search('X=', p) is not None:
                        if re.search('tiltX=', p) is None and re.search('RX', p.upper()) is None:
                            self.motorX = data
                    if re.search('Y \(Scan\)', p) is not None:
                        self.motorX = data
                    elif re.search('Y =', p) is not None or re.search('Y=', p) is not None:
                        if re.search('tiltY=', p) is None and re.search('RY', p.upper()) is None:
                            self.motorY = data

                    if re.search('RX', p.upper()) is not None or re.search('RX \(tool\)', p) is not None:
                        self.motorRx = data * DEG_TO_MRAD
                    if re.search('RY', p.upper()) is not None or re.search('RY \(tool\)', p) is not None:
                        self.motorRy = data * DEG_TO_MRAD
                    if re.search('RZ', p.upper()) is not None or re.search('RZ \(tool\)', p) is not None:
                        self.motorRz = data * DEG_TO_MRAD
                    if re.search('TZ', p.upper()) is not None or re.search('TZ \(tool\)', p) is not None:
                        self.motorTz = data
        except Exception as e:
            print('ImopHasoSlopes->parseComments()')
            print(e)
