# coding=utf-8
'''
Created Mar 21,2018

Class to convert the *.opd files generated by Veeco MSI instrument to *.h5 files. A measurement has the following files
1. Raw patches from different scans  (n scans x m subapertures)
2. Header metadata

@author: ADAPA
'''
import datetime
import os

import h5py
from numpy import nan, unicode
import six

import numpy as np
from PyLOSt.data_in.msi.read_opd import Read_opd_file
from PyLOSt.data_in.util_data import joinFileNameSequence


class OpdToH5():
    filePath = ''
    otype = 'a'
    entryName = ''

    # Data common to all scans in one measurement
    def __init__(self, outFilePath, otype='a', mName='MeasurementEntry', isNewData=False):
        """
        Initialize measurement entry and add metadata

        :param outFilePath: Output h5 file path
        :param otype: H5 file open types: default 'a'-read/write/create access
        :param mName: Measurement entry name
        :param isNewData: True if the current entry is created for the first time or to be overwritten
        :return:
        """
        self.filePath = outFilePath
        self.otype = otype
        self.entryName = mName
        self.h5f = h5py.File(outFilePath, otype)
        if otype == 'a' and isNewData:  # overwriting
            # Save attributes of file
            self.h5f.attrs[u'default'] = u'Data/height'
            self.h5f.attrs[u'version'] = u'1'
            self.h5f.attrs[u'file_name'] = outFilePath
            self.h5f.attrs[u'created_on'] = str(datetime.datetime.now())
            self.h5f.attrs[u'instrument'] = u'Veeco MSI'
            self.h5f.attrs[u'creator'] = u'opd_to_h5.py'
            self.h5f.attrs[u'HDF5_Version'] = six.u(h5py.version.hdf5_version)
            self.h5f.attrs[u'h5py_version'] = six.u(h5py.version.version)

            # Data entry
            self.entry = self.h5f.create_group(self.entryName)
            self.entry.attrs[u'NX_class'] = u'NXentry'
            self.entry.attrs[u'scan_type'] = u'line(1D)'
            self.entry.attrs[u'primary_axis'] = u'X'
            self.entry.attrs[u'secondary_axis'] = u''
            self.entry.attrs[u'flag_ref_subtracted'] = False
            self.entry.attrs[u'flag_gravity_corrected'] = False
            self.entry.attrs[u'flag_piston_corrected'] = False
            self.dataScans = self.entry.create_group('Data')
            self.dataScans.attrs[u'NX_class'] = u'NXdata'

            # meta data e.g. instrument, sample etc...
            self.metaData()
        self.h5f.close()

    @staticmethod
    def finish():
        ##TODO
        print('')

    # Load *.has patches in different scans
    def opdToH5(self, in_folder, prefix, patch_count, h5scan, dircn, scanNo, scanIdx=1, nbSelScans=1,
                patches_selected=[], options=['', '-', 'opd', []], progress_bar=None):
        """
        Function calls *.opd reader (read_opd.Read_opd_file) and saves the data to h5 measurement entry

        :param in_folder: Raw data folder
        :param prefix: File name prefix
        :param patch_count: Number of subapertures
        :param h5scan: Scan_xx group name
        :param dircn: Forward or backward
        :param scanNo: Current scan number
        :return:
        """
        try:
            self.h5f = h5py.File(self.filePath, self.otype)
            self.entry = self.h5f[self.entryName]
            self.dataScans = self.entry['Data']
            hgt_patches = []
            mask_patches = []
            intensity_patches = []
            mx_patches = []
            my_patches = []
            datetime_patches = []
            pidx = 0
            if not np.any(patches_selected):
                patches_selected = list(range(1, patch_count + 1))
            for j in (reversed(patches_selected) if dircn == 'B' else patches_selected):
                pidx = pidx + 1
                if progress_bar:
                    progress_bar.setValue(80 * (pidx / len(patches_selected)) * (scanIdx / nbSelScans))
                fin_path = os.path.join(in_folder, joinFileNameSequence(prefix, scanNo, j, options))
                d = Read_opd_file(fin_path)

                h = d['height']
                mask = np.ones_like(h, dtype='bool')
                hgt_patches.append(h)
                mask_patches.append(mask)

                # motor positions
                mx_patches.append(d['StageX'] if 'StageX' in d else nan)
                my_patches.append(d['StageY'] if 'StageY' in d else nan)

                datetime_patches.append(d['Date'] if 'Date' in d else u'')

                # temp
                # h[h>2e7]=np.nan

            data_shape = np.array(hgt_patches).shape
            # slopes in mrad        
            data = self.dataScans.create_group(h5scan)
            data.attrs[u'NX_class'] = u'NXdata'
            data.attrs[u'signal'] = u'height'
            data.attrs[u'axes'] = np.array([u'.', u'.', u'.'], dtype=h5py.special_dtype(vlen=unicode))
            data.attrs[u'name'] = h5scan
            data.attrs[u'scan_number'] = scanNo
            data.attrs[u'scan_direction'] = dircn
            data.attrs[u'pupil_data_dimensions'] = (d['rows'], d['cols'])
            data.attrs[u'data_dimensions'] = data_shape
            data.attrs[u'data_dimensions_format'] = u'[subaperture count, width pixels, length pixels]'
            data.attrs[u'data_step'] = [d['Pixel_size'], d['Pixel_size']]  # x,y directions
            data.attrs[u'data_step_units'] = u'mm'
            # data.attrs[u'temperature']                  = [] # temp array here or in NXmeasruement?

            data.create_dataset('height', shape=data_shape, dtype='float32', chunks=True)
            data.create_dataset('mask', shape=data_shape, dtype='bool_', chunks=True)
            data['height'][...] = hgt_patches
            data['height'].attrs[u'units'] = u'nm'
            data['height'].attrs[u'scale'] = u'1'
            data['mask'][...] = mask_patches
            if progress_bar:
                progress_bar.setValue(90 * (scanIdx / nbSelScans))

            # motor positions in mm or deg
            data['motor_X'] = np.array(mx_patches).astype(np.float)
            data['motor_X'].attrs[u'units'] = u'mm'
            data['motor_Y'] = np.array(my_patches).astype(np.float)
            data['motor_Y'].attrs[u'units'] = u'mm'

            data['timestamps'] = np.string_(datetime_patches)
            data['timestamps'].attrs[u'format'] = u'DD/MM/YYYY'

            if 'Data/Ref1' in self.h5f:
                data['ref'] = h5py.SoftLink('Data/Ref1')

            if scanNo == 1:
                self.updateMetaData(data)  # update only once
            if progress_bar:
                progress_bar.setValue(100 * (scanIdx / nbSelScans))
            self.h5f.close()
        except Exception as e:
            print('opdToH5 <- OpdToH5')
            print(e)

    def addReference(self, f, loc='Data/Ref1'):
        """
        Add reference to measurement entry

        :param f: Reference file path (opd format)
        :param loc: Relative location of reference under measurement entry
        :return:
        """
        self.h5f = h5py.File(self.filePath, self.otype)
        self.entry = self.h5f[self.entryName]
        if loc in self.h5f:
            del self.h5f[loc]

        d = Read_opd_file(f)
        h = d['height']
        mask = np.ones_like(h, dtype='bool')

        ref = self.entry.create_group(loc)
        ref.attrs[u'timestamp'] = u''
        ref['height'] = h
        ref['height'].attrs[u'units'] = u'nm'
        ref['height'].attrs[u'scale'] = u'1'
        ref['mask'] = mask
        ref['intensity'] = d['data_intensity']
        self.h5f.close()

    def updateScanRef(self, utype='all', refLoc='Data/Ref1', scanStart=0, scanEnd=0):
        """
        Reference is linked to scans (either all scans or between start/end scans)

        :param utype: if utype is 'all', reference is linked to all scans
        :param refLoc: Relative path to reference under measurement entry
        :param scanStart: Start scan number. It is used if utype is not 'all'
        :param scanEnd: End scan number. It is used if utype is not 'all'
        :return:
        """
        self.h5f = h5py.File(self.filePath, self.otype)
        self.entry = self.h5f[self.entryName]
        h5scans = self.entry['Data']
        for i in h5scans.keys():
            if 'NX_class' in h5scans[i].attrs and h5scans[i].attrs['NX_class'] == 'NXdata':  # loop over scans
                if utype == 'all':
                    self.createRefLink(h5scans[i], refLoc)
                else:
                    if h5scans[i].attrs['scan_number'] >= scanStart and h5scans[i].attrs['scan_number'] <= scanEnd:
                        self.createRefLink(h5scans[i], refLoc)
        self.h5f.close()

    def createRefLink(self, h5si, refLoc):
        """
        Create hard link to reference data

        :param h5si: Scan_xx group
        :param refLoc: Reference location
        :return:
        """
        self.h5f = h5py.File(self.filePath, self.otype)
        self.entry = self.h5f[self.entryName]
        if 'ref' in h5si:
            del h5si['ref']
        h5si['ref'] = self.entry[refLoc]  # softlink not working:: h5py.SoftLink(refLoc)
        self.h5f.close()

    def updateMetaData(self, data=None, cntArr=None, instr_id=None, instr_location=None):
        """
        Update meta data

        :param data: Data group in h5
        :param cntArr: Scan/subaperture count array
        :param instr_id: Instrument id
        :param instr_location: Instrument physical location
        :return:
        """
        self.h5f = h5py.File(self.filePath, self.otype)
        self.entry = self.h5f[self.entryName]
        if instr_id:
            self.entry['Instrument'].attrs[u'instr_id'] = instr_id
        if instr_location:
            self.entry['Instrument'].attrs[u'instr_location'] = instr_location
        if data:
            self.entry['Instrument/resolution'][...] = data.attrs[u'data_step']
            self.entry['Instrument/pupil_data_dimensions'][...] = data.attrs[u'pupil_data_dimensions']
        if cntArr:
            self.entry['Measurement/scan_count'][...] = cntArr[1]
            self.entry['Measurement/subaperture_count'][...] = cntArr[0]
        self.h5f.close()

    def metaData(self):
        """
        Meta data such as measurement conditions, sample details etc...

        :return:
        """
        h5instr = self.entry.create_group('Instrument')
        h5instr.attrs[u'NX_class'] = u'NXinstrument'
        h5instr.attrs[u'wavelength'] = u'537.3 nm'
        h5instr.attrs[u'conditions'] = 'external fan'
        h5instr['name'] = 'Veeco MSI @ ESRF'
        h5instr['version'] = 'v1.0.1'
        h5instr['pupil_dimensions'] = [0.0, 0.0]  # mm
        h5instr['pupil_dimensions'].attrs[u'units'] = u'mm'
        h5instr['pupil_dimensions'].attrs[u'format'] = u'Y, X'
        h5instr['pupil_data_dimensions'] = [640, 480]  # no of pixels
        h5instr['pupil_data_dimensions'].attrs[u'units'] = u'# of pixels'
        h5instr['resolution'] = [1.0, 1.0]  # mm pixel size
        h5instr['resolution'].attrs[u'units'] = u'mm'
        h5instr['resolution'].attrs[u'format'] = u'X/Y'
        h5instr['zoom'] = 1
        h5instr['scale_factor'] = 1
        h5instr['scale_factor'].attrs[u'units'] = u'nm'
        h5instr['coordinate_system'] = str({u'X': 1, u'Y': 1, u'Z': 1, u'Rx': 1, u'Ry': 1, u'Rz': 1})
        h5instr['coordinate_system'].attrs[
            'info'] = '+ve if instr coordinates (motor positions) increase from subapertures start to end \n and 1 if no scaling '

        mask = np.full((640, 480), True)
        h5instr['mask'] = mask

        h5msr = self.entry.create_group('Measurement')
        h5msr.attrs[u'NX_class'] = u'NXmeasurement'
        h5msr['scan_count'] = 0
        h5msr['scan_count'].attrs[u'format'] = u'nb scans'
        h5msr['subaperture_count'] = 0
        h5msr['temperature'] = []
        h5msr['detector_to_instr_coord'] = str({u'X': 1, u'Y': 1, u'Z': 1, u'Rx': 1, u'Ry': 1, u'Rz': 1})
        h5msr['instr_to_sample_coord'] = str({u'X': 1, u'Y': 1, u'Z': 1, u'Rx': 1, u'Ry': 1, u'Rz': 1})
        h5msr['type'] = 'AB scan'

        h5sample = self.entry.create_group('Sample')
        h5sample.attrs[u'NX_class'] = u'NXsample'
        h5sample['name'] = ''
        h5sample['no_of_regions'] = 1
        h5sample['shape'] = 'plane'  # for multiple regions, an array sequence
        h5sample['material'] = 'uncoated'
        h5sample['material_bulk'] = 'Si'
        h5sample['total_dimensions'] = []
        h5sample['total_dimensions'].attrs[u'units'] = u'mm'
        h5sample['total_dimensions'].attrs[u'format'] = u'[length, width, height]'
        h5sample['regions_of_interest'] = []
        h5sample['regions_of_interest'].attrs[u'units'] = u'mm'
        h5sample['regions_of_interest'].attrs[u'format'] = u'[[X1,X2,Y1,Y2],...]'
        h5sample['orientation'] = 'up/side/down'
        h5sample['comments'] = '\n mirror placed on cyl rods at pos X=-L/4 & X=L/4 from center, no bender'
        h5sample['params'] = 'Rc/p-q-theta/Major-minor'
        h5sample['params'].attrs[u'units'] = u'm/m-m-mrad/m-m'

        # h5users                                             = self.h5f['Users'] if 'Users' in self.h5f else self.h5f.create_group('Users')
        # h5users.attrs[u'NX_class']                          = u'NXuser'
        # if 'Mercury' not in h5users:
        #     h5u                                                 = h5users.create_group('Mercury')
        #     h5u.attrs[u'NX_class']                              = u'NXuser'
        #     h5u['full_name']                                    = 'Mercury'
        #     h5u['email']                                        = ''
        #     h5u['location']                                     = ''
        #
        # self.entry['User']                                  = h5users['Mercury']

        h5u = self.entry.create_group('User')
        h5u.attrs[u'NX_class'] = u'NXuser'
        h5u['full_name'] = 'user_name'
        h5u['email'] = ''
        h5u['location'] = ''
