# coding=utf-8
import datetime
import os

import h5py as h5
import numpy as np
import pandas as pd


def readDatxFile(fileName):
    """
    This function reads the binary files (.datx) of HDX.
    Need to install the modules: numpy, PIL, pandas in the environment.
    In python3.7, windows10

    :param FileName: datx file path
    :return: Raw data as dictionary
    """

    import time

    d = {}

    # read the file
    with h5.File(fileName, 'r') as f:

        # GroupNames = [n for n in f.keys()]

        Attributes = f['Attributes']
        Attrs_keys = [n for n in Attributes.keys()]
        Att = dict(Attributes.get(Attrs_keys[1]).attrs)

        Dset = f['Data']

        # print(Dset.keys())

        if 'Intensity' in Dset.keys():
            IntensityDatasetNames = [n for n in Dset['Intensity'].keys()]
            Intensity = Dset['Intensity'].get(IntensityDatasetNames[0])
            Intensity_Attr_meta = dict(Intensity.attrs)
            # print(Intensity_Attr_meta)

        SurfaceDatasetNames = [n for n in Dset['Surface'].keys()]
        Surface = Dset['Surface'].get(SurfaceDatasetNames[0])
        Surface_Attr_meta = dict(Surface.attrs)

        SurfaceNoDataPixelValue = Surface_Attr_meta.get('No Data')[0]

        d['camera_width'] = (int(Att['Data Context.Data Attributes.Camera Width:Value'][0]),)
        d['camera_height'] = (int(Att['Data Context.Data Attributes.Camera Height:Value'][0]),)

        # Intensity file
        if 'Intensity' in Dset.keys():
            d['ac_org_x'] = Intensity_Attr_meta['Coordinates'][0][1]
            d['ac_org_y'] = Intensity_Attr_meta['Coordinates'][0][0]
            d['ac_width'] = Intensity_Attr_meta['Coordinates'][0][2]
            d['ac_height'] = Intensity_Attr_meta['Coordinates'][0][3]
            IntensityNoDataPixelValue = Intensity_Attr_meta.get('No Data')[0]
            d['data_intensity'] = np.array(Intensity)
            d['data_intensity'][d['data_intensity'] == IntensityNoDataPixelValue] = -1
            d['data_intensity'][d['data_intensity'] >= 0] = d['data_intensity'][d['data_intensity'] >= 0] * 256
            # d['data_intensity'] =np.reshape(d['data_intensity'],(d['ac_height'],d['ac_width']))
        else:
            d['ac_org_x'] = 0
            d['ac_org_y'] = 0
            d['ac_width'] = 0
            d['ac_height'] = 0
            d['data_intensity'] = []

        d['intf_scale_factor'] = Att['Data Context.Data Attributes.Interf Scale Factor:Value'][0]
        d['wavelength_in'] = Att['Data Context.Data Attributes.Wavelength:Value'][0]
        d['obliquity_factor'] = Surface_Attr_meta.get('Z Converter')[0][2][3]
        d['magnification'] = Att['Data Context.Data Attributes.Encoded Zoom Custom Magnification:Value'][0]
        try:
            d['CameraRes'] = Att['Surface Data Context.Lateral Resolution:Value'][0]  # Att['Data Context.Lateral Resolution:Value'][0]
        except:
            d['CameraRes'] = Att['Data Context.Lateral Resolution:Value'][0]

        # Phase Data Matrix
        d['cn_org_x'] = Surface_Attr_meta['Coordinates'][0][1]
        d['cn_org_y'] = Surface_Attr_meta['Coordinates'][0][0]
        d['cn_width'] = Surface_Attr_meta['Coordinates'][0][2]
        d['cn_height'] = Surface_Attr_meta['Coordinates'][0][3]

        d['data_pixel'] = np.array(Surface)
        # d['data_pixel'] = np.reshape(d['data_pixel'], [d['cn_height'], d['cn_width']])
        d['data_pixel'] = d['data_pixel'].astype('float')

        d['mask'] = np.ones_like(d['data_pixel'], dtype='bool')

        # for i in range(0,d['cn_height']):
        #     for j in range(0,d['cn_width']):
        for i in range(0, d['data_pixel'].shape[0]):
            for j in range(0, d['data_pixel'].shape[1]):
                if d['data_pixel'][i][j] >= SurfaceNoDataPixelValue:  # 2017151:#2097151
                    d['data_pixel'][i][j] = np.nan
                    d['mask'][i][j] = 0

                elif d['data_pixel'][i][j] <= -SurfaceNoDataPixelValue:  # 2017152:#2097152
                    d['data_pixel'][i][j] = np.nan
                    d['mask'][i][j] = 0
                else:
                    pass

        d['S'] = d['intf_scale_factor']
        d['W'] = d['wavelength_in']
        d['O'] = d['obliquity_factor']

        d['number_of_points_phase'] = d['cn_width'] * d['cn_height']
        d['waves'] = d['data_pixel'] * d['S'] * d['O']
        d['meters'] = d['data_pixel'] * d['W'] * d['S'] * d['O']

        d['height_nm'] = d['meters'] * 1e+9

        d['date_measurement'] = int(Att['Data Context.Data Attributes.Time Stamp'][0][0])
        d['number_day'] = d['date_measurement'] / (60 * 60 * 24)
        d['rest'] = d['date_measurement'] - d['number_day'] * 60 * 60 * 24
        d['hour_mesurement'] = d['rest'] / (60 * 60)
        d['rest'] = d['rest'] - d['hour_mesurement'] * 60 * 60
        d['minute_mesurement'] = d['rest'] / 60
        d['rest'] = d['rest'] - d['minute_mesurement'] * 60
        d['second_mesurement'] = d['rest']
        d['time_format'] = time.localtime(d['date_measurement'])
        d['date_str'] = datetime.datetime.fromtimestamp(d['date_measurement']).strftime('%Y-%m-%d %H:%M:%S')
        d['date_datToOpd'] = d['time_format'][0:10]
        d['time_datToOpd'] = d['time_format'][-8:]

        # np.savetxt("meters_dat.txt",meters)
        # Save a txt file in the same Folder with the code, the file shows height(meter) of every pixel, open it with Notepad, otherwise it changes the shape of the matrix.

        # np.savetxt("data_pixel_dat.txt",data_pixel)
        # Save a txt file in the same Folder with the code, the file shows phase value(more information of phase data is in the official instruction) of every pixel, open it with Notepad, otherwise it changes the shape of the matrix.        

        # np.savetxt("intensity_dat.txt",data_intensity)
        # Save a txt file in the same Folder with the code, the file shows intensity value(more information of intensity data is in the official instruction) of every pixel, open it with Notepad, otherwise it changes the shape of the matrix.        

        d['Max_value'] = np.nanmax(d['height_nm'])
        d['Min_value'] = np.nanmin(d['height_nm'])

        f.close()

    try:
        file_path = os.path.split(os.path.splitext(fileName)[0])[0]
        file_name = os.path.split(os.path.splitext(fileName)[0])[1]
        split_name = file_name.split('-P')
        aux_name = ''

        if len(split_name) > 2:
            for i in range(len(split_name) - 1):
                if not i:
                    aux_name = split_name[i]
                else:
                    aux_name = aux_name + '-P' + split_name[i]
        else:
            aux_name = split_name[0]

        motor_df_path = aux_name[:-1] + 'data.csv'
        sub_aperture_number = int(file_name.split('-P')[-1][:])
        motor_df = pd.read_csv(file_path + '//' + motor_df_path)
        d['motorX'] = motor_df[motor_df['Step_Count_X'] == sub_aperture_number - 1]['Motor_X_Pos'].values[0]
        d['motorRx'] = motor_df[motor_df['Step_Count_X'] == sub_aperture_number - 1]['Motor_ROLL_Pos'].values[0]
        d['motorRy'] = motor_df[motor_df['Step_Count_X'] == sub_aperture_number - 1]['Motor_PITCH_Pos'].values[0]
    except Exception as e:
        print('Motor df <- readDatx (Fizeau)')
        print(e)

    return d


def readDatxMatrixSize(fileName):
    """
    This function reads the binary files (.datx) of HDX and returns the size of the data matrix size.
    Need to install the modules: numpy, PIL in the environment.
    In python3.7, windows10

    :param FileName: datx file path
    :return: Raw data size as dictionary
    """

    d = {}

    # read the file
    with h5.File(fileName, 'r') as f:
        # GroupNames = [n for n in f.keys()]

        # Attributes = f['Attributes']
        # Attrs_keys = [n for n in Attributes.keys()]
        # Att = dict(Attributes.get(Attrs_keys[1]).attrs)

        Dset = f['Data']

        # print(Dset.keys())

        SurfaceDatasetNames = [n for n in Dset['Surface'].keys()]
        Surface = Dset['Surface'].get(SurfaceDatasetNames[0])
        Surface_Attr_meta = dict(Surface.attrs)

        # Phase Data Matrix
        d['cn_org_x'] = Surface_Attr_meta['Coordinates'][0][1]
        d['cn_org_y'] = Surface_Attr_meta['Coordinates'][0][0]
        d['cn_width'] = Surface_Attr_meta['Coordinates'][0][2]
        d['cn_height'] = Surface_Attr_meta['Coordinates'][0][3]

        f.close()

    return d
