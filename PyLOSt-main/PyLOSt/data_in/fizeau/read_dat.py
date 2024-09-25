# coding=utf-8
import datetime
import re
from builtins import filter


def readDatFile(fileName):
    """
    This function reads the binary files (.dat) of metropro.
    Need to install the modules: numpy, PIL in the environment.
    In python3.5, windows7

    :param FileName: dat file path
    :return: Raw data as dictionary
    """
    import struct
    import numpy as np
    import time

    d = {}
    try:
        # read the file
        with open(fileName, "rb") as fid:
            badpixelValue = np.nan  # 2147483640

            # Header file
            fid.seek(0)
            magicNumber_encode = fid.read(4)
            headerFormat_encode = fid.read(2)
            headerSize_encode = fid.read(4)
            swinfoDate_encode = fid.read(30)

            # Intensity file
            fid.seek(48)
            ac_org_x_encode = fid.read(2)
            fid.seek(50)
            ac_org_y_encode = fid.read(2)
            ac_width_encode = fid.read(2)
            ac_height_encode = fid.read(2)
            ac_n_buckets_encode = fid.read(2)
            ac_range_encode = fid.read(2)
            ac_n_bytes_encode = fid.read(4)
            fid.seek(164)
            intf_scale_factor_encode = fid.read(4)

            wavelength_in_encode = fid.read(4)
            fid.seek(176)
            obliquity_factor_encode = fid.read(4)
            magnification_encode = fid.read(4)
            CameraRes_encode = fid.read(4)

            # Phase Data Matrix
            fid.seek(64)
            cn_org_x_encode = fid.read(2)
            cn_org_y_encode = fid.read(2)
            cn_width_encode = fid.read(2)
            cn_height_encode = fid.read(2)
            cn_n_bytes_encode = fid.read(4)
            fid.seek(76)
            date_measurement_encode = fid.read(4)

            fid.seek(80)
            comments = fid.read(82)
            try:
                # Current ESRF comments format 'Position= 1  X-coord= 32.75  X-shift= 0.929'
                comments = comments.decode("utf-8")
                comArr = re.split('[= ]', comments)
                comArrFilt = list(filter(bool, comArr))
                d['motorX'] = np.asarray(-1 * np.double(comArrFilt[3])) if comArrFilt[
                                                                               2] == 'X-coord' else np.nan  # if comments format changes??
                d['motorXShift'] = np.asarray(-1 * np.double(comArrFilt[5])) if comArrFilt[
                                                                                    4] == 'X-shift' else np.nan  # if comments format changes??
            except Exception as e:
                print('comments section <- readDat (Fizeau)')
                print(e)

            fid.seek(218)
            phase_res_encode = fid.read(2)

            fid.seek(234)
            camera_width_encode = fid.read(2)
            camera_height_encode = fid.read(2)

            fid.seek(486)
            pixel_width_encode = fid.read(4)
            pixel_height_encode = fid.read(4)

            d['camera_width'] = struct.unpack(">h", camera_width_encode)[0]
            d['camera_height'] = struct.unpack(">h", camera_height_encode)[0]

            # Decode
            # header file
            d['magicNumber'] = struct.unpack(">l", magicNumber_encode)[0]
            d['headerFormat'] = struct.unpack(">h", headerFormat_encode)[0]
            d['headerSize'] = struct.unpack(">l", headerSize_encode)[0]

            # Intensity file
            d['ac_org_x'] = struct.unpack(">h", ac_org_x_encode)[0]
            d['ac_org_y'] = struct.unpack(">h", ac_org_y_encode)[0]
            d['ac_width'] = struct.unpack(">h", ac_width_encode)[0]
            d['ac_height'] = struct.unpack(">h", ac_height_encode)[0]
            d['ac_n_buckets'] = struct.unpack(">h", ac_n_buckets_encode)[0]
            d['ac_range'] = struct.unpack(">h", ac_range_encode)[0]
            d['ac_n_bytes'] = struct.unpack(">l", ac_n_bytes_encode)[0]
            d['intf_scale_factor'] = struct.unpack(">f", intf_scale_factor_encode)[0]
            d['wavelength_in'] = struct.unpack(">f", wavelength_in_encode)[0]
            d['obliquity_factor'] = struct.unpack(">f", obliquity_factor_encode)[0]
            d['magnification'] = struct.unpack(">f", magnification_encode)[0]
            d['CameraRes'] = struct.unpack(">f", CameraRes_encode)[0]

            # Phase Data Matrix
            d['cn_org_x'] = struct.unpack(">h", cn_org_x_encode)[0]
            d['cn_org_y'] = struct.unpack(">h", cn_org_y_encode)[0]
            d['cn_width'] = struct.unpack(">h", cn_width_encode)[0]
            d['cn_height'] = struct.unpack(">h", cn_height_encode)[0]
            d['cn_n_bytes'] = struct.unpack(">l", cn_n_bytes_encode)[0]
            d['phase_res'] = struct.unpack(">h", phase_res_encode)[0]

            d['pixel_width'] = struct.unpack(">f", pixel_width_encode)[0]
            d['pixel_height'] = struct.unpack(">f", pixel_height_encode)[0]

            d['number_of_points'] = d['ac_width'] * d['ac_height'] * d['ac_n_buckets']

            # header size can be different
            fid.seek(d['headerSize'])
            d['way'] = "<" + str(d['number_of_points']) + "h"
            # only if intensity data present
            if d['number_of_points'] != 0:
                data_intensity_encode = fid.read(d['number_of_points'] * 2)  # int16
                d['data_intensity'] = struct.unpack(d['way'], data_intensity_encode)
                d['data_intensity'] = np.asarray(d['data_intensity'])
                d['data_intensity'] = np.reshape(d['data_intensity'], (d['ac_height'], d['ac_width']))
            else:  # no intensity data, create empty matrix
                d['data_intensity'] = np.asarray([])

            # header size can be different
            fid.seek(d['headerSize'] + d['number_of_points'] * 2)
            # not reading full height # XXX: why?
            # d['cn_height'] = d['cn_height'] - 1
            data_point_encode = fid.read(d['cn_width'] * d['cn_height'] * 4)  # float32
            d['decode_way'] = ">" + str(d['cn_width'] * d['cn_height']) + "i"
            d['data_point'] = struct.unpack(d['decode_way'], data_point_encode)

            d['data_pixel'] = np.asarray(d['data_point'])
            d['data_pixel'] = np.reshape(d['data_pixel'], [d['cn_height'], d['cn_width']])
            d['data_pixel'] = d['data_pixel'].astype('float32')  # in float32

            d['mask'] = d['data_pixel'] < 2e9

            # little comment: numpy as some optimized functions to perform the same test
            # see: https://numpy.org/devdocs/reference/generated/numpy.where.html
            # with this way we gain almost a speed factor 8 on large dataset (X4 without the test on negative values)
            d['data_pixel'] = np.where(d['mask'],
                                       d['data_pixel'],
                                       badpixelValue)
            # for i in range(0,d['cn_height']):
            #     for j in range(0,d['cn_width']):
            #         if d['data_pixel'][i][j]>2e9: # invalid pixel is set at 2147483640
            #             d['data_pixel'][i][j] = badpixelValue
            #             d['mask'][i][j] = 0

            #         # elif d['data_pixel'][i][j]<-2e9: # XXX: is this test necessary ?
            #         #     d['data_pixel'][i][j] =badpixelValue
            #         #     d['mask'][i][j] = 0
            #         # else:
            #         #     pass

            d['S'] = d['intf_scale_factor']
            d['W'] = d['wavelength_in']
            d['O'] = d['obliquity_factor']
            d['R'] = 32768  # reduce to int16 scale

            d['number_of_points_phase'] = d['cn_width'] * d['cn_height']
            d['waves'] = d['data_pixel'] * d['S'] * d['O'] / d['R']
            d['height_m'] = d['waves'] * d['W']

            d['height_nm'] = d['height_m'] * 1e+9

            d['date_measurement'] = struct.unpack(">i", date_measurement_encode)[0]
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

            # np.savetxt("height_m_dat.txt",height_m)
            # Save a txt file in the same Folder with the code, the file shows height(meter) of every pixel, open it with Notepad, otherwise it changes the shape of the matrix.

            # np.savetxt("data_pixel_dat.txt",data_pixel)
            # Save a txt file in the same Folder with the code, the file shows phase value(more information of phase data is in the official instruction) of every pixel, open it with Notepad, otherwise it changes the shape of the matrix.

            # np.savetxt("intensity_dat.txt",data_intensity)
            # Save a txt file in the same Folder with the code, the file shows intensity value(more information of intensity data is in the official instruction) of every pixel, open it with Notepad, otherwise it changes the shape of the matrix.

            d['Max_value'] = np.nanmax(d['height_nm'])
            d['Min_value'] = np.nanmin(d['height_nm'])

            # d['height_nm'] = np.flip(d['height_nm'], axis=-2)  # Flip along Y
            # d['height_m'] = np.flip(d['height_m'], axis=-2)  # Flip along Y
            # d['waves'] = np.flip(d['waves'], axis=-2)  # Flip along Y
            # d['data_pixel'] = np.flip(d['data_pixel'], axis=-2)  # Flip along Y
            # d['mask'] = np.flip(d['mask'], axis=-2)  # Flip along Y
            # d['data_intensity'] = np.flip(d['data_intensity'], axis=-2) if d['data_intensity'].ndim>1 else d['data_intensity'] # Flip along Y
    except Exception as e:
        print('readDatFile : ')
        print(e)

    return d
